# Human-on-the-Loop Pipeline Demo — Design

**Date:** 2026-07-14
**Status:** Approved (rev 4 — prompts externalized as markdown+frontmatter+Jinja2; deep_analysis explores repos via tools)
**Stack:** Python ≥3.10, Poetry, pytest, Microsoft Agent Framework (`agent-framework`, `agent-framework-ollama`), Ollama `gemma4:31b` (local; tools + thinking capable, 262k context — verified via `ollama show`)

## 1. Purpose

Demo project showcasing **human-on-the-loop (HOTL)** capability on Microsoft Agent Framework: a multi-phase agent pipeline that runs autonomously but (a) accumulates a ledger of questions needing human adjudication, (b) pauses exactly once at a review gate where a human answers or declines each question, (c) selectively re-runs affected phases with human answers treated as authoritative, and (d) accepts freeform human steering via a scratchpad file read by agents through a tool call.

## 2. Scenario

Fictional retailer **Meridian Retail** commissions a **cloud migration readiness assessment** of its legacy **Order Management System (OMS)** estate. Source data (all synthetic, committed to the repo):

- 3 enterprise PDFs (generated from committed markdown sources via `scripts/make_pdfs.py`, fpdf2)
- 2 fake legacy code repos
- 1 migration-readiness questionnaire template

### Source documents

- **PDF 1 — OMS Application Architecture & Business Context:** order flow, Python 2.7 monolith, cron batch jobs, NFS file shares, Oracle 11g PL/SQL; vaguely mentions "supporting batch processes" (understating the reconciliation function — planted); **no RTO/RPO anywhere** (planted).
- **PDF 2 — Enterprise Cloud Strategy & Patterns:** Azure declared strategic cloud, landing-zone model, PaaS-first patterns, integration guidance; IBM MQ "to be retired, date TBC" (planted).
- **PDF 3 — Cybersecurity & Data Protection Standards:** customer data must "remain in-region" with region unspecified (planted); secrets-management policy (no credentials in code); encryption and data-classification requirements.

### Legacy repos (`sample_data/repos/`)

- **`oms-monolith`:** `order_processor.py` (Python-2 idioms), `s3_uploader.py` (**boto3** — conflicts with Azure mandate), `db.py` (cx_Oracle + PL/SQL calls), hardcoded `/mnt/nfs/orders` paths, `crontab.txt`, ancient pinned `requirements.txt`.
- **`oms-batch-recon`:** nightly financial-reconciliation scripts (duties absent from the docs — planted scope gap), SFTP feed to VendorCo, **hardcoded database credentials** (violates PDF 3 — planted), reads the same Oracle schema.

### Planted gaps and conflicts (ledger fuel)

| # | Planting | Where | Expected ledger question | Likely raised by | Default assumption |
|---|---|---|---|---|---|
| 1 | Recon duties absent from docs | PDF 1 vs `oms-batch-recon` | Is reconciliation functionality in migration scope? | discovery | In scope |
| 2 | Azure mandate vs `boto3`/S3 in code | PDF 2 vs `oms-monolith` | Which target cloud is authoritative? | enterprise_context | Azure, per strategy |
| 3 | No RTO/RPO stated anywhere | PDF 1 | What are the RTO/RPO requirements? | deep_analysis (monolith) | RTO 4h / RPO 1h |
| 4 | "Remain in-region", region unspecified | PDF 3 | Which region/jurisdiction applies? | enterprise_context | EU |
| 5 | MQ retirement "date TBC" | PDF 2 | Replace MQ during migration or keep? | enterprise_context | Plan replacement now |
| 6 | Oracle 11g licensing portability unknown | PDF 1 vs `db.py` | Is the Oracle license BYOL-eligible in cloud? | deep_analysis (monolith) | No; plan managed-PostgreSQL path |
| 7 | Hardcoded DB credentials | `oms-batch-recon` vs PDF 3 | Remediate secrets before or during migration? | deep_analysis (batch-recon) | Vault + rotate before migration |

Phase instructions direct agents: *when evidence conflicts or a decision-critical fact is missing, call `raise_question` with a stated default assumption, then proceed using the default. Check the ledger context you were given first — do not re-raise a question that is already open.* Attribution above is indicative, not enforced.

## 3. Phases

| Order | Phase | Purpose | Sources pre-loaded into prompt |
|---|---|---|---|
| 1 | `discovery` | Initial scoping: learn the **true purpose** of the legacy system to be migrated — business function, users, criticality, actual vs documented scope | All 3 PDFs (extracted text) + file listings + READMEs of both repos |
| 2 | `deep_analysis` | Repo-level deep dive, **parallelised per repo** (one agent per repo, concurrent): code-level blockers, tech debt, data layer, integrations, security smells. One markdown report **per repo** | PDF 1 pre-loaded; the repository is **explored live via tools** (`list_files`, `read_file`) — contents are NOT pre-loaded |
| 3 | `enterprise_context` | Overlay corporate guidance: cloud strategy and patterns, cybersecurity practice, data-protection standards; flag strategy-vs-reality conflicts | PDFs 2 + 3 (+ memory already carries discovery/deep_analysis findings) |
| 4 | `questionnaire` | **Fill in the migration-readiness question template** from accumulated memory; any slot that is unanswerable or conflicting → `raise_question` with a default. The filled template is the phase report | `sample_data/questionnaire_template.md` + full ledger + memory |
| 5 | `review` | Human gate (not an agent) — see §8 | — |
| 6 | `final_report` | Readiness scorecard + migration recommendation + adjudication log, synthesized from post-adjudication memory | memory + phase reports + ledger |

### Questionnaire template (`sample_data/questionnaire_template.md`)

~10 standard migration-readiness slots: business purpose & criticality, migration scope, target platform, migration approach (6R), data store & licensing, RTO/RPO, data residency & classification, integrations & messaging, security posture gaps, timeline constraints. Committed as markdown with blank slots; the questionnaire agent fills each slot citing evidence (and question ids where a default was applied).

## 4. Repository layout

```
agent-framework-hotl/
├── pyproject.toml              # poetry; deps: agent-framework, agent-framework-ollama, pypdf, fpdf2; dev: pytest
├── README.md
├── scratchpad.md               # steering file — stable path, created empty if missing, never truncated
├── sample_data/
│   ├── docs_src/*.md           # markdown sources of the PDFs
│   ├── docs/*.pdf              # 3 generated PDFs (committed for zero-setup)
│   ├── questionnaire_template.md
│   └── repos/
│       ├── oms-monolith/
│       └── oms-batch-recon/
├── scripts/
│   └── make_pdfs.py            # regenerate PDFs from docs_src
├── src/hotl_demo/
│   ├── artifacts.py            # Memory (json), Ledger (jsonl), report writing — pure IO, thread-safe
│   ├── prompts/                # phase prompts: <phase>.md (YAML frontmatter + body) + Jinja2 wrappers
│   │   ├── discovery.md / deep_analysis.md / enterprise_context.md / questionnaire.md
│   │   └── _duties.md, initial.md, revision.md, final_report.md
│   ├── tools.py                # read_scratchpad, raise_question, update_memory (+ repo tools for analyzers)
│   ├── phases.py               # phase definitions + PhaseExecutor (+ per-repo analyzer instances, join)
│   ├── review.py               # ReviewExecutor (the human gate)
│   ├── report.py               # FinalReportExecutor
│   ├── pipeline.py             # build_workflow() graph assembly (fan-out/fan-in for deep_analysis)
│   └── main.py                 # CLI runner (argparse), event loop, review prompts, preflight
└── tests/
```

`poetry run demo` is the single entry point.

## 5. Run artifacts

Each run writes `output/run_<timestamp>/`:

| Artifact | Form | Writer |
|---|---|---|
| `phase_01_discovery.md`, `phase_02_deep_analysis_<repo>.md` (one per repo), `phase_03_enterprise_context.md`, `phase_04_questionnaire.md` | markdown phase reports; overwritten on revision | phase agents' final text |
| `memory.json` | see shape below | agents via `update_memory` tool |
| `ledger.jsonl` | append-only; one JSON object per question | agents via `raise_question` tool |
| `final_report.md` | executive summary, readiness scorecard, recommendation, adjudication log | report executor |

`scratchpad.md` lives at the repo root (stable, human-editable before/mid-run), not in the run dir.

### memory.json shape

```json
{
  "run_id": "…",
  "review_completed": false,
  "sections": {
    "discovery": {"<key>": "<value>"},
    "deep_analysis": {
      "oms-monolith": {"<key>": "<value>"},
      "oms-batch-recon": {"<key>": "<value>"}
    },
    "enterprise_context": {"<key>": "<value>"},
    "questionnaire": {"<slot>": "<filled value>"}
  }
}
```

`deep_analysis` is the only unit-nested section; each analyzer's tools are bound to its repo, so concurrent writers never collide on keys. Writes are serialized with a lock (see §11).

### Ledger entry schema

```json
{
  "id": "q-<seq>",
  "phase": "deep_analysis",
  "unit": "oms-batch-recon",
  "question": "…",
  "context": "…evidence…",
  "default_assumption": "…",
  "status": "open | answered | declined",
  "human_answer": null,
  "asked_at": "<iso8601>"
}
```

`unit` is the repo name for deep_analysis questions, else `null`. Appends set `status: open`; the review gate rewrites entries to `answered` (with `human_answer`) or `declined`.

## 6. Agent tools

Three core tools on every phase agent, all side-effect/steering ops (gemma4:31b tool calling verified):

1. `read_scratchpad()` → scratchpad text, or a note that it is empty. Every phase agent is instructed to consult it before working. **This is the user-mandated steering mechanism.**
2. `raise_question(question, context, default_assumption)` → appends an `open` ledger entry tagged with the calling agent's phase (and unit, for repo analyzers); returns the assigned id.
3. `update_memory(key, value)` → merges into the calling agent's own memory section (phase- and unit-bound by the executor; agents cannot write other sections). Values are flat strings.

**deep_analysis analyzers additionally get repo-exploration tools** bound to their repository (rev 4): `list_files()` (file tree as relative paths) and `read_file(path)` (one file's contents; path-validated against traversal, size-capped). Analyzers explore agentically — the executor no longer pre-loads repo contents for them.

PDF text (pypdf), the questionnaire template, and the current ledger remain **pre-loaded into phase prompts by the executor** — deterministic, no retrieval loops needed at 262k context.

### Prompt templates (rev 4)

Prompts live outside the code as **markdown files with YAML frontmatter, rendered through Jinja2** (`src/hotl_demo/prompts/`): one `<phase>.md` per phase whose frontmatter carries the phase metadata (`name`, `order`, `per_repo`, `report_filename` — the executor id rule stays in code) and whose body is the phase instructions (a Jinja2 template; analyzers receive `{{ unit }}`). Shared wrappers `initial.md` and `revision.md` render the full prompt from `{{ instructions }}`, `{{ sources }}`, `{{ memory }}`, `{{ open_questions }}` (+ `{{ answers }}`, `{{ previous_report }}` for revisions), including the common duties block `_duties.md`; `final_report.md` templates the report-phase prompt. `build_phase_specs` discovers phases by reading this directory — phases are editable (and re-orderable) without touching Python.

## 7. Workflow graph (Agent Framework)

Built with `WorkflowBuilder`. deep_analysis is a **graph-level fan-out/fan-in** (per the official `fan_out_fan_in_edges.py` sample) — one analyzer node per repo runs concurrently. The human pause uses native `ctx.request_info()` / `@response_handler` (per the official `guessing_game_with_human_input.py` sample).

```
                    ┌─ analyze[oms-monolith] ──┐
discovery ── fan-out┤                          ├ join ── enterprise_context ── questionnaire ── review ── final_report
                    └─ analyze[oms-batch-recon]┘                                                  │
    ▲                        ▲  ▲                        ▲                    ▲                   │
    └────────────────────────┴──┴────────────────────────┴────────────────────┴── RevisionTrigger┘
   (revised phase/analyzer sends revision-mode completion straight back to review)
```

- **Messages:** `PhaseTrigger {mode: initial|revision, answers: [...]}` / `PhaseDone {phase, unit, mode}` / `ReportTrigger`. Edge conditions route on message type + mode: initial completions flow forward (analyzers → join, which waits for all repos before triggering enterprise_context); revision completions return directly to `review`, bypassing the join. `questionnaire`'s single edge to `review` carries both modes; `PhaseDone.mode` disambiguates.
- **PhaseExecutor** (one class; instances: discovery, one analyzer per repo, enterprise_context, questionnaire): builds prompt (phase instructions + pre-loaded sources per §3 + current `memory.json` + open-ledger summary + scratchpad reminder), runs its `Agent` (`OllamaChatClient`), writes the phase report from the agent's final text, messages the next node. In revision mode the prompt additionally carries the human answers (marked authoritative) and the phase's previous report; the agent rewrites report + memory entries.
- **Join** is a trivial custom executor: collect `PhaseDone` from every analyzer (initial mode only), then trigger enterprise_context.
- **File-backed shared state:** `memory.json`/`ledger.jsonl` are injected into executors as an artifact-store object bound to the run dir. The files themselves are the demo's long-term-memory story; workflow-internal state is not duplicated.

## 8. Review gate (HOTL centerpiece)

1. On arrival from `questionnaire`, load all `open` ledger questions in ledger order. None → `ReportTrigger` directly.
2. Emit one `ctx.request_info(LedgerQuestionRequest, response_type=str)` per question; the workflow run goes idle (framework-native pause).
3. CLI runner catches `request_info` events and prompts per question: question + context + default assumption. Typed text = answer (**treated as authoritative**); plain ENTER (or whitespace) = decline.
4. Runner resumes via `workflow.run(responses={request_id: answer})`. The `@response_handler` marks each ledger entry `answered`/`declined`.
5. When all responses are in: affected targets = the `(phase, unit)` pairs of entries whose status is `answered`. Declined → no re-run; the default assumption stands.
6. Affected targets re-run **sequentially in phase order** (for deep_analysis, only the affected repo's analyzer; gate dispatches the next `RevisionTrigger` only after the previous revision returns), then `ReportTrigger`.

Duplicate suppression is prompt-level, not structural: every phase receives the current open ledger in its prompt and is instructed not to re-raise existing questions. No curation machinery.

### Review-once rule

Entering the gate sets `review_completed: true` in `memory.json`; the gate's prompt loop is guarded by it. Questions raised during revision runs still append to the ledger but are never prompted — the final report lists them as *open — default assumption applied*.

### Deliberate simplification (marked in code)

Re-runs do not cascade downstream: an answered discovery question re-runs `discovery` only. Downstream phases change only if they raised answered questions themselves. The final verdict still absorbs every answer because `final_report` synthesizes from post-adjudication memory. Upgrade path if needed later: cascade re-runs from the earliest affected phase.

## 9. Final report

The report executor composes `final_report.md` from `memory.json` + phase reports (one LLM call): executive summary, **readiness scorecard, and migration recommendation** — the verdict lives here, synthesized fresh from post-adjudication memory (including the filled questionnaire). It then **deterministically appends the adjudication log** — a table of answered (with human answer), declined, and still-open questions straight from the ledger — so the human-adjudication record is always accurate regardless of LLM behavior. Outputs the report path via `ctx.yield_output`.

## 10. CLI UX

```
$ poetry run demo
Preflight: Ollama OK, gemma4:31b present.
Phase 1/4 discovery              … report written (1 question raised)
Phase 2/4 deep_analysis          … analyzing 2 repos in parallel
  ├ oms-monolith                 … report written (2 questions raised)
  └ oms-batch-recon              … report written (1 question raised)
Phase 3/4 enterprise_context     … report written (3 questions raised)
Phase 4/4 questionnaire          … template filled: 10 slots, 7 from evidence, 3 on defaults
== REVIEW — 7 open questions ==
[q-1] (discovery) Is reconciliation functionality in migration scope?
      Evidence: oms-batch-recon performs financial reconciliation; absent from PDF 1.
      Default if declined: in scope.
      Your answer (ENTER to decline): _
…
Re-running affected: discovery, deep_analysis[oms-batch-recon], enterprise_context
Final report: output/run_20260714_1702/final_report.md
```

## 11. Error handling

- **Preflight:** HTTP GET `localhost:11434/api/tags`; verify server up and `gemma4:31b` present; fail fast with an actionable message.
- **Missing `update_memory` calls:** one bounded nudge turn; if still absent, proceed and note the gap in the phase report (no crash).
- **Tool arg validation:** tools return corrective error strings; the framework feeds them back to the model.
- **Concurrent analyzers:** artifact store guards `memory.json`/`ledger.jsonl` with a `threading.Lock`; ledger ids assigned under the lock; atomic writes (temp file + `os.replace`) — humans may have the files open mid-run.
- **Review input:** empty/whitespace = decline. Ctrl-C aborts cleanly; artifacts persist.

## 12. Testing (pytest; LLM-free by default)

- **Unit (no LLM):** ledger append/update/query incl. unit attribution and id assignment under concurrency; memory merge incl. unit-nested deep_analysis section + review-once flag; scratchpad tool (missing/empty/content); repo-exploration tools (tree, read, traversal guard, missing-file error); prompt-file frontmatter parsing and Jinja2 rendering; affected-target computation `(phase, unit)`; decline semantics; revision-prompt assembly; join logic (waits for all analyzers, initial mode only); workflow graph shape (nodes/edges exist, one analyzer per repo).
- **Live E2E (opt-in):** `@pytest.mark.ollama`, skipped unless `OLLAMA_E2E=1`; drives the full pipeline with scripted stdin answers.
- No mock ChatClient: decision-bearing logic lives in pure functions; LLM wiring is covered by the opt-in live test.

## 13. Decisions log

| Decision | Choice | Alternatives considered |
|---|---|---|
| Scenario | Cloud migration readiness | Modernization assessment, compliance audit |
| Review UX | Interactive CLI, one process | Checkpoint pause/exit/resume; both via flag |
| Closed-pipeline contrast mode | Not built (HOTL only) | `--closed` flag |
| Orchestration | `WorkflowBuilder` graph + custom executors | `SequentialBuilder.with_request_info` (wrong interaction shape); plain asyncio (doesn't showcase framework) |
| Phase purposes (rev 3) | discovery = true-purpose scoping; deep_analysis = per-repo parallel deep dive; enterprise_context = corporate guidance overlay; questionnaire = fill question template | rev 2's ledger-curation questionnaire (dropped — user clarified intent) |
| deep_analysis parallelism | Graph-level fan-out/fan-in, one analyzer node per repo | asyncio.gather inside one executor (simpler, but hides the framework's parallelism feature) |
| Sample repos | 2 (monolith + batch-recon) | 1 (no parallelism story); 3+ (slower demo on a local 31B model) |
| Duplicate questions | Prompt-level suppression (open ledger in every prompt) | rev 2's structural dedupe via curation memory keys (dropped as machinery without a phase to own it) |
| Structured side-effects | Tool calls | `response_format` structured output (unverified through `OllamaChatClient`) |
| Source retrieval | Pre-loaded into prompts, EXCEPT deep_analysis (rev 4: agentic repo exploration via `list_files`/`read_file` per user) | All-preloaded (rev 3); retrieval tools everywhere (flaky for docs) |
| Prompt authoring (rev 4) | Markdown files with YAML frontmatter + Jinja2 rendering in `src/hotl_demo/prompts/` | Python string constants (rev 3 — harder to edit/showcase) |
| Model | `gemma4:31b` (user's "gemma:31b", present locally) | — |
