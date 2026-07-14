# Human-on-the-Loop Pipeline Demo — Design

**Date:** 2026-07-14
**Status:** Approved
**Stack:** Python ≥3.10, Poetry, pytest, Microsoft Agent Framework (`agent-framework`, `agent-framework-ollama`), Ollama `gemma4:31b` (local; tools + thinking capable, 262k context — verified via `ollama show`)

## 1. Purpose

Demo project showcasing **human-on-the-loop (HOTL)** capability on Microsoft Agent Framework: a sequential multi-phase agent pipeline that runs autonomously but (a) accumulates a ledger of questions needing human adjudication, (b) pauses exactly once at a review gate where a human answers or declines each question, (c) selectively re-runs affected phases with human answers treated as authoritative, and (d) accepts freeform human steering via a scratchpad file read by agents through a tool call.

## 2. Scenario

Fictional retailer **Meridian Retail** commissions a **cloud migration readiness assessment** of its legacy **Order Management System (OMS)**. Source data (all synthetic, committed to the repo):

- 3 enterprise-architecture PDFs (generated from committed markdown sources via `scripts/make_pdfs.py`, fpdf2)
- a ~10-file fake legacy code repo

### Planted gaps and conflicts (ledger fuel)

| # | Planting | Where | Expected ledger question | Default assumption |
|---|---|---|---|---|
| 1 | EA doc mandates Azure; code imports `boto3`/S3 | PDF 1 vs `s3_uploader.py` | Which target cloud is authoritative? | Azure, per EA doc |
| 2 | No RTO/RPO stated anywhere | PDF 2 | What are the RTO/RPO requirements? | RTO 4h / RPO 1h (conservative) |
| 3 | Customer data must "remain in-region", region unspecified | PDF 1 | Which region/jurisdiction applies? | EU |
| 4 | MQ retirement "planned", no date | PDF 3 | Replace MQ during migration or keep? | Plan replacement now |
| 5 | Oracle 11g + PL/SQL; license portability unknown | PDF 1/2 vs `db.py` | Is the Oracle license BYOL-eligible in cloud? | No; plan managed-PostgreSQL path |

Phase instructions direct agents: *when evidence conflicts or a decision-critical fact is missing, call `raise_question` with a stated default assumption, then proceed using the default.*

### Mock data contents

- **PDF 1 — Enterprise Architecture Overview:** on-prem VMware estate, Oracle 11g, IBM MQ, DR site; Azure declared strategic cloud; vague data-residency clause.
- **PDF 2 — OMS Application Architecture:** Python 2.7 monolith, cron batch jobs, NFS file shares, PL/SQL stored procedures; no RTO/RPO.
- **PDF 3 — Integration Landscape:** SOAP services, nightly SFTP feed to VendorCo, IBM MQ with unspecified retirement date.
- **Legacy repo (`sample_data/legacy_repo/`):** `order_processor.py` (Python-2 idioms), `s3_uploader.py` (boto3), `db.py` (cx_Oracle + PL/SQL calls), hardcoded `/mnt/nfs/orders` paths, `crontab.txt`, ancient pinned `requirements.txt`, plus a few filler modules.

## 3. Repository layout

```
agent-framework-hotl/
├── pyproject.toml              # poetry; deps: agent-framework, agent-framework-ollama, pypdf, fpdf2; dev: pytest
├── README.md
├── scratchpad.md               # steering file — stable path, created empty if missing, never truncated
├── sample_data/
│   ├── docs_src/*.md           # markdown sources of the PDFs
│   ├── docs/*.pdf              # generated PDFs (committed for zero-setup)
│   └── legacy_repo/            # fake OMS codebase
├── scripts/
│   └── make_pdfs.py            # regenerate PDFs from docs_src
├── src/hotl_demo/
│   ├── artifacts.py            # Memory (json), Ledger (jsonl), report writing — pure IO
│   ├── tools.py                # read_scratchpad, raise_question, update_memory
│   ├── phases.py               # phase definitions + PhaseExecutor
│   ├── review.py               # ReviewGateExecutor
│   ├── report.py               # FinalReportExecutor
│   ├── pipeline.py             # build_workflow() graph assembly
│   └── main.py                 # CLI runner (argparse), event loop, review prompts, preflight
└── tests/
```

`poetry run demo` is the single entry point.

## 4. Run artifacts

Each run writes `output/run_<timestamp>/`:

| Artifact | Form | Writer |
|---|---|---|
| `phase_0N_<name>.md` | markdown phase report | phase agent's final text; overwritten on revision |
| `memory.json` | `{run_id, review_completed, sections: {architecture, codebase, readiness, recommendation}}` | agents via `update_memory` tool |
| `ledger.jsonl` | append-only; one JSON object per question | agents via `raise_question` tool |
| `final_report.md` | executive summary, readiness scorecard, adjudication log | report executor |

`scratchpad.md` lives at the repo root (stable, human-editable before/mid-run), not in the run dir.

### Ledger entry schema

```json
{
  "id": "q-<seq>",
  "phase": "code_audit",
  "question": "…",
  "context": "…evidence…",
  "default_assumption": "…",
  "status": "open | answered | declined",
  "human_answer": null,
  "asked_at": "<iso8601>"
}
```

Appends set `status: open`. The review gate rewrites entries to `answered` (with `human_answer`) or `declined`.

## 5. Agent tools

Exactly three, all side-effect/steering ops (gemma4:31b tool calling verified):

1. `read_scratchpad()` → scratchpad text, or a note that it is empty. Every phase agent is instructed to consult it before working. **This is the user-mandated steering mechanism.**
2. `raise_question(question, context, default_assumption)` → appends an `open` ledger entry tagged with the current phase; returns the assigned id.
3. `update_memory(section, key, value)` → merges into the phase's section of `memory.json`. Section validated against the phase's allowed section.

PDF text (pypdf) and legacy-repo files are **pre-loaded into phase prompts by the executor** — deterministic, no flaky retrieval tool loops; 262k context makes this trivial.

## 6. Workflow graph (Agent Framework)

Built with `WorkflowBuilder`; human pause uses native `ctx.request_info()` / `@response_handler` (same pattern as the official `guessing_game_with_human_input.py` sample).

```
architecture_intake → code_audit → readiness_assessment → migration_recommendation → review_gate → final_report
        ▲ ▲ ▲ ▲                                                                          │
        └─┴─┴─┴────────────── RevisionTrigger (per affected phase) ──────────────────────┘
   (revised phase sends revision-mode completion back to review_gate)
```

- **Messages:** `PhaseTrigger {mode: initial|revision, answers: [...]}` / `PhaseDone {phase, mode}` / `ReportTrigger`. Edge conditions route on message type + mode: initial completions flow forward; revision completions return to the review gate.
- **PhaseExecutor** (one class, four instances): builds prompt (phase instructions + pre-loaded sources + current `memory.json` + scratchpad reminder), runs its `Agent` (`OllamaChatClient`), writes the phase report from the agent's final text, messages the next node. In revision mode the prompt additionally carries the human answers (marked authoritative) and the phase's previous report; the agent rewrites report + memory entries.
- **File-backed shared state:** `memory.json`/`ledger.jsonl` are injected into executors as an artifact-store object bound to the run dir. The files themselves are the demo's long-term-memory story; workflow-internal state is not duplicated.

## 7. Review gate (HOTL centerpiece)

1. On arrival from phase 4, load all `open` ledger questions. None → `ReportTrigger` directly.
2. Emit one `ctx.request_info(LedgerQuestionRequest, response_type=str)` per question; the workflow run goes idle (framework-native pause).
3. CLI runner catches `request_info` events and prompts per question: question + context + default assumption. Typed text = answer (**treated as authoritative**); plain ENTER (or whitespace) = decline.
4. Runner resumes via `workflow.run(responses={request_id: answer})`. The `@response_handler` marks each ledger entry `answered`/`declined`.
5. When all responses are in: affected phases = phases that raised **answered** questions. Declined → no re-run; the default assumption stands.
6. Affected phases re-run **sequentially in phase order** (gate dispatches next `RevisionTrigger` only after the previous revision returns), then `ReportTrigger`.

### Review-once rule

Entering the gate sets `review_completed: true` in `memory.json`; the gate's prompt loop is guarded by it. Questions raised during revision runs still append to the ledger but are never prompted — the final report lists them as *open — default assumption applied*.

### Deliberate simplification (marked in code)

Re-runs do not cascade downstream: an answered phase-1 question re-runs phase 1 only. Downstream phases change only if they raised answered questions themselves. Upgrade path if needed later: cascade re-runs from the earliest affected phase.

## 8. Final report

The report executor composes `final_report.md` from `memory.json` + phase reports (one LLM call): executive summary, readiness scorecard, migration recommendation. It then **deterministically appends the adjudication log** — a table of answered (with human answer), declined, and still-open questions straight from the ledger — so the human-adjudication record is always accurate regardless of LLM behavior. Outputs the report path via `ctx.yield_output`.

## 9. CLI UX

```
$ poetry run demo
Preflight: Ollama OK, gemma4:31b present.
Phase 1/4 architecture_intake … report written (2 questions raised)
Phase 2/4 code_audit          … report written (2 questions raised)
Phase 3/4 readiness_assessment … report written (1 question raised)
Phase 4/4 migration_recommendation … report written
== REVIEW GATE — 5 open questions ==
[q-1] (architecture_intake) Which target cloud is authoritative?
      Evidence: EA overview mandates Azure; s3_uploader.py uses AWS boto3.
      Default if declined: Azure per EA doc.
      Your answer (ENTER to decline): _
…
Re-running affected phases: architecture_intake, code_audit
Final report: output/run_20260714_1702/final_report.md
```

## 10. Error handling

- **Preflight:** HTTP GET `localhost:11434/api/tags`; verify server up and `gemma4:31b` present; fail fast with an actionable message.
- **Missing `update_memory` calls:** one bounded nudge turn; if still absent, proceed and note the gap in the phase report (no crash).
- **Tool arg validation:** tools return corrective error strings; the framework feeds them back to the model.
- **Atomic writes** for `memory.json`/`ledger.jsonl` (temp file + `os.replace`) — humans may have the files open mid-run.
- **Review input:** empty/whitespace = decline. Ctrl-C aborts cleanly; artifacts persist.

## 11. Testing (pytest; LLM-free by default)

- **Unit (no LLM):** ledger append/update/query; memory merge + review-once flag; scratchpad tool (missing/empty/content); affected-phase computation; decline semantics; revision-prompt assembly; workflow graph shape (nodes/edges exist).
- **Live E2E (opt-in):** `@pytest.mark.ollama`, skipped unless `OLLAMA_E2E=1`; drives the full pipeline with scripted stdin answers.
- No mock ChatClient: decision-bearing logic lives in pure functions; LLM wiring is covered by the opt-in live test.

## 12. Decisions log

| Decision | Choice | Alternatives considered |
|---|---|---|
| Scenario | Cloud migration readiness | Modernization assessment, compliance audit |
| Review UX | Interactive CLI, one process | Checkpoint pause/exit/resume; both via flag |
| Closed-pipeline contrast mode | Not built (HOTL only) | `--closed` flag |
| Orchestration | `WorkflowBuilder` graph + custom executors | `SequentialBuilder.with_request_info` (wrong interaction shape); plain asyncio (doesn't showcase framework) |
| Structured side-effects | Tool calls | `response_format` structured output (unverified through `OllamaChatClient`) |
| Source retrieval | Pre-loaded into prompts | Retrieval tools (flakier, no benefit at 262k context) |
| Model | `gemma4:31b` (user's "gemma:31b", present locally) | — |
