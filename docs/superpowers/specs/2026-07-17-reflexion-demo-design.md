# Reflexion Demo — Design

**Date:** 2026-07-17
**Status:** Approved
**Stack:** Python ≥3.10, Poetry, pytest, Microsoft Agent Framework (`agent-framework`, `agent-framework-ollama`), Ollama `gemma4:31b` (local; same client conventions as the HOTL demo)

## 1. Purpose

Standalone demo of the **reflexion mechanism** on Microsoft Agent Framework: a
**worker** agent drafts a migration report using read/write tools; a
**reviewer** agent — holding the *same read tools over the same corpus*
(**information parity**) — independently verifies the draft against the
sources and returns a boolean verdict. Rejection feedback is fed back to the
worker verbatim as steering for the next draft. Two budgets force
convergence, and both end in **tool-stripping**: the agent is told it has
been reasoning long enough and must produce output from what it already
knows.

Standalone means: no imports from `hotl_demo`. The demo shares only
`sample_data/` and the repo's Ollama conventions.

## 2. Scenario

Corpus: the existing `sample_data/` (3 markdown docs in `docs_src/`, 2 small
Python repos), text files only — the binary PDF twins in `docs/` are
excluded.

Default topic (overridable via `--topic`):

> Assess migrating OMS file storage from the NFS file store to Amazon S3.

The corpus already contains reviewer fuel for this topic: the enterprise
cloud strategy mandates **Azure** while `s3_uploader.py` targets S3; the
cybersecurity standards impose in-region data residency and
secrets-management rules; `file_store.py` hardcodes NFS paths. A report that
misses these tensions gives an evidence-grounded reviewer real grounds to
reject.

## 3. Architecture

New package, console script `poetry run reflexion`
(`reflexion = "reflexion_demo.main:main"`).

```text
src/reflexion_demo/
  __init__.py
  main.py       # argparse CLI, console narration, run-dir creation
  graph.py      # Worker/Reviewer executors, messages, WorkflowBuilder
  tools.py      # corpus read tools + write_report/read_report factories
  budget.py     # ToolBudget + stripping function-middleware
  prompting.py  # Jinja2 env + render helpers for the two templates
  prompts/
    worker.md   # Jinja2; initial/revision/finalize variants via context vars
    reviewer.md # Jinja2; verdict instructions
```

### Workflow graph

Modeled on MAF's own
`python/samples/03-workflows/agents/workflow_as_agent_reflection_pattern.py`
and this repo's message-type routing rule (edge conditions are `isinstance`
checks; message types encode meaning, never mode flags):

- **Messages** (dataclasses): `DraftReady(cycle: int, topic: str)`
  worker→reviewer; `ReviewVerdict(approved: bool, feedback: str, cycle: int,
  reviewer_tool_calls: int)` reviewer→worker. The verdict carries the
  reviewer's spent tool-call count so the worker executor — the sole
  `review_log.jsonl` writer — can log complete cycle lines.
- **Edges:** `worker → reviewer → worker` (a cycle). Start executor: worker,
  seeded with the topic string.
- **WorkerExecutor** holds the cycle counter in executor memory. This demo
  has no checkpoint/resume, so in-memory state is safe here — unlike the
  HOTL review gate, whose progress must be ledger-derived.
  - On `str` (topic): initial draft turn → `DraftReady`.
  - On `ReviewVerdict(approved=True)`: `yield_output` (report path + summary).
  - On `ReviewVerdict(approved=False)` with cycles remaining: revision turn;
    the feedback string is inserted verbatim into the revision prompt →
    `DraftReady`.
  - On `ReviewVerdict(approved=False)` with the cycle budget exhausted:
    **forced-finalize turn** — agent constructed with `write_report` only
    (no read tools) and the finalize instruction ("you have been reasoning
    for a long time; produce the report now from the information you
    have") — then `yield_output` regardless, logged with `forced: true`.
- **ReviewerExecutor** runs at most `max_cycles` times. It reads the report
  via `read_report` and spot-checks claims against the corpus with its own
  read tools, then returns a Pydantic verdict
  `{approved: bool, feedback: str}` via `response_format` (as in the MAF
  sample). Malformed verdict → one retry → treat as rejection with generic
  feedback ("the reviewer could not produce a valid verdict; improve
  evidence citations and completeness, then resubmit").
- **Cycle accounting** (`--max-cycles`, default 3): cycle = one draft (or
  revision) turn plus its review. At most 3 reviews and 4 worker turns
  (3 drafts + 1 forced finalize). Approval at any review ends the run.
- One fresh `AgentSession` per agent turn, passed explicitly to `run()` —
  same idiom (and same reason) as `hotl_demo/phases.py`.

### Agents

Both agents are `Agent(client=OllamaChatClient(), ...)` — no-arg client, the
repo standard. `--model` sets `OLLAMA_MODEL` before construction; `--num-ctx`
/ `OLLAMA_NUM_CTX` flows to `default_options={"num_ctx": N}` on both agents
(Ollama silently truncates otherwise). No compaction strategy — add only if
a real run overflows.

## 4. Tools and information parity

All tools are closure-bound factories in `reflexion_demo/tools.py`, same
idioms as `hotl_demo/tools.py`: traversal-guarded resolution, oversized reads
truncated, failures returned as `"ERROR: ..."` strings (never raised) so the
framework feeds them back to the model.

| Tool | Worker | Reviewer | Budgeted | Purpose |
|---|---|---|---|---|
| `list_files` | yes | yes | yes | List corpus text files (`.md`/`.py`/`.txt`) under `sample_data/`, relative paths |
| `read_file` | yes | yes | yes | Read one corpus file by relative path |
| `write_report` | yes | — | no | Atomic write of the full report markdown to the run dir |
| `read_report` | — | yes | yes | Read the current `report.md` |

**Information parity:** both agents get the *identical* corpus binding
(`list_files`/`read_file` over the same root). The reviewer additionally
reads the artifact under review; the worker additionally writes it. Parity
of information, asymmetry of authority: only the worker writes, only the
reviewer approves.

The reviewer judges what was actually **written** (`read_report`), not what
the worker claims in conversation.

## 5. Budgets and tool-stripping

Two independent limits, both explicit CLI knobs:

1. **Tool-call budget** (`--max-tool-calls`, default 12, per agent turn,
   fresh each turn). One function middleware (in `budget.py`) owns a
   counter. Read tools count (`list_files`, `read_file`, `read_report`);
   `write_report` is exempt — it is delivery, not exploration. Coaching is
   anticipatory, not only punitive: the three calls before exhaustion (3, 2,
   1 remaining) each append an escalating countdown line to that call's
   result, so the model can re-plan while there is still runway — this
   worker is coached mid-run, not only nudged after the fact. On the call
   that exhausts the budget, the middleware executes that call normally,
   then strips the read tools for the remainder of the turn via
   `FunctionInvocationContext.remove_tools()` (the framework's sanctioned
   live-mutation point for progressive tool exposure) and appends a closing
   message to that call's result: *"Exploration is closed. You have what
   you need — write the complete report now with write_report."* Subsequent
   model iterations in that turn see only `write_report` (worker) or no
   tools (reviewer, whose verdict is structured output and needs no tool).
   Applied to **both** agents — parity again. The countdown and closing
   wording are byte-identical to `reflection_demo`'s `budget.py` — coaching
   one worker better than the other would be a confound in the A/B the two
   demos exist to run — pinned by `tests/test_budget_wording_parity.py`. See
   `docs/superpowers/specs/2026-07-30-reflection-tool-budget-design.md` for
   the full design.
2. **Review-cycle budget** (`--max-cycles`, default 3). Exhaustion triggers
   the forced-finalize turn (§3): stripping expressed at construction time —
   the finalize agent is built without read tools.

Grounding facts (verified against the installed `agent_framework`): run-level
`tools` **append** to agent-level tools rather than replace them, so
stripping must happen at construction time or via `remove_tools()`;
`FunctionInvocationContext.tools` is documented as a live, mutable list with
`add_tools`/`remove_tools`.

## 6. Prompts

Jinja2 markdown under `src/reflexion_demo/prompts/`, no YAML frontmatter —
there is no phase discovery here, so frontmatter would be dead weight.

- `worker.md` — one template, three variants selected by context vars:
  initial (`topic`), revision (`topic`, `feedback`, cycle numbers), finalize
  (`topic`, finalize instruction). Instructs: explore the corpus with tools,
  ground every claim in a source file, deliver via `write_report`.
- `reviewer.md` — instructs: read the report, independently spot-check its
  claims against the corpus, evaluate accuracy (claims match sources),
  coverage (material conflicts/gaps for the topic addressed — e.g. a
  cloud-provider mandate conflict), and actionability; approve only when all
  hold; on rejection give specific, actionable feedback naming what is
  missing or wrong and which angle to pursue.

The prompts directory joins the markdown lint gate
(`.pymarkdown.json` scan set + the lint test's path list).

## 7. Run artifacts and console

`output/reflexion_<timestamp>/`:

- `report.md` — the deliverable; each revision overwrites atomically
  (temp file + `os.replace`).
- `review_log.jsonl` — append-only, one line per cycle:
  `{cycle, approved, feedback, forced, worker_tool_calls,
  reviewer_tool_calls}` plus a final outcome line. Written solely by the
  worker executor (after each verdict, and at yield); plain append, no
  lock — nothing is concurrent in this demo.

Console narration mirrors the HOTL demo's texture: cycle banners, a line
when a strip fires, verdicts and feedback, final outcome + artifact paths.
CLI stays stdlib (`argparse`/`print`).

## 8. CLI

```bash
poetry run reflexion                       # defaults: topic per §2, 3 cycles, 12 tool calls
poetry run reflexion --topic "..."         # any migration topic over the same corpus
poetry run reflexion --max-cycles 2
poetry run reflexion --max-tool-calls 6
poetry run reflexion --model gemma4:31b    # sets OLLAMA_MODEL
poetry run reflexion --num-ctx 32768       # overrides OLLAMA_NUM_CTX
```

## 9. Error handling

- Tool failures: `ERROR:` strings, never exceptions (repo rule).
- Empty model text: one retry (mirrors `_invoke_report`).
- Worker turn ends without ever calling `write_report` (detected via a
  closure-set flag): one nudge turn ("call write_report with your report
  now") with `write_report` still available; if still absent, persist the
  turn's final message text as `report.md` and continue.
- Reviewer verdict unparseable after one retry: reject with generic
  feedback (§3) — safer than approving unverified output; never crashes the
  loop.

## 10. Testing

LLM-free by default (`-m 'not ollama'`), decision logic in pure functions,
executor tests call handlers directly with small fakes — repo rules. No
`tests/__init__.py`.

- `tests/test_reflexion_budget.py` — counting; `write_report` exemption;
  strip fires exactly at the budget; nudge appended once; reviewer variant
  strips to zero tools.
- `tests/test_reflexion_graph.py` — worker handler paths (approved → yield;
  rejected → revision with feedback in prompt; exhausted → forced finalize,
  `forced: true` logged); reviewer verdict parse + retry + reject fallback;
  cycle accounting (≤ max_cycles reviews, ≤ max_cycles+1 worker turns).
- `tests/test_reflexion_tools.py` — traversal guard; text-file filter (PDFs
  invisible); atomic report write; `read_report` before first write returns
  an `ERROR:` string; write-flag detection.
- Markdown lint gate covers `src/reflexion_demo/prompts/`.
- One live smoke test (`@pytest.mark.ollama`, `OLLAMA_E2E=1`): tiny budgets
  (`--max-cycles 1`-equivalent config) to keep runtime short; asserts a
  report file and a coherent `review_log.jsonl` exist.

## 11. Out of scope (deliberate)

Checkpoint/resume, scratchpad steering, compaction, `ArtifactStore`/ledger
reuse, PDF reading, DevUI, wrapping the workflow `as_agent()`, and README
changes. None serve the mechanism being demoed; any can be added later
without reshaping the design.

## 12. References

- MAF sample read for this design:
  `python/samples/03-workflows/agents/workflow_as_agent_reflection_pattern.py`
  (Worker ↔ Reviewer cyclic graph, structured verdict, `yield_output` on
  approval).
- Also considered and rejected: `AgentLoopMiddleware.with_judge` (judge is a
  bare chat client without tools — breaks information parity);
  plain async for-loop (fewer MAF concepts demonstrated; the graph *is* the
  mechanism being demoed).
