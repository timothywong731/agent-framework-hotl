# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

## What this is

Human-on-the-loop (HOTL) demo on Microsoft Agent Framework (Python): a
multi-phase agent pipeline that assesses a fictional legacy system's cloud
migration readiness, accumulates a question ledger, pauses exactly once at a
human review gate, selectively re-runs answered phases, and takes freeform
steering from `scratchpad.md`. The design spec at
`docs/superpowers/specs/2026-07-14-hotl-pipeline-design.md` is authoritative
for behavioral semantics; the plan next to it is historical.

## Commands

```bash
poetry install
poetry run demo                                # needs local Ollama + gemma4:31b
poetry run demo --pause                        # checkpoint + exit at the review gate
poetry run demo --resume output/run_<ts>       # apply review.jsonl answers, finish the run
poetry run pytest                              # fast, LLM-free; includes the markdown lint gate
poetry run pytest tests/test_review.py -v      # one file
poetry run pytest tests/test_review.py::test_review_once_guard -v   # one test
OLLAMA_E2E=1 poetry run pytest -m ollama -s    # live E2E, ~10 min on gemma4:31b
poetry run pymarkdown --config .pymarkdown.json scan README.md CLAUDE.md src/hotl_demo/prompts
poetry run python scripts/make_pdfs.py         # regenerate PDFs after editing sample_data/docs_src/
```

PowerShell equivalent for the live E2E: `$env:OLLAMA_E2E="1"; poetry run pytest -m ollama -s`.
The model comes from the `OLLAMA_MODEL` env var (or `demo --model`);
`OllamaChatClient` is constructed no-arg everywhere (the compaction
summarizer pins `num_ctx` per call, not via the ctor). `OLLAMA_HOST` may be
scheme-less. `OLLAMA_NUM_CTX` (or `demo --num-ctx`) sizes both the Ollama
window and the compaction budget.

## Architecture

- **Graph** (`pipeline.py`): discovery -> fan-out one analyzer per repo ->
  join -> enterprise_context -> questionnaire -> review (human gate) ->
  final_report, plus revision edges from review back to every phase.
- **Message types encode run mode** (`phases.py`): `PhaseDone`/`AnalysisDone`
  are initial completions, `RevisionDone` is a post-review completion,
  `RevisionTrigger` targets exactly one `(phase, unit)`. Every edge condition
  is an `isinstance` check. To change routing, add message types - never mode
  flags.
- **Review gate** (`review.py`): `ctx.request_info` per open question pauses
  the run; `main.py` collects the `request_info` events, prompts stdin, and
  resumes with `workflow.run(responses={...})`. Review runs ONCE per run
  (`review_completed` latch in memory, set on gate entry). Empty/whitespace
  answer = declined (default assumption stands, no re-run). Only phases with
  ANSWERED questions re-run - sequentially, in pipeline order. Questions
  raised during re-runs are never prompted; the final report lists them as
  open.
  With `--pause` the gate checkpoints and exits (answers land in
  `review.jsonl` - id + answer only); `--resume` restores via
  `gate_checkpoint()`, which selects the checkpoint holding pending
  request_info events - `list_checkpoints` is glob-ordered, never trust
  "latest".
- **File-backed state** (`artifacts.py`): `memory.json` (the `deep_analysis`
  section nests per repo), append-only `ledger.jsonl`, markdown reports - all
  under `output/run_<timestamp>/`. Every mutation goes through
  `ArtifactStore` (one `threading.Lock`, atomic temp-file + `os.replace`);
  the two analyzers write concurrently. Never touch these files another way.
- **Prompts are data** (`src/hotl_demo/prompts/`): one `<phase>.md` per phase
  with YAML frontmatter (`name`, `order`, `per_repo`, `report_filename`) and
  a Jinja2 body; wrappers `initial.md`/`revision.md`/`final_report.md`
  assemble full prompts. `build_phase_specs` discovers phases from this
  directory. Frontmatter `report_filename` uses `{unit}` (str.format); bodies
  use `{{ unit }}` (Jinja2).
- **Tools are the only side-effect channel** (`tools.py`):
  `read_scratchpad`/`raise_question`/`update_memory` are closure-bound to
  their `(phase, unit)` so agents cannot write outside their own section;
  analyzers additionally get `list_files`/`read_file` bound to their repo
  (traversal-guarded). Tools return `"ERROR: ..."` strings, never raise -
  the framework feeds errors back to the model.
- **Live steering** (`steering.py`): the scratchpad is pulled once via
  `read_scratchpad`, and pushed thereafter. `ScratchpadWatch` (one per agent -
  analyzers run concurrently) is polled by function middleware after every
  tool call; a change is handed to the framework's
  `MessageInjectionMiddleware`, which drains it into the agent's next model
  call. An LLM turn cannot be interrupted, so a tool call is the earliest
  possible delivery point - which is why there is no file watcher. A cleared
  scratchpad advances the watermark but notifies nobody.
- **Bounded context** (`compaction.py`): every phase agent gets
  `Agent(compaction_strategy=..., default_options={"num_ctx": N})`; the
  pipeline is selective tool-group eviction -> LLM summarization -> the
  framework's deterministic fallback, budget `0.8 * (num_ctx - 1024)`,
  checked on every model call (tool-loop iterations included). Ranker and
  final-report agents get `num_ctx` only (single-turn). One console line
  when it fires; no artifact files.

## Rules and gotchas

- `review.py` must NOT use `from __future__ import annotations`:
  `@response_handler` validates `ctx` via `inspect.signature`, and string
  annotations break it (see the note at the top of that file).
- Model-output hygiene lives in `phases.py`: `_clean_text` strips leaked
  `<|...|>` template tokens; `_invoke_report` retries once on empty text.
  The retry only sees the earlier exploration because `_run_initial` /
  `on_revision` mint one `AgentSession` per cycle and `_invoke` passes it to
  every `run()` - `Agent.run(session=None)` is stateless per call, so this
  MUST stay explicit. The memory nudge is concatenated, never
  `str.format`-ed - reports contain literal braces.
- Never create `tests/__init__.py`: pytest imports tests as top-level
  modules and `from conftest import ...` depends on it.
- Tests are LLM-free by default (`addopts = "-m 'not ollama'"`).
  Decision-bearing logic belongs in pure functions; executor tests use
  `FakeCtx`/`FakeAgent` from `tests/conftest.py` by calling handler methods
  directly.
- `sample_data/docs_src/*.md` must stay ASCII (fpdf2 core fonts are
  latin-1). After editing, regenerate the PDFs and keep
  `tests/test_sample_data.py` green - the planted gaps/conflicts it asserts
  ARE the demo's question fuel.
- Markdown lint (`pymarkdownlnt` via `.pymarkdown.json`) covers `README.md`,
  `CLAUDE.md`, and the prompts directory only; the spec/plan under `docs/`
  are historical and excluded.
- CLI stays stdlib (`argparse`/`print`); the dependency set is deliberately
  minimal.
- Review-gate progress must stay LEDGER-derived. The framework only persists
  executor state registered via `ctx.set_executor_state` (we use none), so a
  resumed gate is a fresh instance: an in-memory counter made every answer
  look like the last (discovery revised 5x). `test_checkpoint.py`'s cycle
  test guards this.
- Checkpoints are pickled behind `ALLOWED_CHECKPOINT_TYPES` (pipeline.py). A
  type missing from it does NOT raise - `list_checkpoints` returns `[]`, so
  resume silently "finds no checkpoints". The allowlist is derived from the
  message classes and completeness-tested; keep new message dataclasses in
  `_MESSAGE_TYPES`.
