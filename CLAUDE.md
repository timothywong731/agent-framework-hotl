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
poetry run demo --max-questions 5              # review-gate slot budget (0 = never pause)
poetry run reflexion                           # reflexion demo: grounded critic, cyclic graph
poetry run reflection                          # reflection A/B foil: tool-less judge
poetry run reflection --max-tool-calls 4       # quickest tour of the countdown
poetry run pytest                              # fast, LLM-free; includes the markdown lint gate
poetry run pytest tests/test_review.py -v      # one file
poetry run pytest tests/test_review.py::test_review_once_guard -v   # one test
OLLAMA_E2E=1 poetry run pytest -m ollama -s    # live E2E, ~10 min on gemma4:31b
poetry run pymarkdown --config .pymarkdown.json scan README.md CLAUDE.md src/hotl_demo/prompts src/reflexion_demo/prompts src/reflection_demo/prompts
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
  Open questions COMPETE for `--max-questions` slots (default 3): a tool-less
  ranker agent orders them by expected swing on the final report (raise order
  carries no signal); losers are marked `deferred` - terminal, default
  applied - BEFORE the pause, so checkpoints and `review.jsonl` reflect the
  competition. Ranker output is fenced: `validate_ranking` -> one retry ->
  deterministic `(importance, id)` fallback.
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
- **Reflection's tool budget** (`reflection_demo/budget.py`): per **pass**,
  not per turn - `AgentLoopMiddleware` runs every pass inside one
  `agent.run()`, so there is no turn boundary at which to mint a fresh
  counter. `PassBudget.start_pass()` is called from `next_message` instead,
  the one hook that fires between passes. `remove_tools()` is only
  reachable from inside a tool call, so it cannot strip pre-emptively: the
  final pass strips read tools after its first call rather than before it.
  Countdown wording is byte-identical to `reflexion_demo/budget.py`'s, and so
  is the budget paragraph in the two `prompts/worker.md` templates - both
  workers are told the `max_tool_calls` NUMBER, since a worker that knows it
  has 12 can plan a 12-file sweep and one told "a limited number" cannot.
  `tests/test_budget_wording_parity.py` pins the constants and the rendered
  paragraph; the constants alone stayed green through the prompt drift.
  `build_agent` does NOT prime pass 1's budget - `PassBudget` is born
  `spent=0, finalizing=False`, which is pass 1's state at every
  `--max-passes`. Priming a `--max-passes 1` run finalizing stripped its read
  tools on the FIRST call while the prompt had just promised N of them, and
  `next_message` never fires on such a run, so `finalize.md` never explained
  it either.

## Upstream samples (microsoft/agent-framework)

The framework's own examples are the fastest answer to "how is this meant to
be done", and they move faster than the prose docs:
<https://github.com/microsoft/agent-framework/tree/main/python/samples>.
Read them without cloning:

```bash
# the whole file list
gh api 'repos/microsoft/agent-framework/git/trees/main?recursive=1' \
  --jq '.tree[] | select(.path|startswith("python/samples/")) | .path'
# one file
curl -sfL https://raw.githubusercontent.com/microsoft/agent-framework/main/python/samples/<path>
```

Layout: `01-get-started` (8 numbered walkthroughs), `02-agents` (the largest
tree - middleware, tools, compaction, context providers, chat clients,
skills), `03-workflows` (graphs: `control-flow`, `checkpoint`,
`human-in-the-loop`, `orchestrations`, `parallelism`, `composition`,
`declarative` YAML), `04-hosting`, `05-end-to-end`, plus
`autogen-migration` and `semantic-kernel-migration`. `SAMPLE_GUIDELINES.md`
at that root is the house style the samples themselves follow.

Where this repo's pieces come from:

| This repo | Sample |
|---|---|
| reflexion's worker `<->` reviewer cycle | `03-workflows/agents/workflow_as_agent_reflection_pattern.py`, `03-workflows/control-flow/simple_loop.py` |
| `isinstance` edge conditions | `03-workflows/control-flow/edge_condition.py`, `switch_case_edge_group.py`, `multi_selection_edge_group.py` |
| HOTL fan-out one analyzer per repo, then join | `03-workflows/parallelism/fan_out_fan_in_edges.py`, `aggregate_results_of_different_types.py` |
| review gate (`ctx.request_info`) | `03-workflows/human-in-the-loop/sequential_request_info.py`, `concurrent_request_info.py` |
| `--pause` / `--resume` | `03-workflows/checkpoint/checkpoint_with_resume.py`, `checkpoint_with_human_in_the_loop.py` |
| compaction strategies | `02-agents/compaction/` (`basics.py`, `summarization.py`, `custom.py`, `agent_client_overrides.py`) |
| scratchpad steering push | `02-agents/middleware/message_injection_middleware.py` |
| tool budget, `remove_tools`, early stop | `02-agents/middleware/function_based_middleware.py`, `middleware_termination.py` |
| single-agent reflection loops | `02-agents/middleware/agent_loop_middleware_judge.py`, `agent_loop_middleware_refinement.py`, `agent_loop_middleware_todos.py` |

`AgentLoopMiddleware` (public export, `@experimental`) is the framework's
native single-agent loop: `should_continue` for rule-based termination,
`.with_judge(client, criteria=[...])` for a tool-less LLM critic returning
`JudgeVerdict(answered, reasoning)`. Gotcha worth knowing before designing
around it: `max_iterations` short-circuits **before** `should_continue`
(`_harness/_loop.py`), so on the capped pass the judge is never consulted.

Naming warning: `05-end-to-end/evaluation/self_reflection/` is a
*self-reflection* sample that cites the *Reflexion* paper. Upstream uses the
two terms interchangeably; this repo does not - see the README section
contrasting them.

Two caveats when lifting sample code: samples nearly always use
`FoundryChatClient` + `AzureCliCredential`, so swap in a no-arg
`OllamaChatClient()` and keep the shape; and none of them pin `num_ctx`,
which everything here must, or Ollama silently truncates.

## Rules and gotchas

- `review.py` must NOT use `from __future__ import annotations`:
  `@response_handler` validates `ctx` via `inspect.signature`, and string
  annotations break it (see the note at the top of that file).
- **`FunctionInvocationContext.result` is `list[Content]`, not a string**
  (both demos' `budget.py`, `_append_note`): appending a coaching note with
  an f-string (`f"{context.result}\n\n{note}"`) stringifies the list to
  Python reprs (`[<agent_framework._types.Content object at 0x...>]`) and
  destroys the tool's actual output - the model receives a repr instead of
  the file it asked for. Caught only by a live run; it predates this branch
  in `reflexion_demo` too. Append to the last text `Content` in place
  instead of flattening the list to a string. Anyone writing function
  middleware that annotates a tool result needs to know this.
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
  `CLAUDE.md`, and all three `src/*/prompts` directories only; the spec/plan
  under `docs/` are historical and excluded. `tests/test_markdown_lint.py`
  IS the gate - keep the command above matching its target list, or a green
  local run turns into a red suite.
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
- `deferred` is not a verdict: `already_resumed` must only count
  answered/declined, because the gate writes deferrals BEFORE any resume
  exists. `Importance`/`QuestionStatus`/`Phase` are `(str, Enum)` in
  `artifacts.py` - plain strings on disk; compare with enum members, and
  keep `enum.StrEnum` out (3.11+, floor is 3.10).
