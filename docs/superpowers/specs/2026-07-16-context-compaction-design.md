# Context Compaction — Design

**Date:** 2026-07-16
**Status:** Approved
**Extends:** `2026-07-14-hotl-pipeline-design.md` (§ agents, § testing) — authoritative for everything else.
**Verified against:** `agent-framework-core` 1.11.0 as installed in this repo's venv. Claims below were established by reading the installed source (`_compaction.py`, `_clients.py`, `_agents.py`, `agent_framework_ollama/_chat_client.py`), not documentation. One layered claim (§2, last row) should be confirmed by a small spike during implementation.

## 1. Purpose

Nothing bounds an agent's context today. `PhaseExecutor` builds its `Agent` with no
`compaction_strategy` and no `context_providers`, so the framework's auto-added
`InMemoryHistoryProvider` accumulates every turn of a run cycle unbounded: the initial
prompt, every `read_file`/`raise_question` tool call and result, the report, the memory
nudge, the retry. When that exceeds the model's window, **Ollama silently truncates the
oldest tokens** — the system instructions and phase prompt — and the failure mode is
off-task or empty reports with no error anywhere. A single `read_file` result (capped at
20k chars ≈ 5k tokens) plus the initial prompt can already exceed Ollama's default
`num_ctx` of 4096, so this is not hypothetical even on the sample data.

This spec adds a bounded context with two goals, per the brainstorm:

1. **Robustness** — the pipeline never sends more than the window; degradation is
   controlled and chosen, not silent server-side truncation.
2. **Pedagogy** — compaction is a visible part of the demo narrative: one console line
   when it fires, a README section explaining it. Console only; no new artifact files.

## 2. What the framework provides (established facts)

| Question | Finding |
|---|---|
| Does anything compact by default? | **No.** Compaction is fully opt-in; without it the whole history goes to the model every call |
| Where can a strategy be attached? | Chat client ctor, `Agent` ctor, or per-`run()` call (`compaction_strategy=`); agent overrides client, run overrides agent |
| When does it run? | Inside `ChatClient.get_response` via `_prepare_messages_for_model_call` — i.e. **per model call** |
| Does that cover the tool loop? | The function-invocation layer issues each tool-loop iteration through `get_response`, so compaction applies per iteration. **Confirm with a spike** (count strategy invocations across a multi-tool fake run) |
| Does compaction delete messages? | **No.** Messages are annotated into atomic groups (system / user / assistant_text / tool_call) and marked `_excluded`; only the included projection is sent. A tool call is never split from its results |
| What happens if the summarizer fails? | `SummarizationStrategy` logs a warning and reports "no change"; `TokenBudgetComposedStrategy`'s deterministic fallback then excludes oldest groups until under budget. Compaction cannot crash a run |
| Can we make Ollama honor our budget? | Yes — `num_ctx` is a typed `OllamaChatOptions` key and `Agent` accepts `default_options`. (Correction found during implementation: `OllamaChatClient.__init__` does NOT — the summarizer pins `num_ctx` via a per-call `get_response(options=...)` shim instead) |

## 3. Design

### 3.1 One new module: `src/hotl_demo/compaction.py`

Same one-concern-per-module shape as `steering.py`. ~60 lines. Exposes:

- `resolve_num_ctx() -> int` — `OLLAMA_NUM_CTX` env var, default **4096** (Ollama's
  server default). A non-integer value fails fast at startup (plain `int()`, no
  swallowing). `main.py` gains a `--num-ctx` flag that sets the env var, mirroring
  `--model` → `OLLAMA_MODEL`.
- `build_compaction_strategy(label: str, num_ctx: int) -> CompactionStrategy` — builds
  the hybrid pipeline below, wrapped in the logging decorator.

### 3.2 Wiring (the only `phases.py` change)

In `PhaseExecutor.__init__`, **inside the real-`Agent` branch only** (the `agent` test
seam constructs nothing Ollama-related):

```python
Agent(
    client=OllamaChatClient(),
    ...,
    default_options={"num_ctx": num_ctx},
    compaction_strategy=build_compaction_strategy(spec.executor_id, num_ctx),
)
```

`default_options={"num_ctx": ...}` is load-bearing: without it the server truncates at
its own default regardless of what we budget for. The phase agents' `OllamaChatClient`
stays no-arg.

The ranker (`review.py`) and final-report (`report.py`) agents additionally get
`default_options={"num_ctx": resolve_num_ctx()}` — no compaction strategy
(single-turn, no history to compact), but the server must honor the same
window or their large prompts are truncated at Ollama's own default.

### 3.3 The hybrid pipeline

`TokenBudgetComposedStrategy(token_budget, tokenizer=CharacterEstimatorTokenizer(), early_stop=True)`
with two stages, in order:

1. **Deterministic first:** `SelectiveToolCallCompactionStrategy(keep_last_tool_call_groups=2)`
   — excludes older tool-call groups entirely, keeping the newest 2 verbatim. (The spec
   originally named `ToolResultCompactionStrategy`, but its one-line summary embeds the
   full result text — for 20k-char `read_file` results it reclaims almost nothing.
   Exclusion is safe here: durable findings live in `memory.json` and the report draft,
   and a file can always be re-read.)
2. **LLM summarization second, only if still over budget:**
   `SummarizationStrategy(client=_NumCtxSummarizer(num_ctx))` — a lazy shim over
   `OllamaChatClient` that injects `options={"num_ctx": ...}` per call (the client
   ctor takes no default options in 1.11.0)
   — same model via `OLLAMA_MODEL`, framework-default prompt, but explicit
   `target_count=2, threshold=0`: the framework default trigger (more than ~6
   non-system messages) is count-based and never fires for this repo's profile — a few
   huge `read_file` messages; these counts make stage 2 engage whenever the composed
   budget check invokes it. Its own client carries `num_ctx` so the summarization call
   is not itself truncated. One summarizer client per executor (it is a stateless HTTP
   wrapper; nothing is shared).
3. **Fallback (built into the composed strategy):** oldest-first group exclusion until
   under budget. The budget is therefore a hard guarantee.

System instructions survive except in the framework's strict last-resort path.

### 3.4 Budget math

One formula, one knob:

```python
token_budget = int(0.8 * (num_ctx - 1024))
```

1024 tokens reserved for model output; 20% margin because `CharacterEstimatorTokenizer`
is a 4-chars/token heuristic, not a real tokenizer. At the default `num_ctx=4096` the
budget is ~2.4k tokens — sample-data analyzers genuinely cross it, so **the demo shows
real compaction with no artificial demo flag**.

### 3.5 Console visibility

`_LoggedStrategy` implements the framework's `CompactionStrategy` protocol (one async
`__call__`), delegates to the composed pipeline, and prints one line in the existing
demo style **only when compaction changed something**:

```text
  deep_analysis:oms-monolith: compacted context 14 -> 6 messages (~3.1k -> 1.9k tokens)
```

Silent when under budget. Message/token counts come from the framework's
`included_messages` / `included_token_count` helpers. No artifact file.

## 4. Behavior changes and edge cases

- **Report retry / memory nudge** currently "see the whole earlier exploration"
  (`phases.py` docstrings). After this change they see the *compacted* exploration —
  old tool-call groups evicted, plus a summary if stage 2 fired. Intended trade;
  update the docstrings and the CLAUDE.md note to match.
- **Steering injections** are ordinary user-group messages: recent when injected, so
  they survive; an old one can eventually be summarized away. Acceptable — the model
  can always re-read the scratchpad via its tool.
- **Revisions** mint a fresh session with a self-contained prompt; compaction is
  normally idle there and exists only as a guard.
- **Summarizer failure** (Ollama down mid-run, empty output): warning + skip + fallback
  still enforces the budget. Compaction never fails the pipeline.

## 5. Testing (LLM-free, repo convention)

New `tests/test_compaction.py`, driving the composed strategy directly with synthetic
message lists and a `FakeSummaryClient` (canned `get_response`, records calls):

1. Over-budget tool-heavy history → old tool groups excluded, newest 2 verbatim,
   included tokens ≤ budget.
2. Stage 2 fires only when stage 1 is insufficient (assert via the fake's call log).
3. Summarizer raises / returns empty → still ≤ budget, no exception.
4. Log line printed on change; silence when under budget (`capsys`).
5. `resolve_num_ctx`: default, env override, `--num-ctx` sets the env var.

Plus the §2 spike-as-test: an integration check that the strategy is invoked once per
model call across a multi-tool run. This one cannot use the `FakeAgent` seam (it
bypasses the client layer where compaction lives); build a real `Agent` over a scripted
chat client instead, following `test_pipeline.py`'s `_DriveAgent` technique.
We test **our composition and wrapper**, not framework internals. The `OLLAMA_E2E=1`
live test is unchanged and now implicitly exercises compaction.

## 6. Documentation updates

- **README:** short "Bounded context" section — what fires when, the log line to look
  for, the `OLLAMA_NUM_CTX` / `--num-ctx` knob.
- **CLAUDE.md:** add `compaction.py` to the architecture list; add `OLLAMA_NUM_CTX` to
  the commands note; amend the "OllamaChatClient is always constructed no-arg" rule —
  phase-agent clients stay no-arg, the summarizer client carries `default_options` only.

## 7. Out of scope

- Compacting `memory.json` / prompt-side growth (`memory_text`, `open_questions`) —
  prompt inputs, not chat history; the bounded-ledger spec owns question growth.
- Real tokenizers (tiktoken) — the 20% margin covers the heuristic; revisit only if
  truncation is observed despite compaction.
- `CompactionProvider` / history-provider changes — sessions are per-cycle and short;
  the in-run hook covers the actual pressure point.
- Persisted compaction artifacts or report annotations — console only, per brainstorm.
