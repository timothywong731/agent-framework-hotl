# Context Compaction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bound every phase agent's context with a hybrid compaction pipeline (deterministic tool-group eviction → LLM summarization → hard fallback), sized by `OLLAMA_NUM_CTX`, with one console line when it fires.

**Architecture:** One new module `src/hotl_demo/compaction.py` builds a `TokenBudgetComposedStrategy` and wraps it in a logging decorator; `PhaseExecutor` passes it to `Agent(compaction_strategy=..., default_options={"num_ctx": N})`. The ranker and final-report agents get `default_options={"num_ctx": N}` only (single-turn — history compaction has nothing to compact there). Spec: `docs/superpowers/specs/2026-07-16-context-compaction-design.md`.

**Tech Stack:** Python 3.13, `agent-framework-core` 1.11.0 (already installed), pytest, Ollama (live E2E only).

## Global Constraints

- Tests are LLM-free by default (`addopts = "-m 'not ollama'"`); never create `tests/__init__.py`.
- `review.py` must not gain `from __future__ import annotations` (not touched here, but do not "fix" it in passing).
- CLI stays stdlib (`argparse`/`print`).
- Markdown lint gate covers `README.md`, `CLAUDE.md`, `src/hotl_demo/prompts` — run `poetry run pymarkdown --config .pymarkdown.json scan README.md CLAUDE.md src/hotl_demo/prompts` after editing README/CLAUDE.md.
- All commands run via `poetry run ...` from the repo root.
- Framework facts (verified against installed source, do not re-derive): compaction strategies attach per-`Agent` via `compaction_strategy=`; they run inside `BaseChatClient.get_response`, which the tool loop (`FunctionInvocationLayer`) calls once per model iteration; `TokenBudgetComposedStrategy` ends with a deterministic oldest-first fallback, so the budget is a hard guarantee; `SummarizationStrategy` swallows summarizer failures (warn + skip).

---

### Task 0: Amend the spec — stage 1 is Selective eviction, not ToolResult collapse

**Why:** During plan research we read `ToolResultCompactionStrategy`'s source: its `[Tool results: ...]` summary embeds the **full result text** — it removes message overhead only. For this repo's dominant growth (up to 20k-char `read_file` results) it saves ~nothing. `SelectiveToolCallCompactionStrategy` *excludes* old tool-call groups entirely: real deterministic savings; old findings survive via `memory.json` and the report draft, and files can be re-read.

**Files:**
- Modify: `docs/superpowers/specs/2026-07-16-context-compaction-design.md`

- [ ] **Step 1: Edit §3.3 stage 1**

Replace the stage-1 bullet with:

```markdown
1. **Deterministic first:** `SelectiveToolCallCompactionStrategy(keep_last_tool_call_groups=2)`
   — excludes older tool-call groups entirely, keeping the newest 2 verbatim. (The spec
   originally named `ToolResultCompactionStrategy`, but its one-line summary embeds the
   full result text — for 20k-char `read_file` results it reclaims almost nothing.
   Exclusion is safe here: durable findings live in `memory.json` and the report draft,
   and a file can always be re-read.)
```

- [ ] **Step 2: Edit §5 test item 1**

Replace `old tool groups collapsed, newest 2 verbatim,` with `old tool groups excluded, newest 2 verbatim,`.

- [ ] **Step 3: Edit §3.3 stage 2 — explicit trigger counts**

Replace `framework-default prompt and counts` with `framework-default prompt, but explicit `target_count=2, threshold=0``. Append to that bullet:

```markdown
   The framework default trigger (more than ~6 non-system messages) is
   count-based and never fires for this repo's profile — a few huge
   `read_file` messages; `target_count=2, threshold=0` makes stage 2 engage
   whenever the composed budget check invokes it.
```

- [ ] **Step 4: Edit §3.2 — ranker and final-report agents**

Append to §3.2:

```markdown
The ranker (`review.py`) and final-report (`report.py`) agents additionally get
`default_options={"num_ctx": resolve_num_ctx()}` — no compaction strategy
(single-turn, no history to compact), but the server must honor the same
window or their large prompts are truncated at Ollama's own default.
```

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/specs/2026-07-16-context-compaction-design.md
git commit -m "docs: compaction spec - stage 1 is selective eviction (ToolResult collapse keeps full text)"
```

---

### Task 1: `compaction.py` — `resolve_num_ctx` and `token_budget` (pure functions)

**Files:**
- Create: `src/hotl_demo/compaction.py`
- Create: `tests/test_compaction.py`

**Interfaces:**
- Produces: `resolve_num_ctx() -> int` (env `OLLAMA_NUM_CTX`, default 4096, `int()` fail-fast); `token_budget(num_ctx: int) -> int` = `int(0.8 * (num_ctx - 1024))`. Module constants `DEFAULT_NUM_CTX = 4096`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_compaction.py`:

```python
"""Compaction wiring tests: budget math, strategy composition, logging.

All LLM-free: strategies are driven directly on synthetic message lists,
the summarizer is a recorded fake (SummarizationStrategy only needs
``await client.get_response(...)`` returning an object with ``.text``).
"""
import pytest

from hotl_demo.compaction import DEFAULT_NUM_CTX, resolve_num_ctx, token_budget


def test_resolve_num_ctx_default(monkeypatch):
    monkeypatch.delenv("OLLAMA_NUM_CTX", raising=False)
    assert resolve_num_ctx() == DEFAULT_NUM_CTX == 4096


def test_resolve_num_ctx_env_override(monkeypatch):
    monkeypatch.setenv("OLLAMA_NUM_CTX", "16384")
    assert resolve_num_ctx() == 16384


def test_resolve_num_ctx_garbage_fails_fast(monkeypatch):
    monkeypatch.setenv("OLLAMA_NUM_CTX", "lots")
    with pytest.raises(ValueError):
        resolve_num_ctx()


def test_token_budget_reserves_output_and_margin():
    # 0.8 * (4096 - 1024) = 2457.6 -> 2457
    assert token_budget(4096) == 2457
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/test_compaction.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hotl_demo.compaction'`

- [ ] **Step 3: Write the module skeleton**

Create `src/hotl_demo/compaction.py`:

```python
"""Bounded context: the hybrid compaction pipeline for phase agents.

Nothing in agent-framework bounds context by default; without this module the
whole session history goes to Ollama every call and the server silently
truncates the OLDEST tokens (system prompt first) at ``num_ctx``. See
``docs/superpowers/specs/2026-07-16-context-compaction-design.md``.

Pipeline (composed, early-stop, checked on EVERY model call, including each
tool-loop iteration): selective eviction of old tool-call groups -> LLM
summarization of what remains -> the framework's deterministic oldest-first
fallback. The token budget is therefore a hard guarantee.
"""
from __future__ import annotations

import os

DEFAULT_NUM_CTX = 4096
_OUTPUT_RESERVE = 1024   # tokens left for the model's own output
_BUDGET_FRACTION = 0.8   # margin: the 4-chars/token estimator is a heuristic
_KEEP_TOOL_GROUPS = 2    # newest tool-call groups kept verbatim


def resolve_num_ctx() -> int:
    """Read the model context window from ``OLLAMA_NUM_CTX``.

    Returns:
        Window size in tokens; 4096 (Ollama's server default) when unset.

    Raises:
        ValueError: Non-integer value - fail fast at startup, never mid-run.

    Example:
        >>> import os; os.environ.pop("OLLAMA_NUM_CTX", None) and None
        >>> resolve_num_ctx()
        4096
    """
    return int(os.environ.get("OLLAMA_NUM_CTX", DEFAULT_NUM_CTX))


def token_budget(num_ctx: int) -> int:
    """Included-token budget for a window: reserve output, keep a margin.

    Example:
        >>> token_budget(4096)
        2457
    """
    return int(_BUDGET_FRACTION * (num_ctx - _OUTPUT_RESERVE))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/test_compaction.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add src/hotl_demo/compaction.py tests/test_compaction.py
git commit -m "feat: compaction budget math (OLLAMA_NUM_CTX, output reserve, estimator margin)"
```

---

### Task 2: `build_compaction_strategy` — pipeline composition and console logging

**Files:**
- Modify: `src/hotl_demo/compaction.py`
- Modify: `tests/test_compaction.py`

**Interfaces:**
- Consumes: `token_budget` from Task 1.
- Produces: `build_compaction_strategy(label: str, num_ctx: int, summarizer=None) -> CompactionStrategy` — the object is an async callable `(list[Message]) -> bool` (the framework's `CompactionStrategy` protocol). `summarizer` is a test seam (anything with `async get_response(messages, stream=False)` returning an object with `.text`); production default is `OllamaChatClient(default_options={"num_ctx": num_ctx})`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_compaction.py`:

```python
import asyncio

from agent_framework import Content, Message, included_messages, included_token_count

from hotl_demo.compaction import build_compaction_strategy


class FakeSummaryClient:
    """Duck-typed summarizer: SummarizationStrategy only calls
    ``await client.get_response(messages, stream=False)`` and reads ``.text``."""

    def __init__(self, text="SUMMARY OF OLDER CONTEXT", error=None):
        self.calls = []
        self._text = text
        self._error = error

    async def get_response(self, messages, stream=False):
        self.calls.append(messages)
        if self._error is not None:
            raise self._error

        class _Resp:
            text = self._text
        return _Resp()


def _tool_group(i, payload):
    """One atomic tool-call group: assistant function_call + tool result."""
    return [
        Message(role="assistant", contents=[Content(
            type="function_call", call_id=f"c{i}", name="read_file",
            arguments='{"path": "f"}')]),
        Message(role="tool", contents=[Content(
            type="function_result", call_id=f"c{i}", result=payload)]),
    ]


def _history(groups, payload_chars):
    msgs = [Message(role="system", contents=["phase instructions"]),
            Message(role="user", contents=["initial prompt"])]
    for i in range(groups):
        msgs.extend(_tool_group(i, "x" * payload_chars))
    return msgs


def test_over_budget_evicts_old_tool_groups_keeps_newest_two():
    fake = FakeSummaryClient()
    strategy = build_compaction_strategy("t", 4096, summarizer=fake)
    # 8 groups x ~1000 tokens >> 2457 budget; eviction alone suffices.
    messages = _history(groups=8, payload_chars=4000)
    changed = asyncio.run(strategy(messages))
    assert changed
    assert included_token_count(messages) <= 2457
    kept = included_messages(messages)
    kept_call_ids = {c.call_id for m in kept for c in m.contents
                     if c.type == "function_call"}
    assert kept_call_ids == {"c6", "c7"}          # newest 2 groups verbatim
    assert fake.calls == []                        # stage 2 never needed
    assert kept[0].text == "phase instructions"    # system group preserved


def test_summarizer_fires_only_when_eviction_insufficient():
    fake = FakeSummaryClient()
    strategy = build_compaction_strategy("t", 4096, summarizer=fake)
    # 2 giant groups: eviction keeps both (keep_last=2) and stays over budget.
    messages = _history(groups=2, payload_chars=12000)
    asyncio.run(strategy(messages))
    assert len(fake.calls) >= 1                    # stage 2 engaged
    assert included_token_count(messages) <= 2457  # budget still enforced


def test_summarizer_failure_never_raises_budget_still_enforced():
    fake = FakeSummaryClient(error=RuntimeError("ollama down"))
    strategy = build_compaction_strategy("t", 4096, summarizer=fake)
    messages = _history(groups=2, payload_chars=12000)
    changed = asyncio.run(strategy(messages))      # must not raise
    assert changed
    assert included_token_count(messages) <= 2457  # fallback did the work


def test_under_budget_is_untouched_and_silent(capsys):
    fake = FakeSummaryClient()
    strategy = build_compaction_strategy("t", 4096, summarizer=fake)
    messages = _history(groups=1, payload_chars=200)
    changed = asyncio.run(strategy(messages))
    assert not changed
    assert len(included_messages(messages)) == len(messages)
    assert capsys.readouterr().out == ""


def test_log_line_on_change(capsys):
    strategy = build_compaction_strategy(
        "deep_analysis:oms-monolith", 4096, summarizer=FakeSummaryClient())
    messages = _history(groups=8, payload_chars=4000)
    asyncio.run(strategy(messages))
    out = capsys.readouterr().out
    assert out.count("\n") == 1
    assert out.startswith("  deep_analysis:oms-monolith: compacted context ")
    assert "messages" in out and "tokens" in out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/test_compaction.py -v`
Expected: new tests FAIL with `ImportError: cannot import name 'build_compaction_strategy'`

- [ ] **Step 3: Implement the factory and logging wrapper**

Append to `src/hotl_demo/compaction.py` (extend the existing import block):

```python
from typing import Any

from agent_framework import (
    CharacterEstimatorTokenizer,
    SelectiveToolCallCompactionStrategy,
    SummarizationStrategy,
    TokenBudgetComposedStrategy,
    annotate_message_groups,
    included_messages,
    included_token_count,
)


class _LoggedStrategy:
    """CompactionStrategy-protocol wrapper: delegate; one line when it acted.

    Console-only by design (see spec §3.5) - no artifact files.
    """

    def __init__(self, inner: Any, label: str, tokenizer: Any) -> None:
        self._inner = inner
        self._label = label
        self._tokenizer = tokenizer

    async def __call__(self, messages: list) -> bool:
        # Annotate up front so the "before" numbers are real - the composed
        # strategy would otherwise annotate after we counted.
        annotate_message_groups(messages, tokenizer=self._tokenizer)
        msgs_before = len(included_messages(messages))
        toks_before = included_token_count(messages)
        changed = await self._inner(messages)
        if changed:
            print(f"  {self._label}: compacted context "
                  f"{msgs_before} -> {len(included_messages(messages))} messages "
                  f"(~{toks_before} -> {included_token_count(messages)} tokens)")
        return changed


def build_compaction_strategy(label: str, num_ctx: int, summarizer: Any = None):
    """Build the hybrid pipeline for one agent.

    Args:
        label: Executor id used in the console line.
        num_ctx: Model window in tokens; drives the budget.
        summarizer: Test seam - anything with ``async get_response(messages,
            stream=False)`` returning ``.text``. Defaults to a dedicated
            ``OllamaChatClient`` carrying ``num_ctx`` so the summarization
            call is not itself truncated.

    Returns:
        An async ``(messages) -> bool`` satisfying the framework's
        ``CompactionStrategy`` protocol.
    """
    if summarizer is None:
        from agent_framework.ollama import OllamaChatClient
        summarizer = OllamaChatClient(default_options={"num_ctx": num_ctx})
    tokenizer = CharacterEstimatorTokenizer()
    composed = TokenBudgetComposedStrategy(
        token_budget=token_budget(num_ctx),
        tokenizer=tokenizer,
        strategies=[
            SelectiveToolCallCompactionStrategy(
                keep_last_tool_call_groups=_KEEP_TOOL_GROUPS),
            # target_count=2/threshold=0: the default trigger (>~6 non-system
            # MESSAGES) is count-based and never fires for a history of a few
            # huge read_file results - exactly this repo's growth profile.
            SummarizationStrategy(client=summarizer, target_count=2, threshold=0),
        ],
        early_stop=True,
    )
    return _LoggedStrategy(composed, label, tokenizer)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/test_compaction.py -v`
Expected: all PASS. If `test_over_budget_...` fails on `kept_call_ids`, print `included_token_count(messages)` — group sizes vs the 2457 budget may need `payload_chars` nudged; the assertion that matters is `<= 2457` + newest-2-kept.

- [ ] **Step 5: Commit**

```bash
git add src/hotl_demo/compaction.py tests/test_compaction.py
git commit -m "feat: hybrid compaction pipeline (selective eviction -> summarization -> fallback) with console log"
```

---

### Task 3: Spike-as-test — the strategy runs once per model call across the tool loop

**Why:** The whole design hinges on compaction applying *inside* the tool loop (spec §2, flagged claim). This test proves it executably against the real `Agent` + `FunctionInvocationLayer` + `BaseChatClient` stack — no Ollama.

**Files:**
- Modify: `tests/test_compaction.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (framework-only).

- [ ] **Step 1: Write the test**

Append to `tests/test_compaction.py`:

```python
from agent_framework import Agent, BaseChatClient, ChatResponse, FunctionInvocationLayer


class ScriptedToolClient(FunctionInvocationLayer, BaseChatClient):
    """Two-beat script: call the 'ping' tool, then produce final text.

    Composes the same layers OllamaChatClient does (minus middleware and
    telemetry), so the tool loop and the compaction hook are the real ones.
    """

    def __init__(self):
        super().__init__()
        self.model_calls = 0

    async def _inner_get_response(self, *, messages, stream, options, **kwargs):
        self.model_calls += 1
        if self.model_calls == 1:
            return ChatResponse(messages=[Message(role="assistant", contents=[
                Content(type="function_call", call_id="c1", name="ping",
                        arguments="{}")])])
        return ChatResponse(messages=[Message(role="assistant", contents=["done"])])


class CountingStrategy:
    """No-op CompactionStrategy that counts invocations."""

    def __init__(self):
        self.calls = 0

    async def __call__(self, messages):
        self.calls += 1
        return False


def test_compaction_runs_once_per_model_call_in_tool_loop():
    def ping() -> str:
        """Reply with pong."""
        return "pong"

    client = ScriptedToolClient()
    strategy = CountingStrategy()
    agent = Agent(client=client, name="spike", instructions="use tools",
                  tools=[ping], compaction_strategy=strategy)
    result = asyncio.run(agent.run("go"))
    assert result.text == "done"
    assert client.model_calls == 2
    # THE design-carrying assertion: one compaction pass per model call,
    # i.e. compaction also fires between tool iterations.
    assert strategy.calls == 2
```

- [ ] **Step 2: Run it**

Run: `poetry run pytest tests/test_compaction.py::test_compaction_runs_once_per_model_call_in_tool_loop -v`
Expected: PASS. If `ScriptedToolClient` fails to construct (layer ctor requirements), mirror `OllamaChatClient`'s full base list from `agent_framework_ollama/_chat_client.py:287` — `FunctionInvocationLayer, ChatMiddlewareLayer, ChatTelemetryLayer, BaseChatClient` (the extra layers are pass-through mixins, all importable from `agent_framework`). If it still fails, STOP and re-read spec §2 — the design claim needs revisiting, do not paper over it.

- [ ] **Step 3: Commit**

```bash
git add tests/test_compaction.py
git commit -m "test: prove compaction fires per model call inside the tool loop"
```

---

### Task 4: Wire `PhaseExecutor`; window-size the ranker and report agents

**Files:**
- Modify: `src/hotl_demo/phases.py` (imports ~line 28; `Agent(...)` ctor at ~504-510; docstrings at ~465-469 and ~591-597)
- Modify: `src/hotl_demo/review.py:209-210` (`Agent(...)` for the ranker)
- Modify: `src/hotl_demo/report.py:118-119` (`Agent(...)` for the final report)
- Modify: `tests/test_compaction.py`

**Interfaces:**
- Consumes: `build_compaction_strategy(label, num_ctx)`, `resolve_num_ctx()`.

- [ ] **Step 1: Write the failing wiring test**

Append to `tests/test_compaction.py`:

```python
from pathlib import Path

from hotl_demo.artifacts import REPOS, ArtifactStore
from hotl_demo.phases import PhaseExecutor, build_phase_specs


def test_phase_executor_wires_compaction_and_num_ctx(monkeypatch, tmp_path):
    monkeypatch.setenv("OLLAMA_MODEL", "test-model")   # ctor reads env; no server contact
    monkeypatch.setenv("OLLAMA_NUM_CTX", "8192")
    spec = build_phase_specs(Path("sample_data"))[0]
    store = ArtifactStore(tmp_path / "run", REPOS)
    ex = PhaseExecutor(spec, store, scratchpad_path=tmp_path / "scratchpad.md")
    assert ex._agent.compaction_strategy is not None
    assert ex._agent.default_options.get("num_ctx") == 8192
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_compaction.py::test_phase_executor_wires_compaction_and_num_ctx -v`
Expected: FAIL — `compaction_strategy` is `None`.

- [ ] **Step 3: Wire `phases.py`**

Add to the `from .` import block (`src/hotl_demo/phases.py` ~line 28):

```python
from .compaction import build_compaction_strategy, resolve_num_ctx
```

Replace the `Agent(...)` construction (~lines 504-510):

```python
        num_ctx = resolve_num_ctx()
        self._agent = agent or Agent(
            client=OllamaChatClient(),  # model comes from OLLAMA_MODEL env var
            name=spec.executor_id.replace(":", "_"),
            instructions="You are one phase of a multi-agent assessment pipeline.",
            tools=tools,
            middleware=[injector, steering_mw],
            # Both halves matter: num_ctx makes Ollama honor the window we
            # budget for; without it the server truncates oldest-first at its
            # own default and compaction guards a window that does not exist.
            default_options={"num_ctx": num_ctx},
            compaction_strategy=build_compaction_strategy(spec.executor_id, num_ctx),
        )
```

(`agent or Agent(...)` short-circuits, so the `FakeAgent` seam still constructs nothing Ollama-related — Python only evaluates the `Agent(...)` branch when no seam is passed.)

- [ ] **Step 4: Update the two stale docstring claims in `phases.py`**

In the `PhaseExecutor` class docstring (~line 465): change
`the follow-up turn sees the whole earlier exploration` to
`the follow-up turn sees the earlier exploration (compacted when over budget - see compaction.py)`.

In `_invoke_report`'s docstring (~line 594): change
`The retry sees the full exploration because both turns share this cycle's session`
to `The retry shares this cycle's session, so it sees the (possibly compacted) exploration`.

- [ ] **Step 5: Window-size the ranker and report agents**

`src/hotl_demo/review.py` — add the import next to the existing `from .` imports, and extend the `Agent(...)` at line 209:

```python
from .compaction import resolve_num_ctx
```

```python
        self._ranker = ranker or Agent(
            client=OllamaChatClient(),  # model comes from OLLAMA_MODEL env var
            default_options={"num_ctx": resolve_num_ctx()},
```

(keep every other existing kwarg exactly as-is). Same two-line change in `src/hotl_demo/report.py` at line 118. No `compaction_strategy` for these two: single-turn agents have no accumulated history to compact; they just need the server to honor the same window.

- [ ] **Step 6: Run the full suite**

Run: `poetry run pytest`
Expected: all PASS (the wiring test now passes; every pre-existing test unaffected because fakes bypass the real-Agent branch).

- [ ] **Step 7: Commit**

```bash
git add src/hotl_demo/phases.py src/hotl_demo/review.py src/hotl_demo/report.py tests/test_compaction.py
git commit -m "feat: wire compaction + num_ctx into phase agents; window-size ranker and report agents"
```

---

### Task 5: `--num-ctx` CLI flag, README, CLAUDE.md

**Files:**
- Modify: `src/hotl_demo/main.py` (~line 236, after `--model`)
- Modify: `README.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add the flag (mirrors `--model` → env exactly)**

In `_amain()` after the `--model` argument:

```python
    parser.add_argument("--num-ctx", type=int,
                        default=int(os.environ.get("OLLAMA_NUM_CTX", 4096)),
                        help="Ollama context window in tokens; also sizes the "
                             "compaction budget (default: %(default)s)")
```

and after the existing `os.environ["OLLAMA_MODEL"] = args.model` line:

```python
    os.environ["OLLAMA_NUM_CTX"] = str(args.num_ctx)  # compaction.py reads this
```

No dedicated test: two lines of stdlib argparse mirroring `--model` (also untested), and `resolve_num_ctx` is already covered. The live E2E in Task 6 exercises it.

- [ ] **Step 2: README section**

Add after the steering section (match surrounding heading level):

```markdown
## Bounded context (compaction)

Local models have small windows (Ollama defaults to 4096 tokens) and, when a
conversation exceeds one, Ollama silently drops the OLDEST tokens - the system
prompt and task framing - producing off-task reports with no error anywhere.

Every phase agent therefore runs a compaction pipeline on each model call,
including between tool calls: old tool-call groups are evicted first
(newest 2 kept verbatim), the remainder is LLM-summarized if still over
budget, and a deterministic oldest-first fallback makes the budget a hard
guarantee. The budget is `0.8 * (num_ctx - 1024)`. Watch for:

    deep_analysis:oms-monolith: compacted context 14 -> 6 messages (~3124 -> 1893 tokens)

Size the window with `--num-ctx` (or `OLLAMA_NUM_CTX`); the same value is sent
to Ollama as `num_ctx`, so the server window and the compaction budget always
agree. Durable findings survive compaction by design: they live in
`memory.json` and the report drafts, not in chat history.
```

- [ ] **Step 3: CLAUDE.md updates**

1. Commands section: after the `OLLAMA_HOST` sentence add: `` `OLLAMA_NUM_CTX` (or `demo --num-ctx`) sizes both the Ollama window and the compaction budget. ``
2. Amend the same paragraph's `OllamaChatClient` rule to: `` `OllamaChatClient` is constructed no-arg except the compaction summarizer, which carries `default_options={"num_ctx": ...}` only. ``
3. Architecture list — add after the **Live steering** bullet:

```markdown
- **Bounded context** (`compaction.py`): every phase agent gets
  `Agent(compaction_strategy=..., default_options={"num_ctx": N})`; the
  pipeline is selective tool-group eviction -> LLM summarization -> the
  framework's deterministic fallback, budget `0.8 * (num_ctx - 1024)`,
  checked on every model call (tool-loop iterations included). Ranker and
  final-report agents get `num_ctx` only (single-turn). One console line
  when it fires; no artifact files.
```

- [ ] **Step 4: Lint gate**

Run: `poetry run pymarkdown --config .pymarkdown.json scan README.md CLAUDE.md src/hotl_demo/prompts`
Expected: no output (clean). Fix any violations it reports.

- [ ] **Step 5: Full suite once more**

Run: `poetry run pytest`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/hotl_demo/main.py README.md CLAUDE.md
git commit -m "feat: --num-ctx flag; document bounded context in README and CLAUDE.md"
```

---

### Task 6: Live verification (Ollama) and merge

- [ ] **Step 1: Live E2E**

Run (PowerShell): `$env:OLLAMA_E2E="1"; poetry run pytest -m ollama -s`
Expected: PASS in ~10 min on gemma4:31b. Watch stdout for `compacted context` lines — at the default `num_ctx=4096` the analyzers should genuinely trigger them. If **no** line appears AND the E2E passes, check the run's reports for coherence; then re-run with `OLLAMA_NUM_CTX=2048` to force pressure and confirm the log line appears. A crash or empty-report regression here is a STOP: diagnose (likely summarizer behavior on the local model) before merging.

- [ ] **Step 2: One interactive smoke run**

Run: `poetry run demo --pause` (answers land in `review.jsonl`; the run exits at the gate — no stdin needed). Confirm: run completes to the gate, reports exist under `output/run_<ts>/`, compaction lines interleave sanely with phase progress lines.

- [ ] **Step 3: Merge to main**

Per the worktree flow (superpowers:finishing-a-development-branch): from the main checkout,

```bash
git -C "C:/Users/Timothy Wong/Repositories/agent-framework-hotl" merge --no-ff feature/context-compaction -m "feat: bounded context via hybrid compaction"
poetry run pytest   # post-merge sanity in the main checkout
```

then remove the worktree (the plan assumes the worktree branch is named
`feature/context-compaction`; create it with that name).
