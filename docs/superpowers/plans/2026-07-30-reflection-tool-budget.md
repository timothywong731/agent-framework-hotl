# Reflection Tool Budget and Countdown Coaching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give `reflection_demo`'s worker a per-pass tool-call budget with
tool stripping and a forced-finalize pass, and give **both** demos
anticipatory countdown coaching in place of `reflexion_demo`'s single
after-the-fact nudge.

**Architecture:** A new `reflection_demo/budget.py` holds a `PassBudget` plus
function middleware that counts budgeted tool calls, appends escalating
countdown lines at 3/2/1 remaining, and strips the read tools when the budget
is spent or when the pass is the final one. Because
`AgentLoopMiddleware` runs every pass inside a single `agent.run()`, the
per-pass reset happens in `next_message` — the one hook that fires at a pass
boundary. `reflexion_demo/budget.py` gets the same countdown wording, enforced
byte-identical by a cross-demo parity test.

**Tech Stack:** Python ≥3.10, Poetry, pytest (`asyncio_mode = "auto"`),
`agent-framework ~=1.11`, Jinja2, Ollama `gemma4:31b`.

## Global Constraints

- Authoritative spec: `docs/superpowers/specs/2026-07-30-reflection-tool-budget-design.md`.
- **Standalone:** neither demo package may import from the other or from
  `hotl_demo`. The duplication of budget logic is deliberate and is guarded by
  the parity test in Task 2 — do not factor it into a shared module.
- **`COUNTDOWN` and `BUDGET_SPENT` must be byte-identical** in
  `src/reflection_demo/budget.py` and `src/reflexion_demo/budget.py`. Coaching
  only one worker would itself be a new confound in the A/B.
- **No `from __future__ import annotations`** anywhere in either demo package.
- Tools return `"ERROR: ..."` strings, never raise. Middleware never raises.
- CLI stays stdlib (`argparse` / `print`). No new dependencies.
- Tests are LLM-free by default (`addopts = "-m 'not ollama'"`). Never create
  `tests/__init__.py`.
- Markdown under both `prompts/` directories must pass the lint gate.
- Vocabulary: a reflection **pass** is one agent run; a reflexion **cycle** is
  a draft plus its review. Never conflate them.
- **Inline comments must explain WHY, never narrate syntax**, and must never
  describe behaviour the code lacks.
- Do not filter the `ExperimentalWarning` from `AgentLoopMiddleware`.
- Two verified framework facts this design rests on — do not "fix" code that
  depends on them: tools **re-arm on every pass** inside one `agent.run()`,
  and `FunctionInvocationContext.remove_tools()` is only reachable from inside
  a tool call, so tools cannot be stripped pre-emptively.
- Commit after every task.

## File Structure

| File | Responsibility |
|---|---|
| `src/reflection_demo/budget.py` | **new** — `PassBudget`, countdown constants, budget middleware |
| `src/reflexion_demo/budget.py` | modify — replace `BUDGET_NUDGE` with the shared countdown |
| `src/reflection_demo/prompts/finalize.md` | **new** — final-pass instruction |
| `src/reflection_demo/prompts/worker.md` | modify — state the real budget |
| `src/reflection_demo/prompting.py` | modify — `render_finalize_message` |
| `src/reflection_demo/judging.py` | modify — `make_next_message` gains pass-boundary duties |
| `src/reflection_demo/main.py` | modify — `--max-tool-calls`, wire the middleware |
| `src/reflexion_demo/prompts/worker.md` | modify — mention the countdown |

---

### Task 1: `reflection_demo/budget.py`

**Files:**

- Create: `src/reflection_demo/budget.py`
- Test: `tests/test_reflection_budget.py`

**Interfaces:**

- Consumes: nothing from other tasks.
- Produces: `BUDGETED_TOOL_NAMES: frozenset`; `COUNTDOWN: dict[int, str]`;
  `BUDGET_SPENT: str`; `PassBudget(max_calls, spent=0, finalizing=False)` with
  a `remaining` property and `start_pass(*, finalizing: bool) -> None`;
  `make_budget_middleware(budget, budgeted, label) -> middleware`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_reflection_budget.py`:

```python
"""Per-pass tool budget: counting, countdown coaching, strip, finalize."""
from reflection_demo.budget import (
    BUDGET_SPENT,
    BUDGETED_TOOL_NAMES,
    COUNTDOWN,
    PassBudget,
    make_budget_middleware,
)


class _Fn:
    def __init__(self, name):
        self.name = name


class FakeInvocationContext:
    """Duck-typed FunctionInvocationContext.

    Like the framework, ``tools`` is the run's live list - pass the SAME list
    to every context of one pass, since ``remove_tools`` mutates in place.
    """

    def __init__(self, tool_name, tools=("list_files", "read_file", "write_report")):
        self.function = _Fn(tool_name)
        self.tools = tools if isinstance(tools, list) else list(tools)
        self.result = "tool output"
        self.removed = []

    def remove_tools(self, tools):
        self.removed.append(list(tools))
        self.tools[:] = [t for t in self.tools if t not in set(tools)]


async def _call(mw, ctx):
    async def call_next():
        pass
    await mw(ctx, call_next)


def _mw(budget):
    return make_budget_middleware(budget, BUDGETED_TOOL_NAMES, "worker")


async def test_write_report_neither_counts_nor_is_stripped():
    budget = PassBudget(max_calls=2)
    mw = _mw(budget)
    await _call(mw, FakeInvocationContext("write_report"))
    assert budget.spent == 0
    assert "write_report" not in BUDGETED_TOOL_NAMES


async def test_no_countdown_while_there_is_plenty_of_runway():
    """max_calls=6: the first two calls leave 5 and 4 remaining - silent."""
    budget = PassBudget(max_calls=6)
    mw = _mw(budget)
    live = ["list_files", "read_file", "write_report"]
    seen = []
    for _ in range(2):
        ctx = FakeInvocationContext("read_file", live)
        await _call(mw, ctx)
        seen.append(ctx.result)
    assert budget.spent == 2
    for result in seen:
        assert result == "tool output"      # nothing appended at all
    assert live == ["list_files", "read_file", "write_report"]


async def test_countdown_fires_at_three_two_one_with_the_right_text():
    """max_calls=5, so the five calls leave 4, 3, 2, 1 then 0 remaining."""
    budget = PassBudget(max_calls=5)
    mw = _mw(budget)
    live = ["list_files", "read_file", "write_report"]
    seen = []
    for _ in range(5):
        ctx = FakeInvocationContext("read_file", live)
        await _call(mw, ctx)
        seen.append(ctx.result)
    assert seen[0] == "tool output"         # remaining 4 - silent
    assert COUNTDOWN[3] in seen[1]          # remaining 3
    assert COUNTDOWN[2] in seen[2]          # remaining 2
    assert COUNTDOWN[1] in seen[3]          # remaining 1
    assert BUDGET_SPENT in seen[4]          # remaining 0 -> closed
    assert live == ["write_report"]         # read tools stripped


async def test_budget_spent_strips_and_appends_the_closing_message():
    budget = PassBudget(max_calls=1)
    mw = _mw(budget)
    live = ["list_files", "read_file", "write_report"]
    ctx = FakeInvocationContext("read_file", live)
    await _call(mw, ctx)
    assert ctx.removed == [sorted(BUDGETED_TOOL_NAMES)]
    assert BUDGET_SPENT in ctx.result
    assert live == ["write_report"]
    # max_calls=1 leaves no reachable countdown line, which is correct.
    for line in COUNTDOWN.values():
        assert line not in ctx.result


async def test_finalizing_pass_strips_after_the_first_call():
    """remove_tools is only reachable from inside a tool call, so the final
    pass cannot be stripped pre-emptively - it strips on call one."""
    budget = PassBudget(max_calls=12)
    budget.start_pass(finalizing=True)
    mw = _mw(budget)
    live = ["list_files", "read_file", "write_report"]
    ctx = FakeInvocationContext("read_file", live)
    await _call(mw, ctx)
    assert ctx.removed == [sorted(BUDGETED_TOOL_NAMES)]
    assert BUDGET_SPENT in ctx.result
    assert live == ["write_report"]


async def test_strip_guard_suppresses_a_second_nudge_for_stragglers():
    budget = PassBudget(max_calls=1)
    mw = _mw(budget)
    live = ["list_files", "read_file", "write_report"]
    first = FakeInvocationContext("read_file", live)
    await _call(mw, first)
    straggler = FakeInvocationContext("list_files", live)   # same pass, in flight
    await _call(mw, straggler)
    assert budget.spent == 2                # still counted
    assert straggler.removed == []          # but not stripped again
    assert BUDGET_SPENT not in straggler.result


async def test_none_tools_still_gets_the_message():
    # context.tools is None outside a function-calling loop: nothing to
    # inspect or mutate, but the model must still be told why.
    budget = PassBudget(max_calls=1)
    mw = _mw(budget)
    ctx = FakeInvocationContext("read_file")
    ctx.tools = None
    await _call(mw, ctx)                    # must not raise
    assert BUDGET_SPENT in ctx.result
    assert ctx.removed == []


async def test_none_result_is_stringified_not_crashed():
    budget = PassBudget(max_calls=1)
    mw = _mw(budget)
    ctx = FakeInvocationContext("read_file")
    ctx.result = None
    await _call(mw, ctx)
    assert BUDGET_SPENT in ctx.result


def test_start_pass_resets_spent_and_sets_finalizing():
    budget = PassBudget(max_calls=4)
    budget.spent = 3
    budget.start_pass(finalizing=True)
    assert (budget.spent, budget.finalizing) == (0, True)
    budget.start_pass(finalizing=False)
    assert budget.finalizing is False


def test_remaining_clamps_at_zero():
    # spent can exceed max_calls: stragglers count, and the framework
    # re-arms tools per pass.
    budget = PassBudget(max_calls=2, spent=5)
    assert budget.remaining == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_reflection_budget.py -v`

Expected: FAIL — `ModuleNotFoundError: No module named 'reflection_demo.budget'`.

- [ ] **Step 3: Write the module**

Create `src/reflection_demo/budget.py`:

```python
"""Per-pass tool-call budget, countdown coaching, and the read-tool strip.

A deliberate copy of ``reflexion_demo/budget.py``'s structure - demo packages
here are standalone and must be readable end to end without tracing imports
into a sibling - with two differences forced by the loop:

* The budget is per PASS, not per turn. ``AgentLoopMiddleware`` runs every
  pass inside ONE ``agent.run()``, so there is no turn boundary at which a
  fresh counter could be minted; ``start_pass`` is called from
  ``next_message`` instead, the one hook that fires between passes.
* A finalizing pass strips after its FIRST call rather than at the budget,
  because ``remove_tools`` is only reachable from inside a tool call - tools
  cannot be taken away pre-emptively.

``COUNTDOWN`` and ``BUDGET_SPENT`` are byte-identical to
``reflexion_demo/budget.py``'s. Coaching only one of the two workers would
make evidence-gathering differ for a reason unrelated to the critic, which is
a confound in the very A/B these demos exist to run.
``tests/test_budget_wording_parity.py`` enforces the identity.
"""
from dataclasses import dataclass

from agent_framework import FunctionInvocationContext, function_middleware

# The judge holds no tools at all, so unlike reflexion's set this needs no
# read_report entry. write_report is absent on purpose: it is delivery, not
# exploration, so it neither counts nor is ever stripped.
BUDGETED_TOOL_NAMES = frozenset({"list_files", "read_file"})

# Anticipatory, not punitive. Three calls of runway is the point: at one call
# left a model can only pick a single action, so a warning there would be a
# notification rather than something it can plan around.
COUNTDOWN = {
    3: "3 tool calls left. Decide now which gaps matter most and spend them there.",
    2: "2 tool calls left.",
    1: "1 tool call left - your last. Spend it on the single most important gap, then write.",
}

# Positive framing: the worker is told it HAS what it needs, not that it has
# been cut off. The nudge rides on a tool result, so it lands in the model's
# context at the earliest moment it can act on.
BUDGET_SPENT = ("Exploration is closed. You have what you need - write the "
                "complete report now with write_report.")

_PREFIX = "[budget] "


@dataclass
class PassBudget:
    """Per-pass counter for one run; mutated in place at each pass boundary.

    Single-threaded by construction - ``AgentLoopMiddleware`` drives passes
    sequentially inside one ``agent.run()``.
    """

    max_calls: int
    spent: int = 0
    finalizing: bool = False

    @property
    def remaining(self) -> int:
        """Calls left this pass, clamped: ``spent`` can exceed ``max_calls``
        because in-flight stragglers still count."""
        return max(0, self.max_calls - self.spent)

    def start_pass(self, *, finalizing: bool) -> None:
        """Begin a pass. Called from ``next_message`` between passes."""
        self.spent = 0
        self.finalizing = finalizing


def make_budget_middleware(budget: PassBudget, budgeted, label: str):
    """Build the counting/coaching/stripping middleware for one run.

    Args:
        budget: The run's ``PassBudget``, reset at each pass boundary.
        budgeted: Names that count toward - and get stripped at - the budget.
            ``write_report`` must not be in it.
        label: Console tag ("worker") for the strip line.
    """
    @function_middleware
    async def budget_middleware(context: FunctionInvocationContext, call_next) -> None:
        # The call always executes first: the budget bounds exploration, it
        # never discards a result already paid for.
        await call_next()
        if context.function.name not in budgeted:
            return
        budget.spent += 1

        live = context.tools
        # The live tool list is the once-per-pass "already stripped" flag: if
        # the budgeted names are gone, this is an in-flight straggler from the
        # same batch and must not draw a second nudge. When tools is None
        # (called outside a function-calling loop) there is no list to inspect
        # or mutate, so skip the removal but still deliver the message.
        if live is not None and not any(getattr(t, "name", t) in budgeted for t in live):
            return

        if budget.finalizing or budget.remaining == 0:
            if live is not None:
                # Names not present are ignored by the framework, so passing
                # the whole budgeted set is safe.
                context.remove_tools(sorted(budgeted))
            context.result = f"{context.result or ''}\n\n{_PREFIX}{BUDGET_SPENT}"
            reason = "final pass" if budget.finalizing else f"{budget.max_calls} calls"
            print(f"  [{label}] exploration closed ({reason}) - read tools stripped")
            return

        line = COUNTDOWN.get(budget.remaining)
        if line:
            context.result = f"{context.result or ''}\n\n{_PREFIX}{line}"

    return budget_middleware
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_reflection_budget.py -v`

Expected: PASS, 10 tests.

- [ ] **Step 5: Run the full suite**

Run: `.venv\Scripts\python.exe -m pytest -q`

Expected: `210 passed, 3 deselected` (was 200/3; +10). Nothing regressed.

- [ ] **Step 6: Commit**

```bash
git add src/reflection_demo/budget.py tests/test_reflection_budget.py
git commit -m "feat(reflection): per-pass tool budget with countdown coaching"
```

---

### Task 2: port the countdown to `reflexion_demo`, and guard the parity

**Files:**

- Modify: `src/reflexion_demo/budget.py`
- Modify: `tests/test_reflexion_budget.py`
- Modify: `src/reflexion_demo/prompts/worker.md`
- Test: `tests/test_budget_wording_parity.py`

**Interfaces:**

- Consumes: `COUNTDOWN` / `BUDGET_SPENT` wording from Task 1 — copy the two
  literals verbatim.
- Produces: `reflexion_demo.budget.COUNTDOWN`, `.BUDGET_SPENT`, and
  `ToolBudget.remaining`. `BUDGET_NUDGE` is **removed**.

`reflexion_demo` keeps its own `ToolBudget` (per-turn instances make
`start_pass` unnecessary there) and keeps `read_report` in its budgeted set —
its reviewer has tools, which is the intended asymmetry.

- [ ] **Step 1: Write the failing parity test**

Create `tests/test_budget_wording_parity.py`:

```python
"""The two demos' budget wording must not drift.

Both packages are standalone by design, so the budget logic is duplicated
rather than shared. Drift is that duplication's only real failure mode: if one
worker is coached differently from the other, evidence-gathering differs for a
reason unrelated to the critic - a confound in the A/B the demos exist to run.

A test importing both packages does not violate the standalone rule; that
rule binds package code. Precedent: tests/test_reflection_main.py imports
reflexion_demo.main.DEFAULT_TOPIC to pin topic identity.
"""
from reflection_demo import budget as reflection_budget
from reflexion_demo import budget as reflexion_budget


def test_countdown_wording_identical_across_demos():
    assert reflection_budget.COUNTDOWN == reflexion_budget.COUNTDOWN


def test_budget_spent_wording_identical_across_demos():
    assert reflection_budget.BUDGET_SPENT == reflexion_budget.BUDGET_SPENT


def test_both_demos_exempt_write_report_from_the_budget():
    assert "write_report" not in reflection_budget.BUDGETED_TOOL_NAMES
    assert "write_report" not in reflexion_budget.BUDGETED_TOOL_NAMES
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_budget_wording_parity.py -v`

Expected: FAIL — `AttributeError: module 'reflexion_demo.budget' has no
attribute 'COUNTDOWN'`.

- [ ] **Step 3: Update `reflexion_demo/budget.py`**

Replace the `BUDGET_NUDGE` constant with the two shared literals, add
`remaining` to `ToolBudget`, and add the countdown branch. The file's
docstring, `BUDGETED_TOOL_NAMES` and the `>=`-vs-`==` reasoning in the
middleware all stay as they are.

Delete:

```python
BUDGET_NUDGE = (
    "[SYSTEM] Tool budget exhausted - you have been reasoning for a long "
    "time. Your exploration tools have been removed. Produce the report now "
    "from the information you already have."
)
```

Add in its place:

```python
# Byte-identical to reflection_demo/budget.py's. Both workers are coached the
# same so the A/B between the demos differs only in the critic; coaching one
# of them better would be a confound. Enforced by
# tests/test_budget_wording_parity.py.
COUNTDOWN = {
    3: "3 tool calls left. Decide now which gaps matter most and spend them there.",
    2: "2 tool calls left.",
    1: "1 tool call left - your last. Spend it on the single most important gap, then write.",
}

BUDGET_SPENT = ("Exploration is closed. You have what you need - write the "
                "complete report now with write_report.")

_PREFIX = "[budget] "
```

Add to `ToolBudget`, beside the existing `exhausted` property:

```python
    @property
    def remaining(self) -> int:
        """Calls left this turn, clamped - ``spent`` can pass ``max_calls``
        when a re-armed run pushes it past (see the middleware's note)."""
        return max(0, self.max_calls - self.spent)
```

In the middleware, after the existing strip block's `print(...)` and its
`return`, the countdown becomes the fall-through. The block that currently
reads:

```python
        if context.tools is not None:
            if not any(getattr(t, "name", t) in budgeted for t in context.tools):
                return
            context.remove_tools(sorted(budgeted))
        context.result = f"{context.result or ''}\n\n{BUDGET_NUDGE}"
        print(f"  [{label}] tool budget exhausted ({budget.max_calls} calls) - read tools stripped")
```

becomes:

```python
        if context.tools is not None:
            if not any(getattr(t, "name", t) in budgeted for t in context.tools):
                return
            context.remove_tools(sorted(budgeted))
        context.result = f"{context.result or ''}\n\n{_PREFIX}{BUDGET_SPENT}"
        print(f"  [{label}] exploration closed ({budget.max_calls} calls) - read tools stripped")
```

and immediately before the existing `if budget.spent < budget.max_calls: return`
early exit, replace that line with:

```python
        if budget.spent < budget.max_calls:
            # Anticipatory coaching: warn while there is still runway to
            # re-plan, rather than only after the tools are gone.
            line = COUNTDOWN.get(budget.remaining)
            if line:
                context.result = f"{context.result or ''}\n\n{_PREFIX}{line}"
            return
```

- [ ] **Step 4: Update `tests/test_reflexion_budget.py`**

Change the import to bring in the new names, and replace every
`BUDGET_NUDGE` reference:

```python
from reflexion_demo.budget import (
    BUDGET_SPENT,
    BUDGETED_TOOL_NAMES,
    COUNTDOWN,
    ToolBudget,
    make_budget_middleware,
)
```

Every existing `assert BUDGET_NUDGE in <x>.result` becomes
`assert BUDGET_SPENT in <x>.result`, and every
`assert BUDGET_NUDGE not in <x>.result` becomes
`assert BUDGET_SPENT not in <x>.result`. **Keep all existing structural
assertions** — counting, `write_report` exemption, strip-once-per-run,
re-armed second run, `None` tools, `None` result.

Then add one countdown case:

```python
async def test_countdown_warns_while_runway_remains():
    budget = ToolBudget(max_calls=5)
    mw = make_budget_middleware(budget, BUDGETED_TOOL_NAMES, "worker")
    run_tools = ["list_files", "read_file", "read_report"]
    seen = []
    for _ in range(5):
        ctx = FakeInvocationContext("read_file", run_tools)
        await _call(mw, ctx)
        seen.append(ctx.result)
    assert COUNTDOWN[3] not in seen[0]      # remaining 4 - silent
    assert COUNTDOWN[3] in seen[1]
    assert COUNTDOWN[2] in seen[2]
    assert COUNTDOWN[1] in seen[3]
    assert BUDGET_SPENT in seen[4]
```

- [ ] **Step 5: Update `src/reflexion_demo/prompts/worker.md`**

The worker is coached mid-run now, so say so rather than letting it be a
surprise. Add to the initial-mode branch (the `{% else %}` arm), after the
existing "Explore the corpus..." instruction:

```markdown
You have a limited number of tool calls. The last few are announced in the
tool results, and when the budget is spent your exploration tools close and
you write from what you have.
```

Keep the wording inside the existing paragraph structure so the markdown lint
gate stays green.

- [ ] **Step 6: Run the tests**

Run:

```bash
.venv\Scripts\python.exe -m pytest tests/test_budget_wording_parity.py tests/test_reflexion_budget.py tests/test_reflection_budget.py -v
.venv\Scripts\python.exe -m pytest -q
```

Expected: the three targeted files pass; full suite `214 passed, 3 deselected`
(210 + 3 parity + 1 countdown case). Deselected must stay 3.

- [ ] **Step 7: Verify the lint gate**

Run:

```bash
.venv\Scripts\python.exe -m pymarkdown --config .pymarkdown.json scan README.md CLAUDE.md src/hotl_demo/prompts src/reflexion_demo/prompts src/reflection_demo/prompts
```

Expected: exit 0.

- [ ] **Step 8: Commit**

```bash
git add src/reflexion_demo/budget.py src/reflexion_demo/prompts/worker.md \
        tests/test_reflexion_budget.py tests/test_budget_wording_parity.py
git commit -m "feat(reflexion): same countdown coaching, with a parity guard"
```

---

### Task 3: the finalize prompt

**Files:**

- Create: `src/reflection_demo/prompts/finalize.md`
- Modify: `src/reflection_demo/prompts/worker.md`
- Modify: `src/reflection_demo/prompting.py`
- Modify: `tests/test_reflection_prompts.py`

**Interfaces:**

- Consumes: nothing from Tasks 1–2.
- Produces: `render_finalize_message(*, topic: str, max_passes: int) -> str`
  in `reflection_demo.prompting`, keyword-only, `.strip()`ed.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_reflection_prompts.py`:

```python
from reflection_demo.prompting import render_finalize_message


def test_finalize_message_states_it_is_the_last_pass_and_demands_delivery():
    out = render_finalize_message(topic=TOPIC, max_passes=3)
    assert TOPIC in out
    assert "write_report" in out
    assert "3" in out
    lowered = out.lower()
    assert "final" in lowered
    assert "no further review" in lowered


def test_finalize_message_fully_renders():
    out = render_finalize_message(topic=TOPIC, max_passes=2)
    assert "{{" not in out and "{%" not in out


def test_worker_prompt_states_the_tool_budget():
    out = render_worker_prompt(topic=TOPIC, max_passes=3, max_tool_calls=12)
    assert "12" in out
    # The old "There is no tool budget here" claim must be gone.
    assert "no tool budget" not in out.lower()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_reflection_prompts.py -v`

Expected: FAIL — `ImportError: cannot import name 'render_finalize_message'`.

- [ ] **Step 3: Write `finalize.md`**

Create `src/reflection_demo/prompts/finalize.md`:

```markdown
This is your final pass. There will be no further review, and no more
feedback is coming.

## Topic

{{ topic }}

Write the COMPLETE report now from what you already know, and save it by
calling write_report with the full text. Do not start new exploration: if you
call a read tool on this pass, your exploration tools close immediately
afterwards.

This is pass {{ max_passes }} of {{ max_passes }} - whatever write_report
holds when this pass ends is what ships.
```

- [ ] **Step 4: Update `worker.md`**

Replace the paragraph that currently reads:

```markdown
Explore economically: read what you need and no more. There is no tool
budget here, but a bloated transcript crowds out the report.
```

with:

```markdown
You have {{ max_tool_calls }} tool calls per pass. The last few are announced
in the tool results, and when the budget is spent your exploration tools
close and you write from what you have. Spend them on the gaps that matter.
```

- [ ] **Step 5: Update `prompting.py`**

Add `max_tool_calls` to the worker renderer, add the finalize renderer, and
correct the module docstring's now-half-false claim:

```python
"""Prompt rendering: Jinja2 templates in ``prompts/``, one per participant.

There is no revision variant - the loop middleware injects the judge's
feedback as the next pass's input, so the worker prompt is rendered once per
run. There IS a finalize message: the final pass gets a different instruction
(write now, no further review), delivered through ``next_message`` because the
worker prompt cannot be re-rendered mid-run.
"""
```

```python
def render_worker_prompt(*, topic: str, max_passes: int, max_tool_calls: int) -> str:
    """Render the worker's opening prompt for the whole run.

    Args:
        topic: The migration topic under assessment.
        max_passes: The loop cap, for the model's situational awareness.
        max_tool_calls: The per-pass tool budget, so the countdown lines that
            arrive later are expected rather than a surprise.
    """
    return _ENV.get_template("worker.md").render(
        topic=topic, max_passes=max_passes, max_tool_calls=max_tool_calls).strip()


def render_finalize_message(*, topic: str, max_passes: int) -> str:
    """Render the final pass's instruction.

    Delivered via ``next_message``, not as a re-render of the worker prompt:
    ``AgentLoopMiddleware`` owns the agent for the whole run, so the system
    prompt is fixed once and only the per-pass input can change.
    """
    return _ENV.get_template("finalize.md").render(
        topic=topic, max_passes=max_passes).strip()
```

- [ ] **Step 6: Fix the existing worker-prompt tests**

`render_worker_prompt` now requires `max_tool_calls`. Every existing call in
`tests/test_reflection_prompts.py` must pass it — add `max_tool_calls=12` to
each.

- [ ] **Step 7: Run the tests**

Run:

```bash
.venv\Scripts\python.exe -m pytest tests/test_reflection_prompts.py tests/test_markdown_lint.py -v
```

Expected: PASS. If lint fails, fix the markdown in the templates — do not
loosen `.pymarkdown.json`.

- [ ] **Step 8: Commit**

```bash
git add src/reflection_demo/prompting.py src/reflection_demo/prompts \
        tests/test_reflection_prompts.py
git commit -m "feat(reflection): finalize prompt and a worker brief that states the budget"
```

---

### Task 4: wire the budget into the loop

**Files:**

- Modify: `src/reflection_demo/judging.py` (`make_next_message` only)
- Modify: `src/reflection_demo/main.py`
- Modify: `tests/test_reflection_loop.py`
- Modify: `tests/test_e2e_reflection.py`

**Interfaces:**

- Consumes: `PassBudget`, `make_budget_middleware`, `BUDGETED_TOOL_NAMES`
  (Task 1); `render_finalize_message`, `render_worker_prompt(..., max_tool_calls=)`
  (Task 3).
- Produces: `make_next_message(budget, max_passes: int, finalize_message: str)`;
  `build_agent(corpus_root, report_path, judge_instructions, log, max_passes,
  max_tool_calls) -> (agent, flag)`.

`make_next_message` gains the pass-boundary duties because it is the only hook
that fires exactly once between passes. It takes `budget` **duck-typed** — it
calls `start_pass` and nothing else — so `judging.py` needs no import from
`budget.py` and the two stay decoupled.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_reflection_loop.py`:

```python
from reflection_demo.judging import make_next_message


class FakeBudget:
    """Records start_pass calls; next_message only ever calls that."""

    def __init__(self):
        self.passes = []

    def start_pass(self, *, finalizing):
        self.passes.append(finalizing)


def test_next_message_relays_feedback_and_starts_a_normal_pass():
    budget = FakeBudget()
    nxt = make_next_message(budget, max_passes=3, finalize_message="FINALIZE-TEXT")
    out = nxt(iteration=1, feedback="cite the Azure mandate")
    assert "cite the Azure mandate" in out
    assert "write_report" in out
    assert "FINALIZE-TEXT" not in out
    assert budget.passes == [False]


def test_next_message_delivers_finalize_on_the_boundary_into_the_last_pass():
    """max_passes=3: after pass 2 the next pass is the last one."""
    budget = FakeBudget()
    nxt = make_next_message(budget, max_passes=3, finalize_message="FINALIZE-TEXT")
    out = nxt(iteration=2, feedback="still thin on residency")
    assert out == "FINALIZE-TEXT"
    assert budget.passes == [True]


def test_next_message_without_feedback_still_asks_for_a_save():
    budget = FakeBudget()
    nxt = make_next_message(budget, max_passes=5, finalize_message="F")
    assert "write_report" in nxt(iteration=1, feedback=None)


def test_next_message_finalizes_on_the_boundary_even_with_max_passes_two():
    budget = FakeBudget()
    nxt = make_next_message(budget, max_passes=2, finalize_message="FINALIZE-TEXT")
    assert nxt(iteration=1, feedback="more") == "FINALIZE-TEXT"
    assert budget.passes == [True]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_reflection_loop.py -v`

Expected: FAIL — existing `make_next_message()` takes no arguments, so the new
tests raise `TypeError`.

- [ ] **Step 3: Rewrite `make_next_message` in `judging.py`**

Replace the whole existing `make_next_message` with:

```python
def make_next_message(budget, max_passes: int, finalize_message: str):
    """Build the ``next_message`` callable: feedback relay plus pass boundary.

    Two jobs, in one place because ``AgentLoopMiddleware`` calls this exactly
    once between passes and nothing else marks that boundary:

    1. **The verbal-feedback channel.** The middleware's default next-message
       is a bare "continue" nudge that would drop the judge's reasoning on the
       floor; without this relay the judge could reject forever without ever
       saying why.
    2. **Starting the next pass.** The tool budget is per pass, but every pass
       runs inside one ``agent.run()``, so there is no turn boundary at which
       a fresh counter could be minted. ``budget`` is duck-typed - only
       ``start_pass`` is ever called - so this module needs no import from
       ``budget.py``.

    On the boundary into the LAST pass the judge's feedback is replaced by the
    finalize instruction and the pass is marked finalizing, which makes the
    middleware close exploration after that pass's first tool call.
    ``iteration`` counts completed passes, so ``iteration == max_passes - 1``
    is the boundary into pass ``max_passes``.

    Args:
        budget: Anything with ``start_pass(*, finalizing: bool)``.
        max_passes: The loop cap, to recognise the final boundary.
        finalize_message: Pre-rendered final-pass instruction (see
            ``prompting.render_finalize_message``).
    """
    def next_message(*, iteration, feedback=None, **kwargs) -> str:
        finalizing = iteration == max_passes - 1
        budget.start_pass(finalizing=finalizing)
        if finalizing:
            return finalize_message
        if feedback:
            return ("A reviewer judged your previous report incomplete.\n\n"
                    f"Reviewer feedback: {feedback}\n\n"
                    "Revise the report to address it and save the COMPLETE "
                    "revised report with write_report.")
        return ("Keep improving the report and save the complete text with "
                "write_report.")

    return next_message
```

- [ ] **Step 4: Wire `main.py`**

Add the import:

```python
from .budget import BUDGETED_TOOL_NAMES, PassBudget, make_budget_middleware
from .prompting import render_finalize_message, render_judge_instructions, render_worker_prompt
```

`build_agent` gains **two** parameters. It needs `topic` because the finalize
message is rendered from the topic and `build_agent` currently only receives
the already-rendered judge instructions. Final signature:

```python
def build_agent(corpus_root: Path, report_path: Path, topic: str,
                judge_instructions: str, log: RunLog, max_passes: int,
                max_tool_calls: int):
```

Inside, before constructing the loop:

```python
    # One budget for the run; next_message resets it at each pass boundary.
    # Pass 1 has no boundary before it, so it starts non-finalizing here -
    # unless the run is a single pass, in which case that pass IS the last and
    # must close exploration after its first call, exactly as a capped final
    # pass would.
    budget = PassBudget(max_calls=max_tool_calls)
    budget.start_pass(finalizing=max_passes == 1)
```

Then add the middleware to the agent and pass the new args to
`make_next_message`:

```python
    loop = AgentLoopMiddleware(
        make_judge_predicate(OllamaChatClient(), judge_instructions, log,
                             resolve_num_ctx(), report_path),
        max_iterations=max_passes,
        next_message=make_next_message(
            budget, max_passes,
            render_finalize_message(topic=topic, max_passes=max_passes)),
    )
    agent = Agent(
        client=OllamaChatClient(),
        name="worker",
        instructions=_WORKER_INSTRUCTIONS,
        tools=make_corpus_tools(corpus_root) + [write_report],
        middleware=[make_budget_middleware(budget, BUDGETED_TOOL_NAMES, "worker"), loop],
        default_options={"num_ctx": resolve_num_ctx()},
    )
```

Middleware order matters and is deliberate: the budget middleware is a
`function_middleware` (it wraps tool calls) and the loop is an
`AgentMiddleware` (it wraps runs), so they compose without interfering. List
the budget first so the strip line prints before any loop narration.

Add the CLI flag next to `--max-passes`:

```python
    parser.add_argument("--max-tool-calls", type=int, default=12, metavar="N",
                        help="per-pass read-tool budget; the last 3 calls are "
                             "announced and exploration closes when it is spent "
                             "(default: %(default)s)")
```

```python
    if args.max_tool_calls < 1:
        parser.error("--max-tool-calls must be >= 1")
```

Update the `build_agent` call and the console banner in `_amain`:

```python
    agent, flag = build_agent(
        corpus_root, report_path, args.topic,
        render_judge_instructions(topic=args.topic), log,
        args.max_passes, args.max_tool_calls)

    print(f"Topic: {args.topic}")
    print(f"Budget: {args.max_passes} passes, {args.max_tool_calls} tool calls "
          f"per pass (the judge holds NO tools)")
```

and the worker prompt call:

```python
    result = await agent.run(
        render_worker_prompt(topic=args.topic, max_passes=args.max_passes,
                            max_tool_calls=args.max_tool_calls),
        session=session)
```

- [ ] **Step 5: Update the E2E test's call site**

`tests/test_e2e_reflection.py` calls `build_agent` and `render_worker_prompt`.
Both signatures changed. Update to:

```python
    agent, flag = build_agent(
        corpus, report_path, DEFAULT_TOPIC,
        render_judge_instructions(topic=DEFAULT_TOPIC), log,
        max_passes, max_tool_calls=6)
    result = await agent.run(
        render_worker_prompt(topic=DEFAULT_TOPIC, max_passes=max_passes,
                            max_tool_calls=6),
        session=session)
```

Keep everything else in that test as it is, including its `("unjudged", 1)`
assertion — with `max_passes=1` the single pass is still never judged.

- [ ] **Step 6: Update the integration test's call site if needed**

`tests/test_reflection_integration.py` constructs the loop. If it calls
`make_next_message()` or `build_agent(...)`, update those calls for the new
signatures. Its assertions about the relay reaching pass 2 must keep passing —
with `max_passes` of 2 the boundary after pass 1 is now the *finalize*
boundary, so if that test asserts the judge's reasoning reaches pass 2 it will
need `max_passes=3` to keep testing the relay rather than the finalize path.
Make that change if so, and say in your report which you did.

- [ ] **Step 7: Run the tests**

Run:

```bash
.venv\Scripts\python.exe -m pytest tests/test_reflection_loop.py tests/test_reflection_integration.py tests/test_reflection_main.py -v
.venv\Scripts\python.exe -m reflection_demo.main --help
.venv\Scripts\python.exe -m pytest -q
```

Expected: targeted tests pass; `--help` lists `--max-tool-calls` and makes no
network call; full suite green with the deselected count still 3.

- [ ] **Step 8: Commit**

```bash
git add src/reflection_demo/judging.py src/reflection_demo/main.py \
        tests/test_reflection_loop.py tests/test_reflection_integration.py \
        tests/test_e2e_reflection.py
git commit -m "feat(reflection): wire the per-pass budget and finalize pass into the loop"
```

---

### Task 5: documentation

**Files:**

- Modify: `README.md`
- Modify: `CLAUDE.md`
- Modify: `docs/superpowers/specs/2026-07-29-reflection-demo-design.md`
- Modify: `docs/superpowers/specs/2026-07-17-reflexion-demo-design.md`

**Interfaces:**

- Consumes: the finished CLI from Task 4.
- Produces: nothing.

- [ ] **Step 1: Update the README's reflection section**

Three edits:

1. Add the budget to the demo's description and to the `Running the A/B`
   commands: `poetry run reflection --max-tool-calls 12`.
2. **Remove the A/B caveat that the reflection worker is unbounded.** Both
   workers are now identically bounded and identically coached, so that entry
   is obsolete. The delivery-retry caveat stays.
3. Add a short paragraph on the countdown: the last three calls are announced
   with escalating urgency, the closing message tells the worker it has what
   it needs, and the final pass gets a finalize instruction instead of judge
   feedback — with the note that a worker which ignores it gets exactly one
   exploratory call before its tools close, because `remove_tools()` is only
   reachable from inside a tool call.

- [ ] **Step 2: Update the README's reflexion section**

Note that the countdown wording is shared with the reflection demo and
enforced identical by a test, and that its worker is coached mid-run rather
than only nudged after the fact.

- [ ] **Step 3: Update `CLAUDE.md`**

Add to the Commands block, aligned with the surrounding comment column:

```bash
poetry run reflection --max-tool-calls 4        # quickest tour of the countdown
```

Add to the reflection bullet in the architecture section: the budget is per
**pass** and reset from `next_message`, because `AgentLoopMiddleware` runs
every pass inside one `agent.run()` and there is no turn boundary; and
`remove_tools()` is only reachable from inside a tool call, so the final pass
strips after its first call rather than pre-emptively.

- [ ] **Step 4: Update both design specs**

- `2026-07-29-reflection-demo-design.md` §6 says the tool budget is out of
  scope with the reasoning that repeating it adds no insight. That is now
  reversed — replace the paragraph with a pointer to
  `2026-07-30-reflection-tool-budget-design.md` and the two reasons
  (it removes a confound; it fixes the observed `write_report never called`
  run). Also correct §1/§4 wherever they claim the workers differ only in
  tools, since the budget difference is now gone rather than merely
  acknowledged.
- `2026-07-17-reflexion-demo-design.md` §5 describes the single exhaustion
  nudge. Add the countdown, and note the wording is shared with the reflection
  demo and pinned by `tests/test_budget_wording_parity.py`.

- [ ] **Step 5: Verify**

Run:

```bash
.venv\Scripts\python.exe -m pymarkdown --config .pymarkdown.json scan README.md CLAUDE.md src/hotl_demo/prompts src/reflexion_demo/prompts src/reflection_demo/prompts
.venv\Scripts\python.exe -m pytest -q
```

Expected: lint exit 0; suite green.

- [ ] **Step 6: Commit**

```bash
git add README.md CLAUDE.md docs/superpowers/specs/2026-07-29-reflection-demo-design.md \
        docs/superpowers/specs/2026-07-17-reflexion-demo-design.md
git commit -m "docs: countdown coaching and the reflection tool budget"
```

---

## Self-Review

**Spec coverage:**

| Spec section | Task |
|---|---|
| §1 purpose, why the reversal, why both demos | Task 5 (docs), Global Constraints |
| §2 countdown constants and `[budget] ` prefix | Task 1 |
| §3 budget lifetime, `PassBudget`, `next_message` as reset hook | Tasks 1 and 4 |
| §4 finalize pass, `iteration == max_passes - 1` | Tasks 3 and 4 |
| §5 middleware rules 1-6, strip guard, `None` tools | Task 1 (all covered by tests) |
| §5 budgeted-name table | Task 1 (`BUDGETED_TOOL_NAMES`), Task 2 (reflexion keeps `read_report`) |
| §6 reflexion changes incl. its tests and worker prompt | Task 2 |
| §7 prompts: `finalize.md`, `worker.md`, `prompting.py` docstring | Task 3 |
| §8 CLI `--max-tool-calls`, `>= 1`, the `1` edge case | Task 4 (flag), Task 1 (edge-case test) |
| §9 all test cases incl. the parity guard | Tasks 1, 2, 4 |
| §10 removing the unbounded-worker A/B caveat | Task 5 Step 1 |
| §11 out of scope | nothing implements them — correct |

**Placeholder scan:** none. Every code step carries runnable code. Task 4
Step 6 is conditional on what the integration test currently calls, which is
why it instructs the implementer to state what it found — not a placeholder,
a genuine branch the implementer must resolve and report.

**Type consistency:** `PassBudget.start_pass(*, finalizing: bool)` is defined
in Task 1 and called in Task 4 (production) and by `FakeBudget` in Task 4's
tests with the same keyword-only signature. `make_next_message(budget,
max_passes, finalize_message)` is defined in Task 4 Step 3 and exercised with
exactly those three positional args in Step 1's tests.
`render_finalize_message(*, topic, max_passes)` is defined in Task 3 and called
that way in Task 4. `build_agent` gains **two** parameters (`topic` and
`max_tool_calls`) — Task 4 Step 4 states the final signature explicitly and
Steps 5-6 update every call site. `render_worker_prompt` gains
`max_tool_calls` in Task 3 and every call site is updated in Tasks 3 and 4.
