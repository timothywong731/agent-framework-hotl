"""Per-turn tool-call budget, countdown coaching, and the mid-turn read-tool
strip.

One function middleware owns the counter. Read tools count; ``write_report``
is exempt (delivery, not exploration). While calls remain, each budgeted call
past the ``COUNTDOWN`` thresholds appends an anticipatory warning to that
call's result, so the model can re-plan while there is still runway. Once the
budget is spent the middleware executes the call normally, strips the read
tools for the remainder of the run via the framework's live-mutation point
(``FunctionInvocationContext.remove_tools``), and appends ``BUDGET_SPENT`` to
that call's result so the model knows why its tools vanished. A turn can span
several ``agent.run`` calls and the framework rebuilds the live tool list
per run, so the strip re-fires on every run that re-arms a budgeted tool.

``COUNTDOWN`` and ``BUDGET_SPENT`` are byte-identical to
``reflection_demo/budget.py``'s - coaching only one of the two workers would
make evidence-gathering differ for a reason unrelated to the critic, a
confound in the A/B these demos exist to run.
``tests/test_budget_wording_parity.py`` enforces the identity.
"""
from dataclasses import dataclass

from agent_framework import Content, FunctionInvocationContext, function_middleware

BUDGETED_TOOL_NAMES = frozenset({"list_files", "read_file", "read_report"})

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


def _append_note(result, note: str):
    """Append a budget note to a tool result without destroying it.

    By the time function middleware sees ``context.result``, the framework
    has already parsed the tool's raw return value into ``list[Content]``
    (``FunctionTool.parse_result`` - every result, text or otherwise, is
    normalized to that shape). ``f"{result}"`` over that list stringifies
    each ``Content`` to its Python repr
    (``[<agent_framework._types.Content object at 0x...>]``), silently
    replacing the real tool output with an unusable placeholder - verified
    against a live run where the model could no longer read the corpus at
    all. So the note is attached to a text ``Content`` in place (or appended
    as a new one) instead of flattening the whole list to a string.

    A duck-typed fake or a tool wired with ``result_parser=SKIP_PARSING``
    can still hand us a plain ``str`` (or ``None``); that path keeps working
    exactly as before.
    """
    if isinstance(result, list):
        for item in reversed(result):
            if getattr(item, "type", None) == "text":
                item.text = f"{item.text}\n\n{_PREFIX}{note}"
                return result
        # No text Content in the list (e.g. only an image) - append one
        # rather than dropping the note.
        return [*result, Content.from_text(f"{_PREFIX}{note}")]
    return f"{result or ''}\n\n{_PREFIX}{note}"


@dataclass
class ToolBudget:
    """Mutable per-turn counter; a fresh instance is made for every agent turn."""

    max_calls: int
    spent: int = 0

    @property
    def exhausted(self) -> bool:
        return self.spent >= self.max_calls

    @property
    def remaining(self) -> int:
        """Calls left this turn, clamped - ``spent`` can pass ``max_calls``
        when a re-armed run pushes it past (see the middleware's note)."""
        return max(0, self.max_calls - self.spent)


def make_budget_middleware(budget: ToolBudget, budgeted, label: str):
    """Build the counting/stripping middleware for one agent turn.

    Args:
        budget: This turn's fresh counter (never share across turns).
        budgeted: Names of tools that count toward - and get stripped at -
            the budget. ``write_report`` must not be in it.
        label: Console tag ("worker"/"reviewer") for the strip line.
    """
    @function_middleware
    async def budget_middleware(context: FunctionInvocationContext, call_next) -> None:
        await call_next()
        if context.function.name not in budgeted:
            return
        budget.spent += 1
        if budget.spent < budget.max_calls:
            # Anticipatory coaching: warn while there is still runway to
            # re-plan, rather than only after the tools are gone.
            line = COUNTDOWN.get(budget.remaining)
            if line:
                context.result = _append_note(context.result, line)
            return
        # >= not ==: both executors make two run() calls on one budget, and
        # the framework rebuilds the live tool list per run, so a strip does
        # not persist and the re-armed run can push ``spent`` past max. The
        # live list is the once-per-run "stripped" flag: budgeted names
        # present -> strip + nudge; absent -> in-flight batch stragglers
        # from an already-stripped run pass through silently.
        if context.tools is not None:
            if not any(getattr(t, "name", t) in budgeted for t in context.tools):
                return
            # Names not present are ignored by the framework, so passing
            # the whole budgeted set is safe for both agents.
            context.remove_tools(sorted(budgeted))
        context.result = _append_note(context.result, BUDGET_SPENT)
        print(f"  [{label}] exploration closed ({budget.max_calls} calls) - read tools stripped")

    return budget_middleware
