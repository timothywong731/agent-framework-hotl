"""Per-turn tool-call budget and the mid-turn read-tool strip.

One function middleware owns the counter. Read tools count; ``write_report``
is exempt (delivery, not exploration). Once the budget is spent the
middleware executes the call normally, strips the read tools for the
remainder of the run via the framework's live-mutation point
(``FunctionInvocationContext.remove_tools``), and appends a nudge to that
call's result so the model knows why its tools vanished. A turn can span
several ``agent.run`` calls and the framework rebuilds the live tool list
per run, so the strip re-fires on every run that re-arms a budgeted tool.
"""
from dataclasses import dataclass

from agent_framework import FunctionInvocationContext, function_middleware

BUDGETED_TOOL_NAMES = frozenset({"list_files", "read_file", "read_report"})

BUDGET_NUDGE = (
    "[SYSTEM] Tool budget exhausted - you have been reasoning for a long "
    "time. Your exploration tools have been removed. Produce the report now "
    "from the information you already have."
)


@dataclass
class ToolBudget:
    """Mutable per-turn counter; a fresh instance is made for every agent turn."""

    max_calls: int
    spent: int = 0

    @property
    def exhausted(self) -> bool:
        return self.spent >= self.max_calls


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
        context.result = f"{context.result or ''}\n\n{BUDGET_NUDGE}"
        print(f"  [{label}] tool budget exhausted ({budget.max_calls} calls) - read tools stripped")

    return budget_middleware
