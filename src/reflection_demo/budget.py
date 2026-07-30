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
