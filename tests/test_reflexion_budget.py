"""Tool-call budget: counting, exemption, and mid-turn read-tool stripping."""
import pytest

from reflexion_demo.budget import (
    BUDGET_NUDGE,
    BUDGETED_TOOL_NAMES,
    ToolBudget,
    make_budget_middleware,
)


class _Fn:
    def __init__(self, name):
        self.name = name


class FakeInvocationContext:
    """Duck-typed FunctionInvocationContext: the middleware reads
    .function.name, .tools, .result and calls .remove_tools(names)."""

    def __init__(self, tool_name, tools=("list_files", "read_file", "write_report")):
        self.function = _Fn(tool_name)
        self.tools = list(tools)
        self.result = "tool output"
        self.removed = []

    def remove_tools(self, tools):
        self.removed.append(list(tools))


async def _call(mw, ctx):
    async def call_next():
        pass
    await mw(ctx, call_next)


async def test_read_tool_calls_count_and_write_report_is_exempt():
    budget = ToolBudget(max_calls=2)
    mw = make_budget_middleware(budget, BUDGETED_TOOL_NAMES, "worker")
    await _call(mw, FakeInvocationContext("read_file"))
    await _call(mw, FakeInvocationContext("write_report"))
    assert budget.spent == 1
    assert not budget.exhausted


async def test_strip_fires_exactly_once_at_the_budget_with_nudge():
    budget = ToolBudget(max_calls=2)
    mw = make_budget_middleware(budget, BUDGETED_TOOL_NAMES, "worker")
    first = FakeInvocationContext("read_file")
    await _call(mw, first)
    assert first.removed == []          # under budget: untouched
    assert first.result == "tool output"

    second = FakeInvocationContext("list_files")
    await _call(mw, second)
    assert budget.exhausted
    assert second.removed == [sorted(BUDGETED_TOOL_NAMES)]   # strip fired
    assert second.result.startswith("tool output")
    assert BUDGET_NUDGE in second.result

    third = FakeInvocationContext("read_file")   # in-flight batch straggler
    await _call(mw, third)
    assert budget.spent == 3
    assert third.removed == []          # strip + nudge happen only once
    assert BUDGET_NUDGE not in third.result


async def test_strip_survives_a_none_tools_list():
    # context.tools is None when invoked outside a function-calling loop.
    budget = ToolBudget(max_calls=1)
    mw = make_budget_middleware(budget, BUDGETED_TOOL_NAMES, "worker")
    ctx = FakeInvocationContext("read_file")
    ctx.tools = None
    await _call(mw, ctx)                # must not raise
    assert budget.exhausted
    assert BUDGET_NUDGE in ctx.result   # nudge still delivered


async def test_none_result_is_stringified_not_crashed():
    budget = ToolBudget(max_calls=1)
    mw = make_budget_middleware(budget, BUDGETED_TOOL_NAMES, "worker")
    ctx = FakeInvocationContext("read_file")
    ctx.result = None
    await _call(mw, ctx)
    assert BUDGET_NUDGE in ctx.result
