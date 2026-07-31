"""Tool-call budget: counting, exemption, and mid-turn read-tool stripping."""
import pytest

from agent_framework import Content

from reflexion_demo.budget import (
    BUDGET_SPENT,
    BUDGETED_TOOL_NAMES,
    COUNTDOWN,
    ToolBudget,
    make_budget_middleware,
)


class _Fn:
    def __init__(self, name):
        self.name = name


class FakeInvocationContext:
    """Duck-typed FunctionInvocationContext: the middleware reads
    .function.name, .tools, .result and calls .remove_tools(names).

    Like the framework, ``tools`` is the run's live list - pass the SAME
    list to every context of one run (mutations are shared); each new run
    rebuilds it. ``remove_tools`` mutates in place, as the real one does.
    """

    def __init__(self, tool_name, tools=("list_files", "read_file", "write_report"), result="tool output"):
        self.function = _Fn(tool_name)
        self.tools = tools if isinstance(tools, list) else list(tools)
        self.result = result
        self.removed = []

    def remove_tools(self, tools):
        self.removed.append(list(tools))
        self.tools[:] = [t for t in self.tools if t not in set(tools)]


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
    run_tools = ["list_files", "read_file", "write_report"]   # one live list per run
    first = FakeInvocationContext("read_file", run_tools)
    await _call(mw, first)
    assert first.removed == []          # under budget: untouched
    assert COUNTDOWN[1] in first.result  # max_calls=2 -> 1 remaining already

    second = FakeInvocationContext("list_files", run_tools)
    await _call(mw, second)
    assert budget.exhausted
    assert second.removed == [sorted(BUDGETED_TOOL_NAMES)]   # strip fired
    assert run_tools == ["write_report"]                     # live list mutated
    assert second.result.startswith("tool output")
    assert BUDGET_SPENT in second.result

    third = FakeInvocationContext("read_file", run_tools)   # in-flight batch straggler
    await _call(mw, third)
    assert budget.spent == 3
    assert third.removed == []          # strip + nudge happen once per run
    assert BUDGET_SPENT not in third.result


async def test_rearmed_second_run_is_stripped_again_past_the_budget():
    # Both executors call agent.run twice on one budget, and the framework
    # rebuilds the live tool list per run: the second run re-arms the read
    # tools with ``spent`` already at/past max, so '==' would never fire.
    budget = ToolBudget(max_calls=1)
    mw = make_budget_middleware(budget, BUDGETED_TOOL_NAMES, "worker")
    run1 = FakeInvocationContext("read_file", ["list_files", "read_file", "read_report"])
    await _call(mw, run1)
    assert run1.removed == [sorted(BUDGETED_TOOL_NAMES)]

    run2_tools = ["list_files", "read_file", "read_report"]  # rebuilt, re-armed
    run2 = FakeInvocationContext("read_file", run2_tools)
    await _call(mw, run2)
    assert budget.spent == 2                                 # past max
    assert run2.removed == [sorted(BUDGETED_TOOL_NAMES)]     # strip re-fired
    assert run2_tools == []
    assert BUDGET_SPENT in run2.result

    straggler = FakeInvocationContext("list_files", run2_tools)  # same run 2
    await _call(mw, straggler)
    assert straggler.removed == []                           # still once per run
    assert BUDGET_SPENT not in straggler.result


async def test_strip_survives_a_none_tools_list():
    # context.tools is None when invoked outside a function-calling loop.
    budget = ToolBudget(max_calls=1)
    mw = make_budget_middleware(budget, BUDGETED_TOOL_NAMES, "worker")
    ctx = FakeInvocationContext("read_file")
    ctx.tools = None
    await _call(mw, ctx)                # must not raise
    assert budget.exhausted
    assert BUDGET_SPENT in ctx.result   # nudge still delivered


async def test_none_result_is_stringified_not_crashed():
    budget = ToolBudget(max_calls=1)
    mw = make_budget_middleware(budget, BUDGETED_TOOL_NAMES, "worker")
    ctx = FakeInvocationContext("read_file")
    ctx.result = None
    await _call(mw, ctx)
    assert BUDGET_SPENT in ctx.result


async def test_content_list_result_survives_the_note():
    """Regression: real ``context.result`` is ``list[Content]``, not a str -
    the framework parses every tool return value into that shape before
    function middleware runs (see ``FunctionTool.parse_result``). An
    f-string over the list stringifies each ``Content`` to its Python repr
    and destroys the real tool output; this must not happen."""
    budget = ToolBudget(max_calls=1)
    mw = make_budget_middleware(budget, BUDGETED_TOOL_NAMES, "worker")
    ctx = FakeInvocationContext("read_file", result=[Content.from_text("REAL-FILE-CONTENT-12345")])
    await _call(mw, ctx)
    assert isinstance(ctx.result, list)
    assert len(ctx.result) == 1
    assert "REAL-FILE-CONTENT-12345" in ctx.result[0].text   # original output preserved
    assert BUDGET_SPENT in ctx.result[0].text                # note still delivered


async def test_countdown_note_also_survives_a_content_list_result():
    """The countdown branch appends through the same ``_append_note``.

    The regression above only covers the closing branch (``max_calls=1``), yet
    the countdown is where the repr-mangling bug did the most damage: it fires
    up to three times a turn against results the worker still has calls left to
    act on, where the closing branch mangles one result per turn.
    """
    budget = ToolBudget(max_calls=2)
    mw = make_budget_middleware(budget, BUDGETED_TOOL_NAMES, "worker")
    run_tools = ["list_files", "read_file", "write_report"]
    ctx = FakeInvocationContext("read_file", run_tools,
                                result=[Content.from_text("REAL-FILE-CONTENT-12345")])
    await _call(mw, ctx)
    assert ctx.removed == [] and run_tools == ["list_files", "read_file", "write_report"]
    assert len(ctx.result) == 1
    assert "REAL-FILE-CONTENT-12345" in ctx.result[0].text   # original output preserved
    assert COUNTDOWN[1] in ctx.result[0].text                # countdown still delivered


async def test_content_list_with_no_text_content_gets_a_new_text_item():
    """A tool result of only non-text Content (e.g. an image) has nothing to
    append the note to in place - a new text Content carries it instead of
    the note being silently dropped."""
    budget = ToolBudget(max_calls=1)
    mw = make_budget_middleware(budget, BUDGETED_TOOL_NAMES, "worker")
    image = Content.from_data(data=b"\x89PNG", media_type="image/png")
    ctx = FakeInvocationContext("read_file", result=[image])
    await _call(mw, ctx)
    assert len(ctx.result) == 2
    assert ctx.result[0] is image                            # original item untouched
    assert ctx.result[1].type == "text"
    assert BUDGET_SPENT in ctx.result[1].text


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
