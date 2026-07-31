"""Per-pass tool budget: counting, countdown coaching, strip, finalize."""
from agent_framework import Content

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


async def test_content_list_result_survives_the_note():
    """Regression: real ``context.result`` is ``list[Content]``, not a str -
    the framework parses every tool return value into that shape before
    function middleware runs (see ``FunctionTool.parse_result``). An
    f-string over the list stringifies each ``Content`` to its Python repr
    and destroys the real tool output; this must not happen."""
    budget = PassBudget(max_calls=1)
    mw = _mw(budget)
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
    up to three times a pass against results the worker still has calls left to
    act on, where the closing branch mangles one result per pass.
    """
    budget = PassBudget(max_calls=2)
    mw = _mw(budget)
    live = ["list_files", "read_file", "write_report"]
    ctx = FakeInvocationContext("read_file", live,
                                result=[Content.from_text("REAL-FILE-CONTENT-12345")])
    await _call(mw, ctx)
    assert ctx.removed == [] and live == ["list_files", "read_file", "write_report"]
    assert len(ctx.result) == 1
    assert "REAL-FILE-CONTENT-12345" in ctx.result[0].text   # original output preserved
    assert COUNTDOWN[1] in ctx.result[0].text                # countdown still delivered


async def test_content_list_with_no_text_content_gets_a_new_text_item():
    """A tool result of only non-text Content (e.g. an image) has nothing to
    append the note to in place - a new text Content carries it instead of
    the note being silently dropped."""
    budget = PassBudget(max_calls=1)
    mw = _mw(budget)
    image = Content.from_data(data=b"\x89PNG", media_type="image/png")
    ctx = FakeInvocationContext("read_file", result=[image])
    await _call(mw, ctx)
    assert len(ctx.result) == 2
    assert ctx.result[0] is image                            # original item untouched
    assert ctx.result[1].type == "text"
    assert BUDGET_SPENT in ctx.result[1].text


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
