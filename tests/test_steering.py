"""Live steering: the scratchpad watermark and the notification middleware."""
from pathlib import Path

import pytest

from hotl_demo.steering import ScratchpadWatch, make_steering_middleware


def _watch(tmp_path: Path, initial: str | None = None) -> tuple[ScratchpadWatch, Path]:
    pad = tmp_path / "scratchpad.md"
    if initial is not None:
        pad.write_text(initial, encoding="utf-8")
    return ScratchpadWatch(pad), pad


def test_first_poll_baselines_and_reports_nothing(tmp_path):
    # The agent's own read_scratchpad must never be echoed back at it.
    watch, _ = _watch(tmp_path, "focus on security")
    assert watch.poll() is None


def test_unchanged_content_reports_nothing(tmp_path):
    watch, _ = _watch(tmp_path, "focus on security")
    watch.poll()
    assert watch.poll() is None


def test_changed_content_is_reported_once(tmp_path):
    watch, pad = _watch(tmp_path, "focus on security")
    watch.poll()
    pad.write_text("actually, focus on cost", encoding="utf-8")
    assert watch.poll() == "actually, focus on cost"
    assert watch.poll() is None  # reported once, then quiet


def test_noop_resave_of_identical_content_is_silent(tmp_path):
    # Compared by content, not mtime: saving without editing must not notify.
    watch, pad = _watch(tmp_path, "focus on security")
    watch.poll()
    pad.write_text("focus on security", encoding="utf-8")
    assert watch.poll() is None


def test_missing_file_baselines_to_empty(tmp_path):
    watch, pad = _watch(tmp_path)  # no file at all
    assert watch.poll() is None
    pad.write_text("late guidance", encoding="utf-8")
    assert watch.poll() == "late guidance"


def test_cleared_content_returns_empty_then_new_content_reports(tmp_path):
    # Clearing yields "" (falsy - the middleware skips it) but the watermark
    # must still advance, so later content is delivered normally.
    watch, pad = _watch(tmp_path, "focus on security")
    watch.poll()
    pad.write_text("", encoding="utf-8")
    assert watch.poll() == ""
    pad.write_text("new guidance", encoding="utf-8")
    assert watch.poll() == "new guidance"


class FakeInjector:
    """Records enqueue_messages calls in place of MessageInjectionMiddleware."""

    def __init__(self):
        self.calls = []  # list[tuple[session, messages]]

    def enqueue_messages(self, session, messages):
        self.calls.append((session, messages))


class FakeFunctionContext:
    """Duck-typed FunctionInvocationContext: the middleware only reads .session."""

    def __init__(self, session="session-sentinel"):
        self.session = session


async def _call(mw, context):
    """Invoke the middleware, recording whether call_next was awaited."""
    awaited = []

    async def call_next():
        awaited.append(True)

    await mw(context, call_next)
    return awaited


@pytest.mark.asyncio
async def test_change_is_enqueued_once_with_notice_wrapping(tmp_path):
    watch, pad = _watch(tmp_path, "original")
    injector = FakeInjector()
    mw = make_steering_middleware(watch, injector, "analyze:oms-monolith")
    ctx = FakeFunctionContext()

    assert await _call(mw, ctx) == [True]  # first call baselines
    assert injector.calls == []

    pad.write_text("prioritise the Oracle licensing question", encoding="utf-8")
    await _call(mw, ctx)

    assert len(injector.calls) == 1
    session, messages = injector.calls[0]
    assert session == "session-sentinel"
    assert len(messages) == 1
    assert "prioritise the Oracle licensing question" in messages[0]
    assert "STEERING UPDATE" in messages[0]

    await _call(mw, ctx)
    assert len(injector.calls) == 1  # enqueued once, not on every later tool call


@pytest.mark.asyncio
async def test_call_next_is_always_awaited_even_with_no_change(tmp_path):
    watch, _ = _watch(tmp_path, "original")
    mw = make_steering_middleware(watch, FakeInjector(), "discovery")
    assert await _call(mw, FakeFunctionContext()) == [True]
    assert await _call(mw, FakeFunctionContext()) == [True]


@pytest.mark.asyncio
async def test_cleared_scratchpad_enqueues_nothing(tmp_path):
    # Withdrawing guidance is not new guidance to act on.
    watch, pad = _watch(tmp_path, "original")
    injector = FakeInjector()
    mw = make_steering_middleware(watch, injector, "discovery")
    await _call(mw, FakeFunctionContext())
    pad.write_text("", encoding="utf-8")
    await _call(mw, FakeFunctionContext())
    assert injector.calls == []


@pytest.mark.asyncio
async def test_missing_session_is_skipped_not_raised(tmp_path):
    watch, pad = _watch(tmp_path, "original")
    injector = FakeInjector()
    mw = make_steering_middleware(watch, injector, "discovery")
    await _call(mw, FakeFunctionContext(session=None))
    pad.write_text("new guidance", encoding="utf-8")
    await _call(mw, FakeFunctionContext(session=None))  # must not raise
    assert injector.calls == []


@pytest.mark.asyncio
async def test_notice_is_brace_safe(tmp_path):
    # The scratchpad is human-written: braces must survive verbatim.
    watch, pad = _watch(tmp_path, "original")
    injector = FakeInjector()
    mw = make_steering_middleware(watch, injector, "discovery")
    await _call(mw, FakeFunctionContext())
    pad.write_text('use {placeholder} and {"json": true}', encoding="utf-8")
    await _call(mw, FakeFunctionContext())
    assert '{placeholder}' in injector.calls[0][1][0]
    assert '{"json": true}' in injector.calls[0][1][0]


# -- defensive reads: the human picks the editor, and therefore the encoding ----

def test_non_utf8_scratchpad_does_not_raise(tmp_path):
    # Notepad's "Unicode" and PowerShell 5.1's ">" both write UTF-16. poll()
    # runs AFTER call_next(), so raising here would report an already-succeeded
    # tool call as failed.
    watch, pad = _watch(tmp_path, "original")
    watch.poll()
    pad.write_bytes("focus on cost".encode("utf-16"))
    assert watch.poll() is not None  # decoded lossily, not raised


def test_unreadable_scratchpad_leaves_watermark_alone(tmp_path):
    # A file locked mid-save (or deleted between exists() and read) must not
    # lose the edit: report nothing now, pick it up on the next tool call.
    watch, pad = _watch(tmp_path, "original")
    watch.poll()

    def boom(*a, **k):
        raise PermissionError("locked by the editor")

    original_read = type(pad).read_text
    try:
        type(pad).read_text = boom
        assert watch.poll() is None  # swallowed, not raised
    finally:
        type(pad).read_text = original_read

    pad.write_text("late guidance", encoding="utf-8")
    assert watch.poll() == "late guidance"  # watermark never advanced past it


@pytest.mark.asyncio
async def test_non_utf8_scratchpad_never_breaks_the_tool_call(tmp_path):
    watch, pad = _watch(tmp_path, "original")
    injector = FakeInjector()
    mw = make_steering_middleware(watch, injector, "discovery")
    await _call(mw, FakeFunctionContext())
    pad.write_bytes("focus on cost".encode("utf-16"))
    assert await _call(mw, FakeFunctionContext()) == [True]  # call_next still awaited
