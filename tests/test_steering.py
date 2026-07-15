"""Live steering: the scratchpad watermark and the notification middleware."""
from pathlib import Path

from hotl_demo.steering import ScratchpadWatch


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
