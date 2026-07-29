"""CLI helpers, preflight, and the write_report fallback."""
from pathlib import Path

import pytest
from agent_framework import AgentResponse, Message

from reflection_demo.main import (
    DEFAULT_TOPIC,
    ensure_corpus,
    model_present,
    normalize_host,
    persist_fallback,
)
from reflection_demo.tools import make_report_tools


def test_default_topic_matches_the_reflexion_demo_exactly():
    """The A/B is void if the two demos assess different things."""
    from reflexion_demo.main import DEFAULT_TOPIC as REFLEXION_TOPIC
    assert DEFAULT_TOPIC == REFLEXION_TOPIC


def test_normalize_host_adds_a_scheme():
    assert normalize_host("localhost:11434") == "http://localhost:11434"
    assert normalize_host("https://box:1234/") == "https://box:1234"


def test_model_present_resolves_bare_names_to_latest():
    tags = {"models": [{"name": "gemma4:31b"}, {"name": "mistral:latest"}]}
    assert model_present(tags, "gemma4:31b")
    assert model_present(tags, "mistral")
    assert not model_present(tags, "llama3")


def test_ensure_corpus_fails_fast_on_a_missing_directory(tmp_path):
    with pytest.raises(SystemExit) as exc:
        ensure_corpus(tmp_path / "nope")
    assert "not found" in str(exc.value)


def test_ensure_corpus_accepts_a_real_directory(tmp_path):
    ensure_corpus(tmp_path)  # must not raise


def test_persist_fallback_is_a_no_op_when_the_tool_wrote(tmp_path):
    report = tmp_path / "report.md"
    _write, flag = make_report_tools(report)
    report.write_text("written by the tool", encoding="utf-8")
    flag.written = True
    persist_fallback(AgentResponse(messages=[Message("assistant", contents=["chatter"])]),
                     report, flag)
    assert report.read_text(encoding="utf-8") == "written by the tool"


def test_persist_fallback_keeps_the_longest_assistant_reply(tmp_path):
    """The model often emits the full report as chat text, then answers the
    next nudge with filler - the longest reply is the report, not the last."""
    report = tmp_path / "report.md"
    _write, flag = make_report_tools(report)
    result = AgentResponse(messages=[
        Message("assistant", contents=["# The full report, long and detailed"]),
        Message("user", contents=["A reviewer judged your previous report incomplete."]),
        Message("assistant", contents=["Done."]),
    ])
    persist_fallback(result, report, flag)
    assert report.read_text(encoding="utf-8") == "# The full report, long and detailed"


def test_persist_fallback_writes_a_placeholder_when_there_is_nothing(tmp_path):
    report = tmp_path / "report.md"
    _write, flag = make_report_tools(report)
    persist_fallback(AgentResponse(messages=[]), report, flag)
    assert report.read_text(encoding="utf-8") == "(no report produced)"
