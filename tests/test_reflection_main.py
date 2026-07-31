"""CLI helpers, preflight, and the write_report fallback."""
from pathlib import Path

import pytest
from agent_framework import AgentResponse, Message

from reflection_demo.budget import PassBudget
from reflection_demo.judging import RunLog
from reflection_demo.main import (
    DEFAULT_TOPIC,
    build_agent,
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


def _budget_of(agent):
    """The run's ``PassBudget``, dug out of the budget middleware's closure.

    ``build_agent`` returns ``(agent, flag)`` - the budget is internal, and
    widening a production signature for one assertion is the wrong trade. The
    middleware is the only object holding it.
    """
    held = [cell.cell_contents for cell in agent.middleware[0].__closure__ or ()]
    (budget,) = [obj for obj in held if isinstance(obj, PassBudget)]
    return budget


@pytest.mark.parametrize("max_passes", [1, 3])
def test_build_agent_starts_pass_one_non_finalizing(tmp_path, monkeypatch, max_passes):
    """Regression: ``--max-passes 1`` used to prime the budget finalizing.

    That closed exploration on the FIRST budgeted call of the run, right after
    ``worker.md`` promised ``max_tool_calls`` of them - and since
    ``next_message`` never fires on a single-pass run, ``finalize.md`` (the one
    text explaining the strip) could not be delivered either. Pass 1 gets the
    same fresh, non-finalizing budget at every ``--max-passes``; delivery is
    forced by exhaustion, not by pre-emption.
    """
    monkeypatch.setenv("OLLAMA_MODEL", "gemma4:31b")   # OllamaChatClient needs it to construct
    agent, _flag = build_agent(
        tmp_path, tmp_path / "report.md", "topic", "judge instructions",
        RunLog(tmp_path / "log.jsonl"), max_passes, max_tool_calls=6)

    budget = _budget_of(agent)
    assert (budget.finalizing, budget.spent, budget.max_calls) == (False, 0, 6)
