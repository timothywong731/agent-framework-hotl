"""The judge predicate: no tools, records every verdict, relays feedback.

Also guards the framework behaviour the terminal semantics depend on:
``max_iterations`` short-circuits before ``should_continue``.
"""
import pytest
from agent_framework import AgentResponse, JudgeVerdict, Message

from reflection_demo.judging import RunLog, make_judge_predicate, make_next_message


class FakeChatResponse:
    def __init__(self, value=None, text=""):
        self.value = value
        self.text = text


class FakeJudgeClient:
    """Duck-typed chat client: records the messages it was asked to judge."""

    def __init__(self, *responses):
        self._responses = list(responses)
        self.calls = []

    async def get_response(self, messages, options=None, **kwargs):
        self.calls.append({"messages": list(messages), "options": options})
        return self._responses.pop(0)


def _result(text):
    return AgentResponse(messages=[Message("assistant", contents=[text])])


async def test_predicate_stops_on_an_answered_verdict(tmp_path):
    client = FakeJudgeClient(FakeChatResponse(JudgeVerdict(answered=True, reasoning="good")))
    log = RunLog(tmp_path / "log.jsonl")
    predicate = make_judge_predicate(client, "judge instructions", log)

    keep_going, feedback = await predicate(
        iteration=1, last_result=_result("the report"),
        original_messages=[Message("user", contents=["the topic"])])

    assert keep_going is False
    assert feedback == "good"
    assert [(v.pass_no, v.answered) for v in log.verdicts] == [(1, True)]


async def test_predicate_continues_and_relays_reasoning(tmp_path):
    client = FakeJudgeClient(
        FakeChatResponse(JudgeVerdict(answered=False, reasoning="no Azure mandate")))
    log = RunLog(tmp_path / "log.jsonl")
    predicate = make_judge_predicate(client, "judge instructions", log)

    keep_going, feedback = await predicate(
        iteration=2, last_result=_result("draft"),
        original_messages=[Message("user", contents=["topic"])])

    assert keep_going is True
    assert feedback == "no Azure mandate"
    assert log.verdicts[0].pass_no == 2


async def test_predicate_asks_for_structured_output(tmp_path):
    client = FakeJudgeClient(FakeChatResponse(JudgeVerdict(answered=True)))
    predicate = make_judge_predicate(client, "judge instructions",
                                     RunLog(tmp_path / "log.jsonl"))
    await predicate(iteration=1, last_result=_result("r"),
                    original_messages=[Message("user", contents=["t"])])
    assert client.calls[0]["options"] == {"response_format": JudgeVerdict}


async def test_judge_sees_the_reply_not_the_report_file(tmp_path):
    """Information asymmetry: the judge is handed the transcript only."""
    client = FakeJudgeClient(FakeChatResponse(JudgeVerdict(answered=True)))
    predicate = make_judge_predicate(client, "JUDGE-INSTRUCTIONS",
                                     RunLog(tmp_path / "log.jsonl"))
    await predicate(iteration=1, last_result=_result("WHAT-THE-AGENT-SAID"),
                    original_messages=[Message("user", contents=["THE-TOPIC"])])

    blob = "\n".join(m.text for m in client.calls[0]["messages"])
    assert "JUDGE-INSTRUCTIONS" in blob
    assert "THE-TOPIC" in blob
    assert "WHAT-THE-AGENT-SAID" in blob
    # No tools were offered to the judge at all.
    assert "tools" not in (client.calls[0]["options"] or {})


async def test_predicate_falls_back_to_markers(tmp_path):
    client = FakeJudgeClient(FakeChatResponse(None, "all good\nVERDICT: DONE"))
    log = RunLog(tmp_path / "log.jsonl")
    predicate = make_judge_predicate(client, "i", log)
    keep_going, _ = await predicate(iteration=1, last_result=_result("r"),
                                    original_messages=[Message("user", contents=["t"])])
    assert keep_going is False
    assert log.verdicts[0].answered is True


def test_next_message_relays_feedback_and_demands_a_save():
    nxt = make_next_message()
    out = nxt(feedback="cite the Azure mandate")
    assert "cite the Azure mandate" in out
    assert "write_report" in out


def test_next_message_without_feedback_still_asks_for_a_save():
    assert "write_report" in make_next_message()(feedback=None)


async def test_max_iterations_short_circuits_before_the_judge():
    """Guards the terminal semantics against a framework upgrade.

    ``AgentLoopMiddleware._evaluate_stop`` must keep checking the cap BEFORE
    calling ``should_continue``. If this ever changes, the capped pass would
    become judged and ``summarize`` would be wrong.
    """
    from agent_framework import AgentLoopMiddleware

    called = []

    def should_continue(**kwargs):
        called.append(kwargs.get("iteration"))
        return True

    loop = AgentLoopMiddleware(should_continue, max_iterations=2)
    stop, feedback = await loop._evaluate_stop({"iteration": 2}, work_iterations=2)
    assert stop is True
    assert feedback is None
    assert called == [], "the judge must not be consulted once the cap has fired"

    stop, _ = await loop._evaluate_stop({"iteration": 1}, work_iterations=1)
    assert stop is False
    assert called == [1]
