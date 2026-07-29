"""The judge predicate: no tools, records every verdict, relays feedback.

Also guards the framework behaviour the terminal semantics depend on:
``max_iterations`` short-circuits before ``should_continue``.
"""
import pytest
from agent_framework import AgentResponse, Content, JudgeVerdict, Message

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
    predicate = make_judge_predicate(client, "judge instructions", log, 4096)

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
    predicate = make_judge_predicate(client, "judge instructions", log, 4096)

    keep_going, feedback = await predicate(
        iteration=2, last_result=_result("draft"),
        original_messages=[Message("user", contents=["topic"])])

    assert keep_going is True
    assert feedback == "no Azure mandate"
    assert log.verdicts[0].pass_no == 2


async def test_predicate_asks_for_structured_output(tmp_path):
    client = FakeJudgeClient(FakeChatResponse(JudgeVerdict(answered=True)))
    predicate = make_judge_predicate(client, "judge instructions",
                                     RunLog(tmp_path / "log.jsonl"), 4096)
    await predicate(iteration=1, last_result=_result("r"),
                    original_messages=[Message("user", contents=["t"])])
    assert client.calls[0]["options"] == {"response_format": JudgeVerdict, "num_ctx": 4096}


async def test_judge_sees_the_reply_not_the_report_file(tmp_path):
    """Information asymmetry: the judge is handed the transcript only."""
    client = FakeJudgeClient(FakeChatResponse(JudgeVerdict(answered=True)))
    predicate = make_judge_predicate(client, "JUDGE-INSTRUCTIONS",
                                     RunLog(tmp_path / "log.jsonl"), 4096)
    await predicate(iteration=1, last_result=_result("WHAT-THE-AGENT-SAID"),
                    original_messages=[Message("user", contents=["THE-TOPIC"])])

    blob = "\n".join(m.text for m in client.calls[0]["messages"])
    assert "JUDGE-INSTRUCTIONS" in blob
    assert "THE-TOPIC" in blob
    assert "WHAT-THE-AGENT-SAID" in blob
    # No tools were offered to the judge at all.
    assert "tools" not in (client.calls[0]["options"] or {})


async def test_judge_never_sees_tool_results_or_reasoning_traces(tmp_path):
    """Regression: the framework's own message layout leaks raw tool output.

    A real tool-using pass produces a message list with a ``function_result``
    content item carrying the tool's raw output (here: the corpus) alongside
    the worker's final text reply. Splatting ``*last_result.messages`` (what
    the framework's ``_build_judge_condition`` does) forwards that content
    item verbatim, handing the judge corpus text even when the worker's
    actual answer never quotes it. The predicate must forward
    ``last_result.text`` only - exactly one assistant message standing in
    for the whole reply, and no non-text content anywhere in the transcript.
    """
    client = FakeJudgeClient(FakeChatResponse(JudgeVerdict(answered=True)))
    predicate = make_judge_predicate(client, "judge instructions",
                                     RunLog(tmp_path / "log.jsonl"), 4096)
    secret = "ZZTOPSECRETCORPUSMARKER42"
    tool_result = Content.from_function_result("call-1", result=f"The migration standard says: {secret}")
    last_result = AgentResponse(messages=[
        Message("assistant", contents=[tool_result]),
        Message("assistant", contents=["Yes, it mentions a standard."]),
    ])

    await predicate(iteration=1, last_result=last_result,
                    original_messages=[Message("user", contents=["t"])])

    sent = client.calls[0]["messages"]
    assistant_messages = [m for m in sent if m.role == "assistant"]
    assert len(assistant_messages) == 1
    assert assistant_messages[0].text == last_result.text
    blob = "\n".join(m.text for m in sent)
    assert secret not in blob


async def test_predicate_falls_back_to_markers(tmp_path):
    client = FakeJudgeClient(FakeChatResponse(None, "all good\nVERDICT: DONE"))
    log = RunLog(tmp_path / "log.jsonl")
    predicate = make_judge_predicate(client, "i", log, 4096)
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
