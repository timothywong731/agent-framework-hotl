"""The judge predicate: no tools, records every verdict, relays feedback.

Also guards the framework behaviour the terminal semantics depend on:
``max_iterations`` short-circuits before ``should_continue``.
"""
import json

from agent_framework import AgentResponse, ChatResponse, Content, JudgeVerdict, Message

from reflection_demo.judging import RunLog, make_judge_predicate, make_next_message


class FakeChatResponse:
    def __init__(self, value=None, text=""):
        self.value = value
        self.text = text


class UnparseableChatResponse:
    """A reply whose ``value`` RAISES, as the real ``ChatResponse`` does.

    ``ChatResponse.value`` is a lazily-parsing property: with
    ``response_format=JudgeVerdict`` it calls ``model_validate_json`` on first
    access and raises ``pydantic.ValidationError`` for any non-empty reply
    that is not schema-conforming JSON. ``FakeChatResponse`` sets ``value`` as
    a plain attribute and so is structurally incapable of raising - which is
    why the marker-fallback tests passed while the production path crashed.
    """

    def __init__(self, text):
        self.text = text

    @property
    def value(self):
        # Delegate to the real type so the exception is exactly what
        # production raises, not a hand-rolled stand-in that could drift.
        return ChatResponse(messages=[Message("assistant", contents=[self.text])],
                            response_format=JudgeVerdict).value


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


def _report(tmp_path, text=None):
    """A report path, written only when ``text`` is given.

    An unwritten path is the real state of pass 1 before ``write_report``
    fires, so it is the honest default for tests that do not care.
    """
    path = tmp_path / "report.md"
    if text is not None:
        path.write_text(text, encoding="utf-8")
    return path


async def test_predicate_stops_on_an_answered_verdict(tmp_path):
    client = FakeJudgeClient(FakeChatResponse(JudgeVerdict(answered=True, reasoning="good")))
    log = RunLog(tmp_path / "log.jsonl")
    predicate = make_judge_predicate(client, "judge instructions", log, 4096,
                                     _report(tmp_path, "# Report"))

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
    predicate = make_judge_predicate(client, "judge instructions", log, 4096,
                                     _report(tmp_path, "# Report"))

    keep_going, feedback = await predicate(
        iteration=2, last_result=_result("draft"),
        original_messages=[Message("user", contents=["topic"])])

    assert keep_going is True
    assert feedback == "no Azure mandate"
    assert log.verdicts[0].pass_no == 2


async def test_predicate_asks_for_structured_output(tmp_path):
    client = FakeJudgeClient(FakeChatResponse(JudgeVerdict(answered=True)))
    predicate = make_judge_predicate(client, "judge instructions",
                                     RunLog(tmp_path / "log.jsonl"), 4096,
                                     _report(tmp_path, "# Report"))
    await predicate(iteration=1, last_result=_result("r"),
                    original_messages=[Message("user", contents=["t"])])
    assert client.calls[0]["options"] == {"response_format": JudgeVerdict, "num_ctx": 4096}


async def test_judge_sees_what_the_worker_said_and_what_it_wrote(tmp_path):
    """Both channels reach the judge, and the corpus still does not.

    The worker delivers via ``write_report``, so its reply is only an
    acknowledgement; judging that alone rated the claim instead of the work
    and made ``answered`` unreachable. The report is therefore handed over as
    prompt text - by the harness, with no tool given to the judge.
    """
    client = FakeJudgeClient(FakeChatResponse(JudgeVerdict(answered=True)))
    predicate = make_judge_predicate(client, "JUDGE-INSTRUCTIONS",
                                     RunLog(tmp_path / "log.jsonl"), 4096,
                                     _report(tmp_path, "WHAT-THE-AGENT-WROTE"))
    await predicate(iteration=1, last_result=_result("WHAT-THE-AGENT-SAID"),
                    original_messages=[Message("user", contents=["THE-TOPIC"])])

    blob = "\n".join(m.text for m in client.calls[0]["messages"])
    assert "JUDGE-INSTRUCTIONS" in blob
    assert "THE-TOPIC" in blob
    assert "WHAT-THE-AGENT-SAID" in blob
    assert "WHAT-THE-AGENT-WROTE" in blob
    # Labelled, so the judge can tell an acknowledgement from the deliverable.
    assert "SAID" in blob and "WROTE" in blob
    # No tools were offered to the judge at all.
    assert "tools" not in (client.calls[0]["options"] or {})


async def test_judge_is_told_in_words_when_no_report_was_saved(tmp_path):
    """A pass where ``write_report`` never fired has no file to read.

    The prompt must say so rather than carry a bare empty string - a judge
    told nothing was delivered should reject, which is the correct verdict -
    and reading a missing path must not raise.
    """
    client = FakeJudgeClient(FakeChatResponse(JudgeVerdict(answered=False, reasoning="none")))
    predicate = make_judge_predicate(client, "i", RunLog(tmp_path / "log.jsonl"), 4096,
                                     _report(tmp_path))  # deliberately unwritten
    keep_going, _ = await predicate(iteration=1, last_result=_result("all done!"),
                                    original_messages=[Message("user", contents=["t"])])

    assert keep_going is True
    blob = "\n".join(m.text for m in client.calls[0]["messages"])
    assert "no report has been saved" in blob
    assert "report.md" in blob


async def test_judge_is_told_in_words_when_the_report_is_empty(tmp_path):
    """Same branch for a whitespace-only file: an empty artifact is no artifact."""
    client = FakeJudgeClient(FakeChatResponse(JudgeVerdict(answered=False, reasoning="none")))
    predicate = make_judge_predicate(client, "i", RunLog(tmp_path / "log.jsonl"), 4096,
                                     _report(tmp_path, "   \n\n"))
    await predicate(iteration=1, last_result=_result("all done!"),
                    original_messages=[Message("user", contents=["t"])])

    assert "no report has been saved" in "\n".join(
        m.text for m in client.calls[0]["messages"])


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

    The report the harness hands over separately is plain text the worker
    chose to write, not a channel the judge can pull the corpus through.
    """
    client = FakeJudgeClient(FakeChatResponse(JudgeVerdict(answered=True)))
    predicate = make_judge_predicate(client, "judge instructions",
                                     RunLog(tmp_path / "log.jsonl"), 4096,
                                     _report(tmp_path, "# Draft\n\nIt mentions a standard."))
    secret = "ZZTOPSECRETCORPUSMARKER42"
    tool_result = Content.from_function_result("call-1", result=f"The migration standard says: {secret}")
    # The framework emits a function_result under role="tool", NOT under the
    # assistant message - so a fixture that hides it in an assistant message
    # would still satisfy "exactly one assistant message" post-splat and
    # could not catch a reintroduced leak. This mirrors the real layout:
    # assistant text + a separate tool message carrying the raw output.
    last_result = AgentResponse(messages=[
        Message("tool", contents=[tool_result]),
        Message("assistant", contents=["Yes, it mentions a standard."]),
    ])

    await predicate(iteration=1, last_result=last_result,
                    original_messages=[Message("user", contents=["t"])])

    sent = client.calls[0]["messages"]
    assistant_messages = [m for m in sent if m.role == "assistant"]
    assert len(assistant_messages) == 1
    assert assistant_messages[0].text == last_result.text
    # Assert on SERIALIZED content, not on ``m.text``: Message.text already
    # filters to text content, so a .text-only assertion can never see a
    # function_result and is vacuous both before and after the fix. The
    # serialized form is what actually crosses the wire to the judge.
    assert secret not in json.dumps([m.to_dict() for m in sent], default=str)
    # Stronger and layout-independent: nothing but plain text may reach the
    # judge at all, so any future content type (reasoning traces, citations,
    # attachments) is caught without naming it.
    assert all(c.type == "text" for m in sent for c in m.contents)
    # The tool message itself must not be forwarded - role check, since a
    # tool message whose output happened to be empty would slip past the
    # content assertions above. The trailing pair is the harness handing over
    # the report and then asking the question; both are the harness speaking,
    # hence "user".
    assert list(m.role for m in sent) == [
        "system", "user", "user", "user", "assistant", "user", "user"]


async def test_predicate_survives_an_unparseable_structured_reply(tmp_path):
    """Regression: ``ChatResponse.value`` RAISES on a non-conforming reply.

    Without the ``try/except ValueError`` in the predicate this propagates out
    of ``should_continue`` -> ``_evaluate_stop`` -> ``agent.run()`` and kills
    the run before ``persist_fallback`` and ``log.finish`` - no report, no
    outcome line. It also makes the whole ``VERDICT:`` marker branch
    unreachable in production, since the raise happens before
    ``read_verdict`` is ever called.
    """
    client = FakeJudgeClient(UnparseableChatResponse("Looks complete.\nVERDICT: DONE"))
    log = RunLog(tmp_path / "log.jsonl")
    predicate = make_judge_predicate(client, "i", log, 4096, _report(tmp_path, "# Report"))

    keep_going, feedback = await predicate(
        iteration=1, last_result=_result("r"),
        original_messages=[Message("user", contents=["t"])])

    # Fell through to the markers instead of propagating, and still recorded.
    assert keep_going is False
    assert log.verdicts[0].answered is True
    assert "VERDICT: DONE" in feedback


async def test_predicate_fails_open_on_an_unparseable_marker_less_reply(tmp_path):
    """The other half of the guard: no markers either -> keep looping.

    Pins the fail-OPEN direction on the path that actually happens in
    production (a raising ``value``), not just on the synthetic ``value=None``.
    """
    client = FakeJudgeClient(UnparseableChatResponse("I am not sure, honestly."))
    log = RunLog(tmp_path / "log.jsonl")
    predicate = make_judge_predicate(client, "i", log, 4096, _report(tmp_path, "# Report"))

    keep_going, _ = await predicate(iteration=1, last_result=_result("r"),
                                    original_messages=[Message("user", contents=["t"])])

    assert keep_going is True
    assert log.verdicts[0].answered is False


async def test_predicate_falls_back_to_markers(tmp_path):
    client = FakeJudgeClient(FakeChatResponse(None, "all good\nVERDICT: DONE"))
    log = RunLog(tmp_path / "log.jsonl")
    predicate = make_judge_predicate(client, "i", log, 4096, _report(tmp_path, "# Report"))
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
