"""LLM-free integration: a REAL ``Agent`` + REAL ``AgentLoopMiddleware``, two passes.

Every other reflection test calls the predicate, ``make_next_message`` or
``_evaluate_stop`` directly with hand-built kwargs, and the live E2E runs a
single pass - so nothing in the default suite ever constructed the agent, and a
typo in an ``Agent(...)`` or ``AgentLoopMiddleware(...)`` keyword would only
surface under ``OLLAMA_E2E=1``. That gap is what hid both the unparseable-verdict
crash and the missing session. This test drives ``build_agent``'s real wiring
through two passes against a scripted chat client: the judge is called, its
feedback reaches pass 2, the session accumulates, and pass 1's verdict is
deliberately non-conforming so the lazy-parse guard is exercised end to end.
"""
import json

from agent_framework import BaseChatClient, ChatResponse, JudgeVerdict, Message

from reflection_demo.judging import RunLog
from reflection_demo.main import build_agent, persist_fallback
from reflection_demo.prompting import render_judge_instructions, render_worker_prompt

TOPIC = "Assess migrating OMS file storage to Amazon S3."
# A non-conforming pass-1 verdict on purpose: the real ChatResponse.value
# parses lazily and RAISES pydantic.ValidationError on this, so the predicate's
# try/except is the only thing standing between it and a dead run.
JUDGE_PASS_1 = "The report never mentions the Azure mandate.\nVERDICT: MORE"
JUDGE_PASS_2 = json.dumps({"answered": True, "reasoning": "the mandate is now cited"})
WORKER_PASS_1 = "# Draft one\n\nS3 looks fine." * 3
WORKER_PASS_2 = "# Draft two\n\nThe enterprise cloud strategy mandates Azure." * 3


def _scripted_client_class(worker_replies, judge_replies, calls):
    """A no-arg chat client class standing in for BOTH of build_agent's clients.

    ``build_agent`` constructs ``OllamaChatClient()`` twice with no arguments -
    once for the judge, once for the worker - so the stand-in must be no-arg
    constructible. It dispatches on ``response_format``, which only the judge's
    call carries, rather than on construction order (which is an implementation
    detail of ``build_agent`` and would silently mis-route if it changed).
    """
    class ScriptedChatClient(BaseChatClient):
        async def _inner_get_response(self, *, messages, stream, options, **kwargs):
            opts = dict(options or {})
            response_format = opts.get("response_format")
            role = "judge" if response_format is not None else "worker"
            calls.append({"role": role, "messages": list(messages), "options": opts})
            queue = judge_replies if role == "judge" else worker_replies
            text = queue.pop(0) if queue else "(script exhausted)"
            # response_format is threaded back in so ChatResponse.value behaves
            # exactly as production does - lazily parsing, and raising on
            # JUDGE_PASS_1. A fake that skipped this could not reproduce the bug.
            return ChatResponse(messages=[Message("assistant", contents=[text])],
                                response_format=response_format)

    return ScriptedChatClient


async def test_two_passes_through_the_real_loop(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setenv("OLLAMA_NUM_CTX", "2048")
    monkeypatch.setattr(
        "agent_framework.ollama.OllamaChatClient",
        _scripted_client_class([WORKER_PASS_1, WORKER_PASS_2],
                               [JUDGE_PASS_1, JUDGE_PASS_2], calls))

    report_path = tmp_path / "report.md"
    log = RunLog(tmp_path / "log.jsonl")
    max_passes = 3  # above the 2 passes the judge will allow, so the cap does not fire
    agent, flag = build_agent(tmp_path, report_path,
                              render_judge_instructions(topic=TOPIC), log, max_passes)

    session = agent.create_session()
    prompt = render_worker_prompt(topic=TOPIC, max_passes=max_passes)
    result = await agent.run(prompt, session=session)

    worker_calls = [c for c in calls if c["role"] == "worker"]
    judge_calls = [c for c in calls if c["role"] == "judge"]

    # The loop ran the agent twice and consulted the judge after each pass.
    assert len(worker_calls) == 2
    assert len(judge_calls) == 2

    # FINDING 1: pass 1's verdict does not parse, so the predicate had to fall
    # through to the VERDICT: markers instead of letting ValidationError escape.
    # Reaching a second pass at all is the proof the guard held.
    assert [(v.pass_no, v.answered) for v in log.verdicts] == [(1, False), (2, True)]
    assert log.finish(max_passes, report_path) == ("answered", 2)

    # The judge's reasoning reached pass 2's input via make_next_message.
    pass_2_text = "\n".join(m.text for m in worker_calls[1]["messages"])
    assert "A reviewer judged your previous report incomplete." in pass_2_text
    assert "The report never mentions the Azure mandate." in pass_2_text

    # FINDING 2: the session carried the transcript forward. AgentLoopMiddleware
    # REPLACES context.messages between passes, so the original topic prompt can
    # only be in pass 2's input because the auto-attached history provider
    # reloaded it from the session.
    assert TOPIC in pass_2_text
    assert WORKER_PASS_1 in pass_2_text

    # inject_progress=True, narrowed by the session to the LATEST entry only:
    # exactly one "Progress so far:" message, carrying pass 1 and nothing else.
    progress = [m.text for m in worker_calls[1]["messages"] if m.text.startswith("Progress so far:")]
    assert len(progress) == 1

    # num_ctx reached BOTH clients: the worker through Agent.default_options,
    # the judge through the predicate's per-call options (it has no Agent to
    # carry a default for it). Guards the earlier fix on real wiring.
    assert worker_calls[0]["options"]["num_ctx"] == 2048
    assert judge_calls[0]["options"]["num_ctx"] == 2048
    assert judge_calls[0]["options"]["response_format"] is JudgeVerdict

    # The judge holds no tools; the worker does. The whole point of the demo.
    assert not judge_calls[0]["options"].get("tools")
    assert worker_calls[0]["options"]["tools"]

    # write_report was never called by the script, so the fallback persists the
    # longest assistant text across the aggregated two-pass response.
    assert flag.written is False
    persist_fallback(result, report_path, flag)
    assert report_path.read_text(encoding="utf-8") == WORKER_PASS_2
