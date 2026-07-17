"""Verdict parsing and the ReviewerExecutor's two-call review turn."""
import json

from conftest import FakeAgent, FakeCtx

from reflexion_demo.budget import ToolBudget
from reflexion_demo.graph import DraftReady, ReviewerExecutor, ReviewVerdict, parse_verdict


def _factory(agent, spent=0):
    budget = ToolBudget(max_calls=12, spent=spent)
    return lambda: (agent, budget)


def test_parse_verdict_plain_json():
    v = parse_verdict('{"approved": false, "feedback": "missing residency"}')
    assert v is not None and v.approved is False and v.feedback == "missing residency"


def test_parse_verdict_tolerates_code_fences():
    v = parse_verdict('```json\n{"approved": true, "feedback": "ok"}\n```')
    assert v is not None and v.approved is True


def test_parse_verdict_garbage_returns_none():
    assert parse_verdict("I approve of this report.") is None
    assert parse_verdict("") is None


async def test_reviewer_sends_verdict_from_structured_second_call():
    agent = FakeAgent(["I checked the sources.",
                       '{"approved": false, "feedback": "cite file_store.py"}'])
    ctx = FakeCtx()
    await ReviewerExecutor(_factory(agent, spent=5)).on_draft(
        DraftReady(cycle=1, topic="NFS to S3"), ctx)

    [verdict] = ctx.sent
    assert isinstance(verdict, ReviewVerdict)
    assert verdict.approved is False
    assert verdict.feedback == "cite file_store.py"
    assert verdict.cycle == 1
    assert verdict.reviewer_tool_calls == 5

    assert len(agent.prompts) == 2                      # explore, then verdict
    assert "NFS to S3" in agent.prompts[0]
    assert agent.sessions[0] == agent.sessions[1]       # same session: turn 2 sees turn 1
    assert agent.run_kwargs[0] == {}                    # exploration: no format forcing
    assert "response_format" in agent.run_kwargs[1].get("options", {})


async def test_reviewer_retries_once_then_rejects_on_unparseable():
    agent = FakeAgent(["explored", "not json", "still not json"])
    ctx = FakeCtx()
    await ReviewerExecutor(_factory(agent)).on_draft(DraftReady(cycle=2, topic="t"), ctx)

    [verdict] = ctx.sent
    assert verdict.approved is False                    # fail-closed, never approve
    assert "could not produce a valid verdict" in verdict.feedback
    assert len(agent.prompts) == 3                      # explore + verdict + one retry
    assert "not valid JSON" in agent.prompts[2]         # retry names the problem
