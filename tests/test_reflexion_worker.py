"""WorkerExecutor cycle logic, forced finalize, write_report fallback, review log."""
import json

from conftest import FakeAgent, FakeCtx

from reflexion_demo.budget import ToolBudget
from reflexion_demo.graph import (
    DraftReady,
    ReviewerExecutor,
    ReviewVerdict,
    WorkerExecutor,
    build_reflexion_workflow,
)
from reflexion_demo.tools import ReportFlag, atomic_write


def _worker_factory(run_dir, *, write=True, spent=3,
                    texts=("draft text", "retry text")):
    """Fake agent factory: each call yields a FakeAgent whose side effect
    mimics the write_report tool (writes the file, sets the flag)."""
    calls = []

    def factory(finalize=False):
        flag = ReportFlag()
        agent_holder = {}

        def side_effect(prompt):
            if write:
                atomic_write(run_dir / "report.md", f"# Draft after: {prompt[:40]}")
                flag.written = True

        agent = FakeAgent(list(texts), side_effect=side_effect)
        agent_holder["agent"] = agent
        calls.append({"finalize": finalize, "agent": agent, "flag": flag})
        return agent, ToolBudget(max_calls=12, spent=spent), flag

    factory.calls = calls
    return factory


def _log_lines(run_dir):
    path = run_dir / "review_log.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


async def test_on_topic_drafts_and_announces_cycle_one(tmp_path):
    factory = _worker_factory(tmp_path)
    worker = WorkerExecutor(factory, tmp_path, max_cycles=3, max_tool_calls=12)
    ctx = FakeCtx()
    await worker.on_topic("NFS to S3", ctx)

    assert ctx.sent == [DraftReady(cycle=1, topic="NFS to S3")]
    assert factory.calls[0]["finalize"] is False
    assert "NFS to S3" in factory.calls[0]["agent"].prompts[0]
    assert (tmp_path / "report.md").exists()


async def test_approved_verdict_yields_and_logs(tmp_path):
    factory = _worker_factory(tmp_path)
    worker = WorkerExecutor(factory, tmp_path, max_cycles=3, max_tool_calls=12)
    await worker.on_topic("t", FakeCtx())

    ctx = FakeCtx()
    await worker.on_verdict(
        ReviewVerdict(approved=True, feedback="good", cycle=1, reviewer_tool_calls=7), ctx)

    assert ctx.sent == []
    assert len(ctx.outputs) == 1 and "approved" in ctx.outputs[0]
    lines = _log_lines(tmp_path)
    assert lines[0] == {"cycle": 1, "approved": True, "feedback": "good",
                        "forced": False, "worker_tool_calls": 3,
                        "reviewer_tool_calls": 7}
    assert lines[1]["outcome"] == "approved" and lines[1]["cycles"] == 1


async def test_rejection_with_cycles_left_revises_with_feedback(tmp_path):
    factory = _worker_factory(tmp_path)
    worker = WorkerExecutor(factory, tmp_path, max_cycles=3, max_tool_calls=12)
    await worker.on_topic("t", FakeCtx())

    ctx = FakeCtx()
    await worker.on_verdict(
        ReviewVerdict(approved=False, feedback="cover Azure mandate",
                      cycle=1, reviewer_tool_calls=2), ctx)

    assert ctx.sent == [DraftReady(cycle=2, topic="t")]
    assert ctx.outputs == []
    revision_agent = factory.calls[1]["agent"]
    assert factory.calls[1]["finalize"] is False
    assert "cover Azure mandate" in revision_agent.prompts[0]
    assert "# Draft after:" in revision_agent.prompts[0]   # previous report inlined


async def test_rejection_at_budget_forces_toolless_finalize(tmp_path):
    factory = _worker_factory(tmp_path)
    worker = WorkerExecutor(factory, tmp_path, max_cycles=3, max_tool_calls=12)
    await worker.on_topic("t", FakeCtx())

    ctx = FakeCtx()
    await worker.on_verdict(
        ReviewVerdict(approved=False, feedback="still wrong",
                      cycle=3, reviewer_tool_calls=4), ctx)

    assert ctx.sent == []                                   # no fourth review
    assert len(ctx.outputs) == 1 and "forced" in ctx.outputs[0]
    assert factory.calls[1]["finalize"] is True             # read tools stripped at construction
    assert "reasoning for a long time" in factory.calls[1]["agent"].prompts[0]
    lines = _log_lines(tmp_path)
    assert lines[0]["forced"] is False and lines[0]["approved"] is False
    assert lines[1] == {"cycle": 4, "approved": False, "feedback": "still wrong",
                        "forced": True, "worker_tool_calls": 3,
                        "reviewer_tool_calls": None}
    assert lines[2]["outcome"] == "forced" and lines[2]["cycles"] == 3


async def test_missing_write_report_gets_one_nudge_then_text_fallback(tmp_path):
    # The classic failure: the model emits the full report as plain chat text,
    # then answers the nudge with filler. The fallback must keep the report.
    factory = _worker_factory(
        tmp_path, write=False,                              # tool never "runs"
        texts=["# Migration Report\n\nLots of real content here.", "Done."])
    worker = WorkerExecutor(factory, tmp_path, max_cycles=3, max_tool_calls=12)
    await worker.on_topic("t", FakeCtx())

    agent = factory.calls[0]["agent"]
    assert len(agent.prompts) == 2                          # draft + one nudge
    assert "write_report" in agent.prompts[1]
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert report == "# Migration Report\n\nLots of real content here."


async def test_fallback_uses_nudge_text_when_draft_said_nothing(tmp_path):
    factory = _worker_factory(
        tmp_path, write=False,
        texts=["", "# Report delivered only after the nudge."])
    worker = WorkerExecutor(factory, tmp_path, max_cycles=3, max_tool_calls=12)
    await worker.on_topic("t", FakeCtx())

    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert report == "# Report delivered only after the nudge."


async def test_full_loop_reject_then_approve_llm_free(tmp_path):
    worker_factory = _worker_factory(tmp_path)
    reviewer_scripts = [
        ["looked at sources", '{"approved": false, "feedback": "add residency"}'],
        ["looked again", '{"approved": true, "feedback": "solid"}'],
    ]

    def reviewer_factory():
        return FakeAgent(reviewer_scripts.pop(0)), ToolBudget(max_calls=12, spent=1)

    workflow = build_reflexion_workflow(
        WorkerExecutor(worker_factory, tmp_path, max_cycles=3, max_tool_calls=12),
        ReviewerExecutor(reviewer_factory),
    )
    outputs = []
    async for event in workflow.run("NFS to S3", stream=True):
        if event.type == "output":
            outputs.append(event.data)

    assert len(outputs) == 1 and "approved" in outputs[0]
    assert len(worker_factory.calls) == 2                   # draft + one revision
    lines = _log_lines(tmp_path)
    assert [ln.get("approved") for ln in lines[:2]] == [False, True]
    assert lines[2]["outcome"] == "approved"
