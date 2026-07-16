"""Graph assembly: edge predicates, join barrier, and an LLM-free drive of the real graph."""
from pathlib import Path

import pytest

from conftest import DRIVE_TARGETS, DriveAgent, FakeCtx

from hotl_demo.artifacts import REPOS, ArtifactStore
from hotl_demo.phases import AnalysisDone, PhaseDone, ReportTrigger, RevisionDone, RevisionTrigger
from hotl_demo.pipeline import JoinAnalyses, build_workflow, is_type, revision_for


def test_is_type_condition():
    cond = is_type(PhaseDone)
    assert cond(PhaseDone("discovery")) is True
    assert cond(RevisionDone("discovery")) is False
    assert cond("random string") is False


def test_revision_for_condition_matches_phase_and_unit():
    cond = revision_for("deep_analysis", "oms-monolith")
    assert cond(RevisionTrigger("deep_analysis", "oms-monolith", [])) is True
    assert cond(RevisionTrigger("deep_analysis", "oms-batch-recon", [])) is False
    assert cond(RevisionTrigger("discovery", None, [])) is False
    assert cond(ReportTrigger()) is False


@pytest.mark.asyncio
async def test_join_waits_for_all_analyzers():
    join = JoinAnalyses(expected=2)
    ctx = FakeCtx()
    await join.on_analysis(AnalysisDone("oms-monolith"), ctx)
    assert ctx.sent == []
    await join.on_analysis(AnalysisDone("oms-monolith"), ctx)  # duplicate unit: still waiting
    assert ctx.sent == []
    await join.on_analysis(AnalysisDone("oms-batch-recon"), ctx)
    assert ctx.sent == [PhaseDone("deep_analysis")]


def test_build_workflow_smoke_and_shape(tmp_path, monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "test-model")
    store = ArtifactStore(tmp_path / "run", repos=REPOS)
    workflow = build_workflow(store, Path("sample_data"), scratchpad_path=tmp_path / "pad.md")
    assert {ex.id for ex in workflow.get_executors_list()} == {
        "discovery", "analyze:oms-monolith", "analyze:oms-batch-recon",
        "join", "enterprise_context", "questionnaire", "review", "final_report",
    }


# -- LLM-free drive of the real assembled graph -----------------------------

async def test_workflow_graph_drive_gate_revisions_report(tmp_path, monkeypatch):
    from hotl_demo import phases, report, review
    from hotl_demo.review import LedgerQuestionRequest

    store = ArtifactStore(tmp_path / "run", repos=REPOS)
    calls: list[tuple[str, str, str]] = []

    def agent_factory(*_, name="", **__):
        return DriveAgent(name, store, calls)

    for mod in (phases, report, review):
        monkeypatch.setattr(mod, "Agent", agent_factory)
        monkeypatch.setattr(mod, "OllamaChatClient", lambda: None)
    workflow = build_workflow(store, Path("sample_data"), scratchpad_path=tmp_path / "pad.md")

    # initial pass: 5 raised, ranker consulted once, top-3 (reversed order:
    # the LAST raised) presented, the other 2 deferred before the pause
    requests = {}
    async for ev in workflow.run("start", stream=True):
        if ev.type == "request_info" and isinstance(ev.data, LedgerQuestionRequest):
            requests[ev.request_id] = ev.data
    assert {q.question_id for q in requests.values()} == {"q-3", "q-4", "q-5"}
    assert all(q.importance == "medium" and q.impact for q in requests.values())
    assert [c for c in calls if c[1] == "rank"] and len([c for c in calls if c[1] == "rank"]) == 1
    statuses = {e["id"]: e["status"] for e in store.read_ledger()}
    assert statuses["q-1"] == "deferred" and statuses["q-2"] == "deferred"
    initial = [name for name, kind, _ in calls if kind == "initial"]
    assert initial[0] == "discovery" and len(initial) == 5
    assert initial.index("enterprise_context") > initial.index("analyze_oms-monolith")
    assert initial.index("enterprise_context") > initial.index("analyze_oms-batch-recon")

    # answer all three: revisions are DERIVED from the ledger (q-2/q-3 analyzer
    # attribution is racy between the concurrent analyzers), then the report
    outputs = []
    stream = workflow.run(stream=True, responses={rid: f"answer to {q.question_id}"
                                                  for rid, q in requests.items()})
    async for ev in stream:
        assert ev.type != "request_info"        # review-once: never prompts again
        if ev.type == "output":
            outputs.append(ev.data)
    assert outputs == [str(store.run_dir / "final_report.md")]
    ledger = store.read_ledger()
    answered = {(e["phase"], e["unit"]) for e in ledger if e["status"] == "answered"}
    expected = [n for n, t in DRIVE_TARGETS.items() if t in answered]  # pipeline order
    assert [name for name, kind, _ in calls if kind == "revision"] == expected

    # the questionnaire's revision raised q-6 post-gate: open, never prompted
    assert [e["status"] for e in ledger] == (
        ["deferred", "deferred", "answered", "answered", "answered", "open"])
    final = store.read_report("final_report.md")
    assert "FINAL-VERDICT" in final and "## Adjudication log" in final
    assert "Raised during revision?" in final   # open question surfaces in the log
