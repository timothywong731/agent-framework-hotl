from pathlib import Path

import pytest

from conftest import FakeCtx

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


def test_build_workflow_smoke(tmp_path, monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "test-model")
    store = ArtifactStore(tmp_path / "run", repos=REPOS)
    workflow = build_workflow(store, Path("sample_data"), scratchpad_path=tmp_path / "pad.md")
    assert workflow is not None
