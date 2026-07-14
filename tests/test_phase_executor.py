import pytest

from conftest import FakeAgent, FakeCtx

from hotl_demo.artifacts import REPOS, ArtifactStore
from hotl_demo.phases import (
    AnalysisDone,
    PhaseDone,
    PhaseExecutor,
    PhaseSpec,
    RevisionDone,
    RevisionTrigger,
)


@pytest.fixture()
def store(tmp_path):
    return ArtifactStore(tmp_path / "run", repos=REPOS)


def _spec(name="discovery", unit=None, executor_id=None):
    return PhaseSpec(
        name=name, unit=unit, executor_id=executor_id or name,
        report_filename=f"phase_x_{name}{'_' + unit if unit else ''}.md",
        instructions="do the thing", load_sources=lambda: "SOURCES",
    )


def _executor(store, tmp_path, spec, agent):
    return PhaseExecutor(spec, store, scratchpad_path=tmp_path / "pad.md", agent=agent)


@pytest.mark.asyncio
async def test_initial_run_writes_report_and_sends_phase_done(store, tmp_path):
    spec = _spec()
    agent = FakeAgent(
        ["# Report"], side_effect=lambda p: store.update_memory("discovery", None, "k", "v")
    )
    ctx = FakeCtx()
    await _executor(store, tmp_path, spec, agent).on_start("start", ctx)
    assert store.read_report(spec.report_filename) == "# Report"
    assert ctx.sent == [PhaseDone("discovery")]
    assert len(agent.prompts) == 1  # memory was updated -> no nudge


@pytest.mark.asyncio
async def test_analyzer_sends_analysis_done(store, tmp_path):
    spec = _spec("deep_analysis", "oms-monolith", "analyze:oms-monolith")
    agent = FakeAgent(
        ["# R"], side_effect=lambda p: store.update_memory("deep_analysis", "oms-monolith", "k", "v")
    )
    ctx = FakeCtx()
    await _executor(store, tmp_path, spec, agent).on_upstream(PhaseDone("discovery"), ctx)
    assert ctx.sent == [AnalysisDone("oms-monolith")]


@pytest.mark.asyncio
async def test_nudge_fires_once_when_no_memory_written(store, tmp_path):
    spec = _spec()
    agent = FakeAgent(["# Report", "ignored nudge reply"])  # never writes memory
    ctx = FakeCtx()
    await _executor(store, tmp_path, spec, agent).on_start("start", ctx)
    assert len(agent.prompts) == 2                      # initial + exactly one nudge
    assert "update_memory" in agent.prompts[1]          # nudge asks for memory calls
    report = store.read_report(spec.report_filename)
    assert report.startswith("# Report")
    assert "no memory entries" in report                # noted, not crashed
    assert ctx.sent == [PhaseDone("discovery")]         # pipeline continues


@pytest.mark.asyncio
async def test_revision_run_rewrites_report_and_sends_revision_done(store, tmp_path):
    spec = _spec()
    store.write_report(spec.report_filename, "OLD")
    open_id = store.raise_question("enterprise_context", None, "Still open?", "ctx", "EU")
    agent = FakeAgent(["NEW"])
    ctx = FakeCtx()
    trig = RevisionTrigger("discovery", None, answers=[{
        "id": "q-9", "question": "Scope?", "human_answer": "recon in scope",
        "default_assumption": "in scope",
    }])
    await _executor(store, tmp_path, spec, agent).on_revision(trig, ctx)
    assert store.read_report(spec.report_filename) == "NEW"
    assert ctx.sent == [RevisionDone("discovery", None)]
    assert "OLD" in agent.prompts[0]                 # previous report included
    assert "recon in scope" in agent.prompts[0]      # human answer included
    assert open_id in agent.prompts[0]               # open ledger included (dup suppression)
    assert "Still open?" in agent.prompts[0]


@pytest.mark.asyncio
async def test_empty_agent_text_falls_back(store, tmp_path):
    spec = _spec()
    agent = FakeAgent([""], side_effect=lambda p: store.update_memory("discovery", None, "k", "v"))
    ctx = FakeCtx()
    await _executor(store, tmp_path, spec, agent).on_start("start", ctx)
    assert "no text" in store.read_report(spec.report_filename)
