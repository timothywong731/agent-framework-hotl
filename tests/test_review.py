"""Review gate: request emission, verdict handling, re-run queue, review-once rule."""
import pytest

from conftest import FakeCtx

from hotl_demo.artifacts import REPOS, ArtifactStore
from hotl_demo.phases import PhaseDone, ReportTrigger, RevisionDone, RevisionTrigger
from hotl_demo.review import (
    LedgerQuestionRequest,
    ReviewExecutor,
    affected_targets,
    answers_for,
)

ORDER = [
    ("discovery", None),
    ("deep_analysis", "oms-monolith"),
    ("deep_analysis", "oms-batch-recon"),
    ("enterprise_context", None),
    ("questionnaire", None),
]


@pytest.fixture()
def store(tmp_path):
    s = ArtifactStore(tmp_path / "run", repos=REPOS)
    s.raise_question("discovery", None, "Scope?", "recon undocumented", "in scope")
    s.raise_question("deep_analysis", "oms-batch-recon", "Secrets?", "hardcoded pw", "vault first")
    s.raise_question("enterprise_context", None, "Region?", "unspecified", "EU")
    return s


def test_affected_targets_ordered_and_deduped():
    ledger = [
        {"phase": "enterprise_context", "unit": None, "status": "answered"},
        {"phase": "discovery", "unit": None, "status": "answered"},
        {"phase": "discovery", "unit": None, "status": "answered"},   # second answered q, same phase
        {"phase": "deep_analysis", "unit": "oms-batch-recon", "status": "declined"},
    ]
    assert affected_targets(ledger, ORDER) == [("discovery", None), ("enterprise_context", None)]


def test_answers_for_filters_phase_unit_and_status():
    ledger = [
        {"id": "q-1", "phase": "deep_analysis", "unit": "oms-monolith", "status": "answered"},
        {"id": "q-2", "phase": "deep_analysis", "unit": "oms-batch-recon", "status": "answered"},
        {"id": "q-3", "phase": "deep_analysis", "unit": "oms-monolith", "status": "declined"},
    ]
    assert [a["id"] for a in answers_for(ledger, "deep_analysis", "oms-monolith")] == ["q-1"]


@pytest.mark.asyncio
async def test_gate_emits_one_request_per_open_question_and_sets_flag(store):
    ctx = FakeCtx()
    review = ReviewExecutor(store, ORDER)
    await review.on_questionnaire_done(PhaseDone("questionnaire"), ctx)
    assert store.review_completed() is True
    assert [r.question_id for r in ctx.requests] == ["q-1", "q-2", "q-3"]
    assert all(isinstance(r, LedgerQuestionRequest) for r in ctx.requests)
    assert ctx.sent == []  # nothing dispatched until answers arrive


@pytest.mark.asyncio
async def test_answers_drive_sequential_revisions_then_report(store):
    ctx = FakeCtx()
    review = ReviewExecutor(store, ORDER)
    await review.on_questionnaire_done(PhaseDone("questionnaire"), ctx)
    reqs = list(ctx.requests)
    # q-1 answered, q-2 answered, q-3 declined (whitespace)
    await review.on_answer(reqs[0], "recon is in scope", ctx)
    await review.on_answer(reqs[1], "rotate now, vault during migration", ctx)
    assert ctx.sent == []                              # still waiting for q-3
    await review.on_answer(reqs[2], "   ", ctx)
    assert len(ctx.sent) == 1                          # first revision dispatched
    t1 = ctx.sent[0]
    assert isinstance(t1, RevisionTrigger) and (t1.phase, t1.unit) == ("discovery", None)
    assert t1.answers[0]["human_answer"] == "recon is in scope"
    # ledger updated
    statuses = {e["id"]: e["status"] for e in store.read_ledger()}
    assert statuses == {"q-1": "answered", "q-2": "answered", "q-3": "declined"}
    # revision completes -> next affected target
    await review.on_revision_done(RevisionDone("discovery", None), ctx)
    t2 = ctx.sent[1]
    assert (t2.phase, t2.unit) == ("deep_analysis", "oms-batch-recon")
    # last revision completes -> report
    await review.on_revision_done(RevisionDone("deep_analysis", "oms-batch-recon"), ctx)
    assert isinstance(ctx.sent[2], ReportTrigger)


@pytest.mark.asyncio
async def test_all_declined_goes_straight_to_report(store):
    ctx = FakeCtx()
    review = ReviewExecutor(store, ORDER)
    await review.on_questionnaire_done(PhaseDone("questionnaire"), ctx)
    for r in list(ctx.requests):
        await review.on_answer(r, "", ctx)
    assert len(ctx.sent) == 1 and isinstance(ctx.sent[0], ReportTrigger)


@pytest.mark.asyncio
async def test_no_open_questions_goes_straight_to_report(tmp_path):
    empty_store = ArtifactStore(tmp_path / "run2", repos=REPOS)
    ctx = FakeCtx()
    review = ReviewExecutor(empty_store, ORDER)
    await review.on_questionnaire_done(PhaseDone("questionnaire"), ctx)
    assert ctx.requests == []
    assert len(ctx.sent) == 1 and isinstance(ctx.sent[0], ReportTrigger)
    assert empty_store.review_completed() is True


@pytest.mark.asyncio
async def test_review_once_guard(store):
    ctx = FakeCtx()
    review = ReviewExecutor(store, ORDER)
    store.set_review_completed()                       # gate already consumed
    await review.on_questionnaire_done(PhaseDone("questionnaire"), ctx)
    assert ctx.requests == []                          # never prompts again
    assert len(ctx.sent) == 1 and isinstance(ctx.sent[0], ReportTrigger)
