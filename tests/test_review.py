"""Review gate: request emission, verdict handling, re-run queue, review-once rule."""
import pytest

from conftest import FakeAgent, FakeCtx

from hotl_demo.artifacts import REPOS, ArtifactStore
from hotl_demo.phases import PhaseDone, ReportTrigger, RevisionDone, RevisionTrigger
from hotl_demo.review import (
    LedgerQuestionRequest,
    ReviewExecutor,
    affected_targets,
    answers_for,
    fallback_order,
    split_ranked,
    validate_ranking,
)

ORDER = [
    ("discovery", None),
    ("deep_analysis", "oms-monolith"),
    ("deep_analysis", "oms-batch-recon"),
    ("enterprise_context", None),
    ("questionnaire", None),
]


@pytest.fixture(autouse=True)
def _ollama_model_env(monkeypatch):
    # ReviewExecutor's fallback ranker eagerly constructs OllamaChatClient()
    # in __init__ (never lazily), so any ReviewExecutor(...) built without an
    # explicit ranker= needs OLLAMA_MODEL present, same as test_pipeline.py /
    # test_checkpoint.py. Harmless for tests that do pass ranker=.
    monkeypatch.setenv("OLLAMA_MODEL", "test-model")


@pytest.fixture()
def store(tmp_path):
    s = ArtifactStore(tmp_path / "run", repos=REPOS)
    s.raise_question("discovery", None, "Scope?", "recon undocumented", "in scope",
                      importance="medium", impact="swings the verdict")
    s.raise_question("deep_analysis", "oms-batch-recon", "Secrets?", "hardcoded pw", "vault first",
                      importance="medium", impact="swings the verdict")
    s.raise_question("enterprise_context", None, "Region?", "unspecified", "EU",
                      importance="medium", impact="swings the verdict")
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
async def test_fresh_executor_acts_only_on_last_answer(store):
    # Resume scenario: after a checkpoint restore the gate is a NEW instance
    # that never saw on_questionnaire_done. The old in-memory _awaiting counter
    # started at 0 there, so EVERY answer looked like the last -> overlapping
    # revision queues, discovery revised 5x, final_report written 5x (measured).
    # The guard must be ledger-derived: dispatch only when nothing is open.
    review = ReviewExecutor(store, ORDER)              # fresh: no gate entry
    ctx = FakeCtx()
    reqs = [
        LedgerQuestionRequest(q["id"], q["phase"], q["unit"], q["question"],
                              q["context"], q["default_assumption"],
                              importance=q["importance"], impact=q["impact"])
        for q in store.open_questions()
    ]
    for r in reqs[:-1]:
        await review.on_answer(r, f"answer to {r.question_id}", ctx)
        assert ctx.sent == []                          # verdicts still outstanding
    await review.on_answer(reqs[-1], "final answer", ctx)
    assert len(ctx.sent) == 1                          # exactly one dispatch
    assert isinstance(ctx.sent[0], RevisionTrigger)
    assert (ctx.sent[0].phase, ctx.sent[0].unit) == ("discovery", None)


@pytest.mark.asyncio
async def test_review_once_guard(store):
    ctx = FakeCtx()
    review = ReviewExecutor(store, ORDER)
    store.set_review_completed()                       # gate already consumed
    await review.on_questionnaire_done(PhaseDone("questionnaire"), ctx)
    assert ctx.requests == []                          # never prompts again
    assert len(ctx.sent) == 1 and isinstance(ctx.sent[0], ReportTrigger)


def _entry(qid, importance="medium", impact="swings the verdict"):
    return {"id": qid, "phase": "discovery", "unit": None, "question": f"{qid}?",
            "context": "ctx", "default_assumption": "d",
            "importance": importance, "impact": impact, "status": "open",
            "human_answer": None, "asked_at": "t"}


def test_validate_ranking_accepts_any_permutation_with_noise():
    ids = ["q-1", "q-2", "q-3"]
    assert validate_ranking(ids, "q-2\nq-3\nq-1") == ["q-2", "q-3", "q-1"]
    # bullets, numbering, prose - the ids are extracted in order
    assert validate_ranking(ids, "1. q-3\n- q-1\nfinally q-2") == ["q-3", "q-1", "q-2"]


def test_validate_ranking_rejects_missing_duplicate_foreign():
    ids = ["q-1", "q-2", "q-3"]
    assert validate_ranking(ids, "q-1\nq-2") is None                 # missing
    assert validate_ranking(ids, "q-1\nq-1\nq-2\nq-3") is None       # duplicate
    assert validate_ranking(ids, "q-1\nq-2\nq-9") is None            # foreign
    assert validate_ranking(ids, "") is None


def test_split_ranked_winners_by_prefix_outputs_in_ledger_order():
    qs = [_entry("q-1"), _entry("q-2"), _entry("q-3")]
    presented, deferred = split_ranked(["q-3", "q-1", "q-2"], qs, 2)
    assert [q["id"] for q in presented] == ["q-1", "q-3"]   # ledger order kept
    assert [q["id"] for q in deferred] == ["q-2"]
    presented, deferred = split_ranked(["q-2", "q-1", "q-3"], qs, 0)
    assert presented == [] and [q["id"] for q in deferred] == ["q-1", "q-2", "q-3"]


def test_fallback_order_is_tier_then_numeric_id():
    qs = [_entry("q-10", "low"), _entry("q-2", "high"), _entry("q-3", "medium"),
          _entry("q-1", "high")]
    assert fallback_order(qs) == ["q-1", "q-2", "q-3", "q-10"]


def _competitive_store(tmp_path):
    s = ArtifactStore(tmp_path / "run_rank", repos=REPOS)
    s.raise_question("discovery", None, "Q1?", "ctx", "d",
                     importance="low", impact="tidies wording")
    s.raise_question("enterprise_context", None, "Q2?", "ctx", "d",
                     importance="high", impact="changes the 6R approach")
    s.raise_question("questionnaire", None, "Q3?", "ctx", "d",
                     importance="medium", impact="sets one slot")
    return s


@pytest.mark.asyncio
async def test_gate_ranks_presents_winners_defers_losers(tmp_path, capsys):
    store = _competitive_store(tmp_path)
    ranker = FakeAgent(["q-2\nq-3\nq-1"])
    review = ReviewExecutor(store, ORDER, max_questions=2, ranker=ranker)
    ctx = FakeCtx()
    await review.on_questionnaire_done(PhaseDone("questionnaire"), ctx)
    assert [r.question_id for r in ctx.requests] == ["q-2", "q-3"]   # ledger order
    assert (ctx.requests[0].importance, ctx.requests[0].impact) == (
        "high", "changes the 6R approach")
    statuses = {e["id"]: e["status"] for e in store.read_ledger()}
    assert statuses == {"q-1": "deferred", "q-2": "open", "q-3": "open"}
    assert "presenting 2 of 3 open questions (1 deferred to defaults)" in capsys.readouterr().out
    assert len(ranker.prompts) == 1
    for token in ("q-1", "q-2", "q-3", "changes the 6R approach", "[importance: high]"):
        assert token in ranker.prompts[0]
    assert "asked_at" not in ranker.prompts[0]       # raise time carries no signal


@pytest.mark.asyncio
async def test_gate_retries_once_on_invalid_ranking(tmp_path):
    store = _competitive_store(tmp_path)
    ranker = FakeAgent(["utter garbage", "q-3\nq-2\nq-1"])
    review = ReviewExecutor(store, ORDER, max_questions=1, ranker=ranker)
    await review.on_questionnaire_done(PhaseDone("questionnaire"), FakeCtx())
    assert len(ranker.prompts) == 2
    assert "invalid" in ranker.prompts[1]
    assert [e["id"] for e in store.open_questions()] == ["q-3"]      # retry ranking won


@pytest.mark.asyncio
async def test_gate_falls_back_to_tier_order_when_ranker_stays_invalid(tmp_path, capsys):
    store = _competitive_store(tmp_path)
    ranker = FakeAgent(["nope", "still nope"])
    review = ReviewExecutor(store, ORDER, max_questions=1, ranker=ranker)
    await review.on_questionnaire_done(PhaseDone("questionnaire"), FakeCtx())
    # fallback: tier then id -> q-2 (high) wins the single slot
    assert [e["id"] for e in store.open_questions()] == ["q-2"]
    assert "falling back" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_no_ranker_call_when_everything_fits(tmp_path):
    store = _competitive_store(tmp_path)
    ranker = FakeAgent(["should never be consulted"])
    review = ReviewExecutor(store, ORDER, max_questions=3, ranker=ranker)
    ctx = FakeCtx()
    await review.on_questionnaire_done(PhaseDone("questionnaire"), ctx)
    assert ranker.prompts == []
    assert len(ctx.requests) == 3 and ctx.sent == []


@pytest.mark.asyncio
async def test_zero_slots_defers_everything_straight_to_report(tmp_path):
    store = _competitive_store(tmp_path)
    ranker = FakeAgent(["should never be consulted"])
    review = ReviewExecutor(store, ORDER, max_questions=0, ranker=ranker)
    ctx = FakeCtx()
    await review.on_questionnaire_done(PhaseDone("questionnaire"), ctx)
    assert ranker.prompts == [] and ctx.requests == []
    assert len(ctx.sent) == 1 and isinstance(ctx.sent[0], ReportTrigger)
    assert all(e["status"] == "deferred" for e in store.read_ledger())
    assert store.review_completed() is True
