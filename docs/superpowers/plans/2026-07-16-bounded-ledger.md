# Bounded Ledger Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Questions carry a validated `importance` enum and an `impact` statement; the review gate has `--max-questions` slots (default 3) and a semantic LLM ranker picks the winners — losers are `deferred` and their defaults stand.

**Architecture:** Typed `(str, Enum)`s land in `artifacts.py` (plain strings on disk — no reader changes). The gate gains a tool-less ranker `Agent` prompted via `prompts/rank_questions.md`; its output is fenced by pure functions (`validate_ranking` → one corrective retry → deterministic degraded fallback) so a flaky local model can never kill the run. Deferral is written to the ledger *before* the workflow idles, which keeps `--pause`/`review.jsonl` correct for free. Raise order and `asked_at` carry zero ranking signal.

**Tech Stack:** Python ≥3.10 (`enum.StrEnum` is 3.11+ — use `(str, Enum)`), Poetry, pytest (explicit `@pytest.mark.asyncio` per convention), `agent-framework ~=1.11`, Jinja2 prompts.

**Spec:** `docs/superpowers/specs/2026-07-15-bounded-ledger-design.md` (rev 2) — authoritative. The pipeline and checkpointing specs stay authoritative for everything they cover.

## Global Constraints

- **`review.py` must NOT use `from __future__ import annotations`** — `@response_handler` inspects annotations at runtime.
- **Never create `tests/__init__.py`**; tests import shared doubles via `from conftest import ...`.
- **Tests are LLM-free by default** (`addopts = "-m 'not ollama'"`). The ranker is exercised through its `ranker=` seam (`FakeAgent`) or a monkeypatched `review.Agent` — never a real client.
- **Every ledger/memory mutation goes through `ArtifactStore`.** Tools return `ERROR: ...` strings, never raise.
- **Suite must be green at the end of every task.** Task 2 flips the gate's default to 3 slots, which changes what the 5-question graph tests observe — those test rewrites are *inside* Task 2 for exactly this reason. Do not split them out.
- **Environment note:** `poetry` is NOT on PATH in this shell. Run tests as `.venv/Scripts/python.exe -m pytest` from the repo root (same venv), substituting wherever this plan says `poetry run pytest`.
- **Commit style:** bare messages, no trailers — matching the existing feature commits.
- **Markdown lint** gate covers `README.md`, `CLAUDE.md`, and `src/hotl_demo/prompts` (the new `rank_questions.md` is linted; Jinja in markdown is fine — every existing prompt has it).
- **Verified facts** (do not re-litigate):
  - `(str, Enum)` members are JSON-serialized as their string values and compare equal to plain strings in both directions.
  - Jinja2 `{{ q.id }}` resolves dict keys (attribute-then-getitem), so ledger dicts render directly.
  - `build_phase_specs` only parses prompt files whose stem is in `PHASES` — a new `rank_questions.md` is ignored by phase discovery.
  - Graph tests monkeypatch `phases.Agent` / `report.Agent` factories; the same pattern extends to `review.Agent`.
  - Constructing `Agent(client=OllamaChatClient(), ...)` performs no network I/O; only `.run()` does.

---

### Task 1: Schema — enums, `importance`/`impact`, `defer_questions`, `unresolved_questions`

**Files:**
- Modify: `src/hotl_demo/artifacts.py`
- Modify: `src/hotl_demo/tools.py` (the `raise_question` tool)
- Modify: `tests/test_artifacts.py`, `tests/test_tools.py`, `tests/test_review.py` (fixture), `tests/test_phase_executor.py` (one call), `tests/conftest.py` (DriveAgent's two calls)
- Test: `tests/test_artifacts.py`, `tests/test_tools.py`

**Interfaces:**
- Produces (later tasks rely on these exact names):
  - `Phase`, `QuestionStatus` (OPEN/ANSWERED/DECLINED/DEFERRED), `Importance` (HIGH/MEDIUM/LOW) — all `(str, Enum)` in `hotl_demo.artifacts`; `PHASES = tuple(p.value for p in Phase)` unchanged in shape.
  - `ArtifactStore.raise_question(phase, unit, question, context, default_assumption, *, importance: str, impact: str) -> str` — keyword-only new params, stored verbatim (validation lives in the tool).
  - `ArtifactStore.defer_questions(ids: list[str]) -> None` — one lock, one atomic rewrite, `KeyError` before writing if any id is unknown.
  - `ArtifactStore.unresolved_questions() -> list[dict]` — status `open` or `deferred`, ledger order.
  - Ledger entries gain `"importance"` and `"impact"` keys (plain strings).

- [ ] **Step 1: Write the failing store tests**

Append to `tests/test_artifacts.py` (and extend its import line to `from hotl_demo.artifacts import PHASES, REPOS, ArtifactStore, Importance, Phase, QuestionStatus`):

```python
def _raise(store, n=1, **kw):
    kw.setdefault("importance", "medium")
    kw.setdefault("impact", "swings the verdict")
    return [store.raise_question("discovery", None, f"Q{i}?", "ctx", "default", **kw)
            for i in range(n)]


def test_enums_are_plain_strings_on_disk(store):
    _raise(store, importance=Importance.HIGH)
    raw = json.loads((store.run_dir / "ledger.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert raw["importance"] == "high" and isinstance(raw["importance"], str)
    assert raw["impact"] == "swings the verdict"
    assert raw["status"] == "open" and isinstance(raw["status"], str)
    # str-enum equality works in both directions
    assert raw["status"] == QuestionStatus.OPEN
    assert Phase.DISCOVERY == "discovery" and PHASES[0] == "discovery"


def test_defer_questions_is_atomic_and_terminal(store):
    q1, q2, q3 = _raise(store, 3)
    store.defer_questions([q1, q3])
    statuses = {e["id"]: e["status"] for e in store.read_ledger()}
    assert statuses == {q1: "deferred", q2: "open", q3: "deferred"}
    assert store.read_ledger()[0]["human_answer"] is None   # untouched
    assert [e["id"] for e in store.open_questions()] == [q2]


def test_defer_questions_unknown_id_raises_before_writing(store):
    (q1,) = _raise(store)
    with pytest.raises(KeyError):
        store.defer_questions([q1, "q-99"])
    assert store.read_ledger()[0]["status"] == "open"       # nothing was written


def test_unresolved_includes_open_and_deferred_in_ledger_order(store):
    q1, q2, q3 = _raise(store, 3)
    store.defer_questions([q2])
    store.resolve_question(q1, "answered", "yes")
    assert [e["id"] for e in store.unresolved_questions()] == [q2, q3]
```

- [ ] **Step 2: Run them to verify they fail**

Run: `poetry run pytest tests/test_artifacts.py -v`
Expected: FAIL — `ImportError: cannot import name 'Importance' from 'hotl_demo.artifacts'`.

- [ ] **Step 3: Add the enums and store methods**

In `src/hotl_demo/artifacts.py`, add `from enum import Enum` to the imports, then replace:

```python
PHASES: tuple[str, ...] = ("discovery", "deep_analysis", "enterprise_context", "questionnaire")
REPOS: tuple[str, ...] = ("oms-monolith", "oms-batch-recon")
```

with:

```python
class Phase(str, Enum):
    """Pipeline phases. ``(str, Enum)`` so members serialize as plain strings
    and compare equal to the literals used across artifacts on disk."""

    DISCOVERY = "discovery"
    DEEP_ANALYSIS = "deep_analysis"
    ENTERPRISE_CONTEXT = "enterprise_context"
    QUESTIONNAIRE = "questionnaire"


class QuestionStatus(str, Enum):
    """Ledger question lifecycle. ``deferred`` = lost the review-gate slot
    competition; terminal, default assumption applies."""

    OPEN = "open"
    ANSWERED = "answered"
    DECLINED = "declined"
    DEFERRED = "deferred"


class Importance(str, Enum):
    """Agent-declared question importance; one ranker input among several."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# Derived so existing consumers (memory sections, spec discovery, validation)
# keep receiving plain strings in the established order.
PHASES: tuple[str, ...] = tuple(p.value for p in Phase)
REPOS: tuple[str, ...] = ("oms-monolith", "oms-batch-recon")
```

- [ ] **Step 4: Extend `raise_question` and add the two methods**

Change `raise_question`'s signature and entry (keyword-only new params; the tool validates, the store records):

```python
    def raise_question(self, phase: str, unit: str | None, question: str,
                       context: str, default_assumption: str, *,
                       importance: str, impact: str) -> str:
```

Extend its docstring Args with:

```
            importance: One of the :class:`Importance` values - the raising
                agent's own estimate; validated at the tool layer.
            impact: How the human's answer would change the migration
                decision - the justification for the claimed importance.
```

and inside the entry dict, after `"context": context,` add:

```python
                "impact": impact,
                "importance": str(importance),
```

and change `"status": "open",` to `"status": QuestionStatus.OPEN.value,`.

In `open_questions`, change the filter to `e["status"] == QuestionStatus.OPEN` and append after it:

```python
    def unresolved_questions(self) -> list[dict]:
        """Return entries still lacking an outcome: ``open`` or ``deferred``.

        Exists for prompt-level duplicate suppression - a deferred question
        must stay visible to agents or a revision re-raises it as new.
        :meth:`open_questions` keeps its stricter meaning (presented and
        unresolved), which the gate, pause seeding, and answer guard rely on.
        """
        wanted = (QuestionStatus.OPEN, QuestionStatus.DEFERRED)
        return [e for e in self.read_ledger() if e["status"] in wanted]

    def defer_questions(self, ids: list[str]) -> None:
        """Mark the slot-competition losers ``deferred`` - one atomic rewrite.

        Args:
            ids: Ledger ids to defer.

        Raises:
            KeyError: Any unknown id - raised BEFORE writing, so a bad call
                cannot half-apply.
        """
        with self._lock:
            entries = self._read_ledger_unlocked()
            by_id = {e["id"]: e for e in entries}
            missing = [i for i in ids if i not in by_id]
            if missing:
                raise KeyError(missing[0])
            for i in ids:
                by_id[i]["status"] = QuestionStatus.DEFERRED.value
            _atomic_write(self._ledger_path, "".join(json.dumps(e) + "\n" for e in entries))
```

- [ ] **Step 5: Teach the tool the two parameters**

In `src/hotl_demo/tools.py`, extend the artifacts import to `from .artifacts import ArtifactStore, Importance` and replace the `raise_question` tool with:

```python
    @tool(approval_mode="never_require")
    def raise_question(question: str, context: str, default_assumption: str,
                       importance: str, impact: str) -> str:
        """Raise a question that requires human clarification or adjudication.
        Use when evidence conflicts or a decision-critical fact is missing.
        Provide: the question; the evidence context; the default assumption
        you will proceed with until a human answers; importance - exactly one
        of "high" (answer materially changes migration approach, scope, or
        cost), "medium" (affects one workstream or sequencing), or "low"
        (clarification that changes no decision); and impact - one or two
        sentences on how the human's answer would change the migration
        decision. Returns the question id."""
        # Validation failures return ERROR strings (never raise): the framework
        # feeds them back so the model can correct its call.
        if not question.strip() or not default_assumption.strip():
            return "ERROR: question and default_assumption must both be non-empty. Retry with both."
        if not impact.strip():
            return "ERROR: impact must explain how the human's answer would change the migration decision."
        try:
            level = Importance(importance.strip().lower())
        except ValueError:
            return "ERROR: importance must be exactly one of: high, medium, low."
        qid = store.raise_question(
            phase, unit, question.strip(), context.strip(), default_assumption.strip(),
            importance=level.value, impact=impact.strip(),
        )
        return f"Recorded {qid}. Proceed using your stated default assumption."
```

- [ ] **Step 6: Add the failing tool tests, then sweep every call site**

Append to `tests/test_tools.py`:

```python
def test_raise_question_validates_importance_and_impact(store, tmp_path):
    _, _, raise_question, _ = _tools(store, tmp_path)
    out = raise_question("Q?", "ctx", "d", "critical", "changes everything")
    assert out.startswith("ERROR") and "high, medium, low" in out
    assert raise_question("Q?", "ctx", "d", "high", "  ").startswith("ERROR")
    ok = raise_question("Q?", "ctx", "d", " High ", "changes everything")
    assert "q-1" in ok
    entry = store.read_ledger()[0]
    assert entry["importance"] == "high" and entry["impact"] == "changes everything"
    assert store.read_ledger()[0]["status"] == "open"
```

Update the two existing tool tests: `test_raise_question_appends_with_phase_and_unit` calls become `raise_question("RTO?", "not stated in PDF 2", "assume 4h", "medium", "sets the DR budget")`; in `test_raise_question_validates_args` both calls gain trailing `, "medium", "impact"` arguments.

Sweep the direct store call sites, appending `, importance="medium", impact="swings the verdict"` (keyword) to every `store.raise_question(...)` call:

- `tests/conftest.py` — DriveAgent's two calls (initial + revision branches).
- `tests/test_review.py` — the three calls in the `store` fixture.
- `tests/test_phase_executor.py` — the single call in `test_revision_run_rewrites_report_and_sends_revision_done`.
- `tests/test_artifacts.py` — the calls in `test_raise_question_assigns_sequential_ids_and_appends`, `test_open_questions_and_resolve`, and `test_concurrent_raises_get_unique_ids` (inside `worker`).

- [ ] **Step 7: Teach the raising agents the criteria (`_duties.md`)**

In `src/hotl_demo/prompts/_duties.md`, replace duty 3's sentence "Provide the question, the evidence context, and the default assumption you will proceed with - then proceed using that default." with:

```markdown
   Provide: the question; the evidence context; the default assumption you
   will proceed with; an importance - "high" if the answer materially changes
   the migration approach, scope, or cost, "medium" if it affects one
   workstream or its sequencing, "low" if it only tightens the report; and an
   impact - one or two sentences on how the human's answer would change the
   migration decision. Then proceed using your stated default.
```

and change its closing sentence "never re-raise a question that is already open; reference its id instead." to "never re-raise a question that is already listed there - open or deferred; reference its id instead."

Run: `poetry run pytest tests/test_markdown_lint.py -v` — Expected: PASS (prompts are lint-gated).

- [ ] **Step 8: Run the whole suite and commit**

Run: `poetry run pytest`
Expected: all green (the new store/tool tests pass; every swept call site compiles and behaves as before).

```bash
git add src/hotl_demo/artifacts.py src/hotl_demo/tools.py src/hotl_demo/prompts/_duties.md tests/
git commit -m "feat: typed enums + importance/impact on questions; defer/unresolved store ops

Phase/QuestionStatus/Importance are (str, Enum) - plain strings on disk, so
ledger readers and the frontend are untouched. raise_question records the
agent's importance (tool-validated against the enum, ERROR retry) and impact
(how the answer would change the migration decision). defer_questions marks
slot-competition losers in one atomic rewrite; unresolved_questions exists so
deferred questions stay visible for duplicate suppression."
```

---

### Task 2: The gate competes — ranker, fencing, deferral (and the graph tests that observe it)

**Files:**
- Create: `src/hotl_demo/prompts/rank_questions.md`
- Modify: `src/hotl_demo/review.py`
- Modify: `tests/conftest.py` (DriveAgent: ranker script + revision-raiser moves to questionnaire)
- Modify: `tests/test_review.py`, `tests/test_main.py` (LedgerQuestionRequest constructions), `tests/test_pipeline.py`, `tests/test_checkpoint.py`
- Test: `tests/test_review.py`

**Interfaces:**
- Consumes (Task 1): `Importance`, `QuestionStatus`, `defer_questions`, ledger `importance`/`impact` keys.
- Produces: `validate_ranking(candidate_ids, text) -> list[str] | None`; `split_ranked(ranked_ids, open_questions, max_questions) -> tuple[list[dict], list[dict]]`; `fallback_order(open_questions) -> list[str]`; `ReviewExecutor(store, revision_order, max_questions=3, ranker=None)`; `LedgerQuestionRequest` + `importance: str` + `impact: str` fields. Task 3 threads `max_questions` from the CLI.

**Why the graph tests are in this task:** defaulting `max_questions=3` immediately changes what every 5-question graph test observes (3 presented, 2 deferred). Rewriting them here is what keeps the suite green at the task boundary.

- [ ] **Step 1: Write the failing executor-level tests**

Append to `tests/test_review.py` (extend its imports with `from conftest import FakeAgent, FakeCtx` replacing the bare `FakeCtx` import, and add `from hotl_demo.review import fallback_order, split_ranked, validate_ranking` names to the existing `hotl_demo.review` import):

```python
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
```

Existing tests in this file need no behavioral edits: the fixture raises exactly 3 questions and the new default is 3 slots, so no competition fires. The `test_fresh_executor_acts_only_on_last_answer` test builds `LedgerQuestionRequest` from ledger dicts — extend its constructor call with `importance=q["importance"], impact=q["impact"]` once Step 3 adds the fields. `ReviewExecutor(store, ORDER)` call sites stay valid (`max_questions` defaults to 3).

- [ ] **Step 2: Run to verify they fail**

Run: `poetry run pytest tests/test_review.py -v`
Expected: FAIL — `ImportError: cannot import name 'fallback_order' from 'hotl_demo.review'`.

- [ ] **Step 3: Implement the ranker machinery in `review.py`**

Extend the imports (keep NO future-import):

```python
import re
from dataclasses import dataclass

from agent_framework import Agent, Executor, WorkflowContext, handler, response_handler
from agent_framework.ollama import OllamaChatClient

from .artifacts import ArtifactStore, Importance
from .phases import PROMPT_ENV, PhaseDone, ReportTrigger, RevisionDone, RevisionTrigger
```

Add `importance: str` and `impact: str` fields to `LedgerQuestionRequest` (after `default_assumption`), with docstring lines:

```
        importance: The raising agent's declared importance (high/medium/low).
        impact: How the human's answer would change the migration decision.
```

After `answers_for`, add:

```python
_QID_RE = re.compile(r"q-\d+")


def validate_ranking(candidate_ids: list[str], text: str) -> list[str] | None:
    """Extract a ranking from the model's text; None unless it is exactly right.

    Tolerant of bullets, numbering, and prose (ids are regex-extracted in
    order) but strict about the set: every candidate exactly once, nothing
    else. Anything less returns ``None`` so the caller can retry or fall back.

    Args:
        candidate_ids: The ids that must appear.
        text: Raw model output.

    Returns:
        The extracted ordering, or ``None`` when it is not a permutation.

    Example:
        >>> validate_ranking(["q-1", "q-2"], "1. q-2\\n2. q-1")
        ['q-2', 'q-1']
        >>> validate_ranking(["q-1", "q-2"], "q-2") is None
        True
    """
    found = _QID_RE.findall(text or "")
    return found if sorted(found) == sorted(candidate_ids) else None


def split_ranked(ranked_ids: list[str], open_questions: list[dict],
                 max_questions: int) -> tuple[list[dict], list[dict]]:
    """Split into (presented, deferred): winners are the ranked prefix.

    Both output lists are in LEDGER order - ranking decides membership, not
    display order, so prompts and seeded answer sheets stay stable.

    Args:
        ranked_ids: Ids ordered most-influential-first.
        open_questions: The open ledger entries, in ledger order.
        max_questions: Slot count.

    Returns:
        ``(presented, deferred)`` ledger-entry lists.
    """
    winners = set(ranked_ids[:max_questions])
    presented = [q for q in open_questions if q["id"] in winners]
    deferred = [q for q in open_questions if q["id"] not in winners]
    return presented, deferred


def fallback_order(open_questions: list[dict]) -> list[str]:
    """Degraded ranking used ONLY when the ranker fails twice.

    Importance tier, then numeric id - deterministic so a broken model still
    yields a defensible selection. The semantic path is the normal one.

    Args:
        open_questions: Open ledger entries.

    Returns:
        Ids ordered by (tier, numeric id).

    Example:
        >>> fallback_order([{"id": "q-2", "importance": "low"},
        ...                 {"id": "q-7", "importance": "high"}])
        ['q-7', 'q-2']
    """
    tier = {Importance.HIGH: 0, Importance.MEDIUM: 1, Importance.LOW: 2}
    return [q["id"] for q in sorted(
        open_questions,
        key=lambda q: (tier[Importance(q["importance"])], int(q["id"].split("-")[1])),
    )]
```

- [ ] **Step 4: Wire the ranker into `ReviewExecutor`**

Replace `__init__` with:

```python
    def __init__(self, store: ArtifactStore,
                 revision_order: list[tuple[str, str | None]],
                 max_questions: int = 3, ranker: object | None = None) -> None:
        """Remember the store, the re-run ordering, and the slot budget.

        Deliberately no adjudication counters here: gate progress must be
        derived from the ledger so that a checkpoint resume (fresh instance)
        cannot diverge from a live run. See the checkpointing spec.

        Args:
            store: The run's shared artifact store.
            revision_order: Every ``(phase, unit)`` in pipeline order,
                exactly as built from the phase specs.
            max_questions: Slot budget; ``0`` defers everything (the fully
                autonomous defaults-only run).
            ranker: Test seam - a scripted stand-in replaces the tool-less
                Ollama-backed ranking ``Agent`` when provided.
        """
        super().__init__(id="review")
        self._store = store
        self._revision_order = revision_order
        self._max_questions = max_questions
        self._ranker = ranker or Agent(
            client=OllamaChatClient(),  # model comes from OLLAMA_MODEL env var
            name="review_ranker",
            instructions="You rank open review questions by how much their "
                         "answers would change the final migration readiness report.",
        )
        self._queue: list[RevisionTrigger] = []
```

Replace `on_questionnaire_done`'s body after the latch (keep the `review_completed` guard and `set_review_completed()` lines exactly as they are) with:

```python
        open_qs = self._store.open_questions()
        if not open_qs:
            await ctx.send_message(ReportTrigger())
            return
        if self._max_questions == 0:
            presented, deferred = [], open_qs
        elif len(open_qs) <= self._max_questions:
            # No competition: never spend an LLM call ranking a full fit.
            presented, deferred = open_qs, []
        else:
            ranked = await self._rank(open_qs)
            presented, deferred = split_ranked(ranked, open_qs, self._max_questions)
        if deferred:
            # Written BEFORE the workflow idles: a --pause checkpoint and its
            # seeded review.jsonl already reflect the competition.
            self._store.defer_questions([q["id"] for q in deferred])
            print(f"\n== REVIEW - presenting {len(presented)} of {len(open_qs)} "
                  f"open questions ({len(deferred)} deferred to defaults) ==")
        else:
            print(f"\n== REVIEW - {len(open_qs)} open questions ==")
        if not presented:
            await ctx.send_message(ReportTrigger())
            return
        for q in presented:
            # One request_info per question: the workflow idles after this
            # handler returns, until the runner calls run(responses={...}).
            await ctx.request_info(
                request_data=LedgerQuestionRequest(
                    question_id=q["id"], phase=q["phase"], unit=q["unit"],
                    question=q["question"], context=q["context"],
                    default_assumption=q["default_assumption"],
                    importance=q["importance"], impact=q["impact"],
                ),
                response_type=str,
            )
```

and add the `_rank` method after it:

```python
    async def _rank(self, open_qs: list[dict]) -> list[str]:
        """One semantic ranking call, fenced: validate -> retry once -> fallback.

        Args:
            open_qs: Open ledger entries competing for the slots.

        Returns:
            Ids ordered most-influential-first; always a valid permutation.
        """
        ids = [q["id"] for q in open_qs]
        prompt = PROMPT_ENV.get_template("rank_questions.md").render(
            questions=open_qs, max_questions=self._max_questions)
        first = await self._ranker.run(prompt)
        ranked = validate_ranking(ids, first.text or "")
        if ranked is None:
            retry = (prompt
                     + "\n\nYour previous response was invalid:\n"
                     + (first.text or "(empty)")
                     + f"\n\nRespond with exactly {len(ids)} lines - the ids "
                     + ", ".join(ids) + " - one per line, most influential first.")
            second = await self._ranker.run(retry)
            ranked = validate_ranking(ids, second.text or "")
        if ranked is None:
            print("  [ranker] invalid ranking after retry - falling back to importance order")
            ranked = fallback_order(open_qs)
        return ranked
```

Update the class docstring's step 1 to:

```
    1. ``on_questionnaire_done`` - latch ``review_completed`` (the
       review-once rule); when open questions exceed the slot budget, one
       semantic ranking call picks the winners (validated, retried once,
       degraded fallback) and the losers are marked ``deferred`` in the
       ledger BEFORE pausing; then one ``request_info`` per presented
       question idles the run until the CLI resumes it with responses.
```

- [ ] **Step 5: Write the ranking prompt template**

Create `src/hotl_demo/prompts/rank_questions.md`:

```markdown
You are the review-gate ranker for a cloud migration readiness assessment.
Below are {{ questions|length }} open questions raised by analysis agents.
Only {{ max_questions }} can be presented to the human reviewer; the rest
proceed on their stated default assumptions.

Rank ALL of them by expected influence on the final migration readiness
report: the questions whose human answers would change the report's verdict,
scope, cost, or approach the most belong first. Judge the substance of each
question, its impact statement, and its evidence. The declared importance is
the raising agent's own estimate - weigh it, but your judgment prevails.
Question ids and the order below carry NO signal.

{% for q in questions %}
- {{ q.id }} [importance: {{ q.importance }}] {{ q.question }}
  Impact if answered: {{ q.impact }}
  Evidence: {{ q.context }}
  Default if unanswered: {{ q.default_assumption }}
{% endfor %}

Respond with exactly {{ questions|length }} lines: the question ids, one per
line, most influential first. No other text.
```

- [ ] **Step 6: Run the executor tests**

Run: `poetry run pytest tests/test_review.py -v`
Expected: PASS — all pre-existing plus the new ones.

- [ ] **Step 7: Fix the `LedgerQuestionRequest` constructions in `tests/test_main.py`**

All three constructor calls gain the two new arguments — in `test_prompt_human_declines_on_eof` and both inside `test_map_answers_missing_id_declines_and_unknown_ids_surface`, change the pattern

```python
LedgerQuestionRequest("q-1", "discovery", None, "Q?", "c", "d")
```

to

```python
LedgerQuestionRequest("q-1", "discovery", None, "Q?", "c", "d", "medium", "impact")
```

Run: `poetry run pytest tests/test_main.py -v` — Expected: PASS.

- [ ] **Step 8: Teach `DriveAgent` the ranker and move its revision-raiser**

In `tests/conftest.py`, replace `DriveAgent.run` with (and update the class docstring's second sentence to "raises one question per phase on the initial pass, one extra during the questionnaire's revision, answers the review ranker with REVERSED ledger order, records call order."):

```python
    async def run(self, prompt, *, session=None):
        if self.name == "review_ranker":
            # Reversed ledger order: deterministic, and deliberately NOT the
            # raise order - proves selection follows the ranking.
            ids = [q["id"] for q in self.store.open_questions()]
            self.calls.append((self.name, "rank", prompt))
            return FakeAgentResult("\n".join(reversed(ids)))
        if self.name == "final_report":
            self.calls.append((self.name, "report", prompt))
            return FakeAgentResult("FINAL-VERDICT")
        kind = "revision" if "## HUMAN ANSWERS" in prompt else "initial"
        self.calls.append((self.name, kind, prompt))
        phase, unit = DRIVE_TARGETS[self.name]
        if kind == "initial":
            self.store.update_memory(phase, unit, f"finding_{len(self.calls)}", "v")
            self.store.raise_question(phase, unit, f"Q from {self.name}?", "ctx", "default",
                                      importance="medium", impact="swings the verdict")
        elif self.name == "questionnaire":
            # Post-gate question: questionnaire is a ranking WINNER (reversed
            # order favors late raisers), so its revision reliably fires when
            # its question is answered - unlike discovery, which now defers.
            self.store.raise_question(phase, unit, "Raised during revision?", "ctx", "post-gate",
                                      importance="low", impact="none")
        return FakeAgentResult(f"REPORT[{self.name}][{kind}]")
```

- [ ] **Step 9: Rewrite the two graph tests for the 3-presented / 2-deferred reality**

With reversed-ledger ranking and 3 slots, winners are the LAST three raised: `q-3`, `q-4`, `q-5`; `q-1` (discovery) and `q-2` (one analyzer — attribution between the two concurrent analyzers is racy, so expectations must be DERIVED from the ledger, never hardcoded by name).

In `tests/test_pipeline.py`, replace the body of `test_workflow_graph_drive_gate_revisions_report` with:

```python
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
```

In `tests/test_checkpoint.py`, replace the body of `test_pause_resume_cycle_revises_each_target_exactly_once` from the `patch_agents` helper onward with:

```python
    def patch_agents(store, calls):
        def agent_factory(*_, name="", **__):
            return DriveAgent(name, store, calls)
        for mod in (phases, report, review):
            monkeypatch.setattr(mod, "Agent", agent_factory)
            monkeypatch.setattr(mod, "OllamaChatClient", lambda: None)

    # ---- process 1: run to the gate, checkpointing on ----
    store1, calls1 = ArtifactStore(run_dir, repos=REPOS), []
    patch_agents(store1, calls1)
    wf1 = build_workflow(store1, data, scratchpad_path=tmp_path / "pad.md",
                         checkpoint_storage=storage)
    pending1 = {}
    async for ev in wf1.run("start", stream=True):
        if ev.type == "request_info" and isinstance(ev.data, LedgerQuestionRequest):
            pending1[ev.request_id] = ev.data
    # 5 raised; reversed-order ranking presents the LAST three; 2 deferred
    assert {q.question_id for q in pending1.values()} == {"q-3", "q-4", "q-5"}

    # the pause artifact: seeded with EXACTLY the presented set
    sheet = render_review_lines(store1.open_questions())
    assert [json.loads(l)["id"] for l in sheet.splitlines()] == ["q-3", "q-4", "q-5"]
    answered = "".join(
        line.replace('"answer": ""', f'"answer": "answer to {json.loads(line)["id"]}"') + "\n"
        for line in sheet.splitlines()
    )
    answers = parse_review_answers(answered)
    # decline-by-omission (q-5 is deterministically questionnaire's - it raises
    # last) and an unknown id, exercising the sheet's edge semantics end to end
    del answers["q-5"]
    answers["q-99"] = "ghost"

    # ---- process 2: fresh store, fresh workflow, same run dir ----
    store2, calls2 = ArtifactStore(run_dir, repos=REPOS), []
    patch_agents(store2, calls2)
    wf2 = build_workflow(store2, data, scratchpad_path=tmp_path / "pad.md",
                         checkpoint_storage=storage)
    gate = gate_checkpoint(await storage.list_checkpoints(workflow_name=WORKFLOW_NAME))
    assert gate is not None

    pending2 = {}
    async for ev in wf2.run(checkpoint_id=gate.checkpoint_id, stream=True):
        if ev.type == "request_info" and isinstance(ev.data, LedgerQuestionRequest):
            pending2[ev.request_id] = ev.data
    assert set(pending2) == set(pending1)      # stable request ids
    assert calls2 == []                        # restore re-ran ZERO phases (and no re-rank)

    outputs = []
    responses, unknown = map_answers(pending2, answers)
    assert unknown == ["q-99"]                 # the ghost id surfaced, not applied
    async for ev in wf2.run(stream=True, responses=responses):
        assert ev.type != "request_info"       # review-once survives the restart
        if ev.type == "output":
            outputs.append(ev.data)

    # one ordered queue; expectations DERIVED (analyzer attribution is racy);
    # q-5 declined by omission means the questionnaire never revises
    ledger = store2.read_ledger()
    answered_targets = {(e["phase"], e["unit"]) for e in ledger if e["status"] == "answered"}
    expected = [n for n, t in DRIVE_TARGETS.items() if t in answered_targets]
    assert [name for name, kind, _ in calls2 if kind == "revision"] == expected
    assert [name for name, kind, _ in calls2 if kind == "rank"] == []   # gate never re-ranks
    assert [name for name, kind, _ in calls2 if kind == "report"] == ["final_report"]
    assert outputs == [str(store2.run_dir / "final_report.md")]
    assert [e["status"] for e in ledger] == (
        ["deferred", "deferred", "answered", "answered", "declined"])
```

Also extend that test's conftest import line to `from conftest import DRIVE_TARGETS, DriveAgent` and its `from hotl_demo import phases, report` to `from hotl_demo import phases, report, review` (adjusting the in-function import accordingly).

- [ ] **Step 10: Run the whole suite and commit**

Run: `poetry run pytest`
Expected: all green.

```bash
git add src/hotl_demo/review.py src/hotl_demo/prompts/rank_questions.md tests/
git commit -m "feat: the review gate competes - semantic ranker, slot budget, deferral

When open questions exceed max_questions (default 3), one tool-less ranker
Agent orders them by expected swing on the final report - question, impact,
and evidence judged semantically, the declared importance weighed but not
decisive, raise order and asked_at carrying zero signal. Output is fenced:
validate_ranking demands an exact permutation, one corrective retry, then a
deterministic (tier, id) fallback announced on stdout. Losers are marked
deferred in one atomic write BEFORE the pause, so checkpoints and seeded
review.jsonl sheets already reflect the competition. No ranking call when
everything fits or when max_questions=0 (the fully autonomous run)."
```

---

### Task 3: Plumbing and surfaces — CLI flag, resume guard, adjudication log, dup suppression

**Files:**
- Modify: `src/hotl_demo/pipeline.py` (thread `max_questions`)
- Modify: `src/hotl_demo/main.py` (`--max-questions`, `already_resumed`, display lines)
- Modify: `src/hotl_demo/report.py` (deferred branch)
- Modify: `src/hotl_demo/phases.py` (dup suppression via `unresolved_questions`)
- Test: `tests/test_main.py`, `tests/test_report.py`, `tests/test_phase_executor.py`

**Interfaces:**
- Consumes: `ReviewExecutor(..., max_questions=...)` (Task 2); `QuestionStatus`, `unresolved_questions` (Task 1).
- Produces: `build_workflow(store, base_dir, scratchpad_path=..., checkpoint_storage=None, max_questions=3)`; `demo --max-questions N`.

**Note:** between Tasks 2 and 3 the CLI's `already_resumed` would falsely refuse a first resume of a competitive pause (deferred entries look like verdicts to the old predicate). No test exercises that window; this task closes it — do not reorder past it.

- [ ] **Step 1: Write the failing tests**

In `tests/test_main.py`, replace `test_already_resumed_derives_from_ledger_not_the_sheet` with:

```python
def test_already_resumed_means_verdicts_not_just_non_open():
    # A --pause run defers slot-competition losers BEFORE any resume exists,
    # so deferred entries must not read as "a resume already ran". Only
    # verdicts - answered/declined - prove that.
    assert already_resumed([]) is False
    assert already_resumed([_q("q-1"), _q("q-2")]) is False                    # all open
    assert already_resumed([{**_q("q-1"), "status": "deferred"}]) is False     # deferred != verdict
    assert already_resumed([_q("q-1"), {**_q("q-2"), "status": "answered"}]) is True
    assert already_resumed([{**_q("q-1"), "status": "declined"}]) is True
```

Append to `tests/test_main.py`:

```python
def test_prompt_human_shows_importance_and_impact(monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda _: "an answer")
    q = LedgerQuestionRequest("q-1", "discovery", None, "Scope?", "ctx", "in scope",
                              "high", "changes the 6R approach")
    assert _prompt_human(q) == "an answer"
    out = capsys.readouterr().out
    assert "Importance: high" in out and "Impact if answered: changes the 6R approach" in out
```

Append to `tests/test_report.py` (its import from `hotl_demo.report` must include `render_adjudication_log` — extend the existing import line if it does not already):

```python
def test_adjudication_log_deferred_branch_is_distinct_from_open():
    log = render_adjudication_log([
        {"id": "q-1", "phase": "discovery", "unit": None, "question": "A?",
         "status": "deferred", "human_answer": None, "default_assumption": "da"},
        {"id": "q-2", "phase": "questionnaire", "unit": None, "question": "B?",
         "status": "open", "human_answer": None, "default_assumption": "db"},
    ])
    assert "deferred (over slot limit) - default applied: da" in log
    assert "open - default assumption applied: db" in log
```

Append to `tests/test_phase_executor.py`:

```python
@pytest.mark.asyncio
async def test_prompts_suppress_deferred_questions_too(store, tmp_path):
    # A deferred question is terminal but must stay visible to agents, or a
    # revision re-raises it as new (and post-gate it would sit open forever).
    qid = store.raise_question("discovery", None, "Deferred one?", "ctx", "d",
                               importance="low", impact="none")
    store.defer_questions([qid])
    spec = _spec()
    agent = FakeAgent(["# R"], side_effect=lambda p: store.update_memory("discovery", None, "k", "v"))
    await _executor(store, tmp_path, spec, agent).on_start("start", FakeCtx())
    assert qid in agent.prompts[0]           # suppression list includes deferred
```

- [ ] **Step 2: Run to verify they fail**

Run: `poetry run pytest tests/test_main.py tests/test_report.py tests/test_phase_executor.py -v`
Expected: FAIL — the `already_resumed` deferred case (returns `True` today), the `_prompt_human` construction (`TypeError` until the display lines land is NOT the failure — the constructor already takes 8 args from Task 2; the missing `Importance:` output is), the missing deferred wording in the log, and the deferred id absent from the phase prompt.

- [ ] **Step 3: Thread the flag through `pipeline.py`**

Change `build_workflow`'s signature to:

```python
def build_workflow(store: ArtifactStore, base_dir: Path,
                   scratchpad_path: Path = SCRATCHPAD_PATH,
                   checkpoint_storage: CheckpointStorage | None = None,
                   max_questions: int = 3):
```

add to its docstring Args:

```
        max_questions: Review-gate slot budget (``0`` = defer everything and
            never pause); threaded to :class:`ReviewExecutor`.
```

and change the gate construction to:

```python
    review = ReviewExecutor(store, revision_order=[(s.name, s.unit) for s in specs],
                            max_questions=max_questions)
```

- [ ] **Step 4: CLI flag, tightened guard, display lines in `main.py`**

Extend the artifacts import to `from .artifacts import REPOS, ArtifactStore, QuestionStatus`.

In `_amain`'s parser block, after the `--data` argument add:

```python
    parser.add_argument("--max-questions", type=int, default=3, metavar="N",
                        help="review-gate slot budget: open questions compete and only "
                             "the top N are presented, the rest proceed on their defaults "
                             "(default: %(default)s; 0 = never pause, fully autonomous)")
```

and directly after `args = parser.parse_args()`:

```python
    if args.max_questions < 0:
        parser.error("--max-questions must be >= 0")
```

Pass the flag in BOTH `build_workflow` calls (resume and fresh):

```python
        workflow = build_workflow(store, args.data, checkpoint_storage=storage,
                                  max_questions=args.max_questions)
```

(the resume-path gate never re-runs, so the value is inert there — passed for signature honesty).

Replace `already_resumed`'s return line and the tail of its docstring:

```python
    Returns:
        True when any entry carries a human verdict (answered or declined).
        ``deferred`` entries are written by the GATE before any resume exists
        - losing the slot competition is not a verdict and must not read as
        "already resumed".
    """
    verdicts = (QuestionStatus.ANSWERED, QuestionStatus.DECLINED)
    return any(e["status"] in verdicts for e in ledger)
```

In `_prompt_human`, after the `print(f"\n[{q.question_id}] ...")` line add:

```python
    print(f"      Importance: {q.importance}")
    print(f"      Impact if answered: {q.impact}")
```

In `_write_pause_files`' per-question loop, after the `[{q['id']}]` print add:

```python
        print(f"      Importance: {q['importance']}")
        print(f"      Impact if answered: {q['impact']}")
```

- [ ] **Step 5: Deferred branch in `report.py`**

In `render_adjudication_log`, extend the module's artifacts import (`from .artifacts import ArtifactStore, QuestionStatus`) and replace the status chain with:

```python
        if e["status"] == QuestionStatus.ANSWERED:
            resolution = f"answered: {e['human_answer']}"
        elif e["status"] == QuestionStatus.DECLINED:
            resolution = f"declined - default applied: {e['default_assumption']}"
        elif e["status"] == QuestionStatus.DEFERRED:
            resolution = f"deferred (over slot limit) - default applied: {e['default_assumption']}"
        else:
            resolution = f"open - default assumption applied: {e['default_assumption']}"
```

and add to its docstring's per-status sentence: "``deferred`` shows the applied default explicitly marked as a slot-limit outcome."

- [ ] **Step 6: Dup suppression in `phases.py`**

In `PhaseExecutor._run_initial`, change `self._store.open_questions(),` (the `build_initial_prompt` argument) to `self._store.unresolved_questions(),` — and identically in `on_revision` for `build_revision_prompt`. Add one comment above the first:

```python
            # unresolved = open + deferred: a deferred question must stay in the
            # suppression list or a revising agent re-raises it as new.
```

- [ ] **Step 7: Run the whole suite and commit**

Run: `poetry run pytest`
Expected: all green.

```bash
git add src/hotl_demo/pipeline.py src/hotl_demo/main.py src/hotl_demo/report.py src/hotl_demo/phases.py tests/
git commit -m "feat: --max-questions CLI; deferred-aware resume guard, adjudication log, suppression

--max-questions (default 3, 0 = fully autonomous) threads main -> build_workflow
-> ReviewExecutor. already_resumed now demands a VERDICT (answered/declined):
the gate defers losers before any resume exists, and deferred must not read as
'already resumed'. The adjudication log gets an explicit deferred branch,
phase prompts suppress deferred questions (open + deferred), and the review
banner/pause sheet show each question's importance and impact."
```

---

### Task 4: Documentation — README, CLAUDE.md

**Files:**
- Modify: `README.md` (ledger entry, lifecycle diagram, gate rules, run section)
- Modify: `CLAUDE.md` (commands, architecture bullet, gotcha)

Both are lint-gated; README uses ASCII hyphens (` - `), no em-dashes.

- [ ] **Step 1: README — schema and lifecycle**

In the `### The ledger entry` JSON example, add after the `"context"` line:

```json
  "impact": "If the producer is also migrating, the feed moves to object storage in one step; if not, an NFS bridge must be budgeted.",
  "importance": "medium",
```

and change its `"status"` line to `"status": "open | answered | declined | deferred",`. After the paragraph below the example (ending "that pair is exactly what the gate later re-runs."), add:

```markdown
`importance` (high / medium / low) and `impact` - how the human's answer
would change the migration decision - are agent-declared, tool-validated,
and exist for the slot competition below.
```

In the `### Lifecycle of a question` mermaid block, add one edge after the existing `O -->` lines:

```
    O -->|"loses the slot competition"| DEF["status: deferred<br>default applied, terminal"]
```

- [ ] **Step 2: README — the competition in the gate rules and run section**

In `## The review gate`, renumber and insert as new rule 2 (after "It runs exactly once."):

```markdown
2. **Slots are scarce - questions compete.** The gate presents at most
   `--max-questions` (default 3). When more are open, one LLM ranking call
   orders them by expected swing on the final report - judging each
   question's substance and impact statement, with the declared importance
   as one input; raise order carries no signal. Losers are marked
   `deferred`: their defaults stand and the adjudication log says so. The
   ranker is fenced (validated output, one retry, deterministic fallback),
   and `--max-questions 0` defers everything - the fully autonomous run.
```

In `## Run the demo`, after the "Type an answer..." paragraph, add:

```markdown
Pass `--max-questions N` to change the slot budget (default 3; `0` never
pauses and runs entirely on defaults). Paused runs created before this
feature cannot be resumed - the checkpointed question schema changed - so
start those assessments fresh.
```

- [ ] **Step 3: CLAUDE.md — commands, architecture, gotcha**

Commands block, after the `--resume` line:

```bash
poetry run demo --max-questions 5              # review-gate slot budget (0 = never pause)
```

Architecture review-gate bullet, append:

```markdown
  Open questions COMPETE for `--max-questions` slots (default 3): a tool-less
  ranker agent orders them by expected swing on the final report (raise order
  carries no signal); losers are marked `deferred` - terminal, default
  applied - BEFORE the pause, so checkpoints and `review.jsonl` reflect the
  competition. Ranker output is fenced: `validate_ranking` -> one retry ->
  deterministic `(importance, id)` fallback.
```

Rules and gotchas, append:

```markdown
- `deferred` is not a verdict: `already_resumed` must only count
  answered/declined, because the gate writes deferrals BEFORE any resume
  exists. `Importance`/`QuestionStatus`/`Phase` are `(str, Enum)` in
  `artifacts.py` - plain strings on disk; compare with enum members, and
  keep `enum.StrEnum` out (3.11+, floor is 3.10).
```

- [ ] **Step 4: Lint, full suite, commit**

Run: `poetry run pytest tests/test_markdown_lint.py -v && poetry run pytest`
Expected: lint clean (README, CLAUDE.md, prompts including `rank_questions.md`), all green.

```bash
git add README.md CLAUDE.md
git commit -m "docs: bounded ledger - slot competition, importance/impact, deferred lifecycle"
```

---

## Verification

- [ ] `poetry run pytest` — full suite green, LLM-free.
- [ ] Live default run (needs Ollama): `poetry run demo` — expect `== REVIEW - presenting 3 of N open questions (M deferred to defaults) ==` when N>3, importance/impact lines under each prompt, and deferred rows in the final adjudication log.
- [ ] `poetry run demo --max-questions 0` — never pauses; report lands with every question `deferred (over slot limit)`.
- [ ] `poetry run demo --pause` then `--resume` — `review.jsonl` seeded with exactly the presented ids; resume revises only answered targets; a second resume still refuses.
- [ ] `OLLAMA_E2E=1 poetry run pytest -m ollama -s` — the scripted loop answers whatever the real ranker presents.
