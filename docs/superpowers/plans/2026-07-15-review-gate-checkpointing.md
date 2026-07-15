# Review-Gate Checkpointing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `demo --pause` checkpoints and exits at the review gate; `demo --resume <run_dir>` restores it days later, applies answers from `review.jsonl`, and drives revisions to the final report — re-running zero phases.

**Architecture:** The framework checkpoints every superstep into a `FileCheckpointStorage` under the run directory. The gate checkpoint is selected *semantically* (the one with `pending_request_info_events`), never positionally. `ReviewExecutor` loses its in-memory `_awaiting` counter (which a resume silently corrupts into 5x re-runs) in favor of a ledger-derived guard. Human answers live in `review.jsonl` as `{"id", "answer"}` lines only — the questions stay in `ledger.jsonl`.

**Tech Stack:** Python ≥3.10, Poetry, pytest (`asyncio_mode = "auto"`, explicit `@pytest.mark.asyncio` by convention), `agent-framework ~=1.11` (all claims verified against 1.11.0 by an executable spike).

**Spec:** `docs/superpowers/specs/2026-07-15-review-gate-checkpointing-design.md` — authoritative.

## Global Constraints

- **`review.py` must NOT use `from __future__ import annotations`** — `@response_handler` inspects annotations at runtime; string annotations break it.
- **Never create `tests/__init__.py`** — pytest imports tests as top-level modules; `from conftest import ...` depends on it.
- **Tests are LLM-free by default** (`addopts = "-m 'not ollama'"`). No test in this plan may contact Ollama.
- **Every ledger/memory mutation goes through `ArtifactStore`.** `review.jsonl` is written by `main.py` (it is the human's file, not a store artifact) but answers are applied only via `store.resolve_question`.
- **Loud failures**: a malformed `review.jsonl` line must abort with its line number, never degrade to "decline".
- **The interactive default path must not change behavior**: `demo` with no flags builds the workflow with `checkpoint_storage=None` — byte-for-byte today's flow. The `OLLAMA_E2E` test must keep passing untouched.
- **Verified framework facts** (do not re-litigate; all empirical against 1.11.0):
  - `WorkflowBuilder(name=..., start_executor=..., checkpoint_storage=...)` — constructor args. **`with_checkpointing()` does not exist** despite the docs.
  - `FileCheckpointStorage(path, *, allowed_checkpoint_types=[...])`. Checkpoints are **pickled**; a type missing from the allowlist makes every load fail, and `list_checkpoints` **swallows the failure and returns `[]`** — indistinguishable from "no checkpoints".
  - `storage.list_checkpoints(workflow_name=...)` is **async** and **glob-ordered, not chronological**.
  - `WorkflowCheckpoint(workflow_name, graph_signature_hash, ...)` is a plain dataclass with `checkpoint_id`, `iteration_count`, `pending_request_info_events` fields.
  - `workflow.run(checkpoint_id=..., stream=True)` re-emits pending `request_info` events with **stable request ids** and re-runs **zero** phases; a following `workflow.run(stream=True, responses=...)` continues to completion.
  - `ArtifactStore(run_dir, repos)` over an existing directory **preserves** `memory.json` and `ledger.jsonl` (seed is guarded by `exists()`).

---

### Task 1: Ledger-derived guard in `ReviewExecutor` (delete `_awaiting`)

**Files:**
- Modify: `src/hotl_demo/review.py`
- Test: `tests/test_review.py`

**Interfaces:**
- Consumes: `ArtifactStore.open_questions() -> list[dict]` (entries with `status == "open"`, ledger order); `ArtifactStore.resolve_question(id, status, answer)`.
- Produces: `ReviewExecutor` whose `on_answer` dispatches revisions only when `store.open_questions()` is empty. No signature changes; `_awaiting` attribute removed. Task 4's integration test relies on a fresh `ReviewExecutor` instance behaving correctly mid-adjudication.

**Why:** `_awaiting` is a plain Python attribute; the framework only checkpoints state registered via `ctx.set_executor_state()`, which this codebase never calls. On resume the executor is a fresh object with `_awaiting == 0`, so **every** answer looks like the last: the spike measured `discovery` revised 5x and `final_report` written 5x, silently. The ledger is file-backed and already knows how many verdicts are outstanding.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_review.py`:

```python
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
                              q["context"], q["default_assumption"])
        for q in store.open_questions()
    ]
    for r in reqs[:-1]:
        await review.on_answer(r, f"answer to {r.question_id}", ctx)
        assert ctx.sent == []                          # verdicts still outstanding
    await review.on_answer(reqs[-1], "final answer", ctx)
    assert len(ctx.sent) == 1                          # exactly one dispatch
    assert isinstance(ctx.sent[0], RevisionTrigger)
    assert (ctx.sent[0].phase, ctx.sent[0].unit) == ("discovery", None)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `poetry run pytest tests/test_review.py::test_fresh_executor_acts_only_on_last_answer -v`
Expected: FAIL at the first `assert ctx.sent == []` — the fresh executor's `_awaiting` is 0, so the first answer decrements it to -1 and dispatches immediately.

- [ ] **Step 3: Replace the counter with the ledger guard**

In `src/hotl_demo/review.py`:

1. In `__init__`, delete the line `self._awaiting = 0`.
2. In `on_questionnaire_done`, delete the line `self._awaiting = len(open_qs)`.
3. In `on_answer`, replace:

```python
        self._awaiting -= 1
        if self._awaiting > 0:
            return  # more verdicts still inbound; act only on the last one
```

with:

```python
        if self._store.open_questions():
            return  # verdicts still outstanding - ledger-derived, so a resumed
            # run (fresh executor instance) behaves identically to this one
```

4. In the `ReviewExecutor` class docstring, replace the step-2 line:

```
    2. ``on_answer`` (once per question) - record the verdict; when the last
       one lands, build the ordered re-run queue and dispatch its head.
```

with:

```
    2. ``on_answer`` (once per question) - record the verdict; when the ledger
       has no open questions left (all verdicts in - a FILE-backed check, so a
       checkpoint-resumed gate, which is a fresh instance, behaves the same),
       build the ordered re-run queue and dispatch its head.
```

5. Update the `__init__` docstring's Args intro line from "Remember the store and the canonical re-run ordering." to:

```
        """Remember the store and the canonical re-run ordering.

        Deliberately no adjudication counters here: gate progress must be
        derived from the ledger so that a checkpoint resume (fresh instance)
        cannot diverge from a live run. See the checkpointing spec.
```

(keep the existing Args block below it).

- [ ] **Step 4: Run the full review suite**

Run: `poetry run pytest tests/test_review.py -v`
Expected: PASS — all 7 pre-existing tests plus the new one. The pre-existing tests exercise the same last-answer semantics through the public handlers, so they prove the swap preserved behavior.

- [ ] **Step 5: Run the whole suite and commit**

Run: `poetry run pytest`
Expected: all green (~85 passed, 1 deselected).

```bash
git add src/hotl_demo/review.py tests/test_review.py
git commit -m "fix: derive review-gate progress from the ledger, not an in-memory counter

A checkpoint-resumed gate is a fresh executor instance; the framework only
persists state registered via ctx.set_executor_state, so _awaiting restarted
at 0 and every answer looked like the last - measured: discovery revised 5x,
final_report written 5x, no exception. open_questions() is file-backed and
already the source of truth."
```

---

### Task 2: Checkpoint wiring in `pipeline.py`

**Files:**
- Modify: `src/hotl_demo/pipeline.py`
- Test: `tests/test_checkpoint.py` (new)

**Interfaces:**
- Consumes: message dataclasses from `phases.py`/`review.py`; `WorkflowCheckpoint`, `CheckpointStorage` from `agent_framework`.
- Produces (Tasks 3–4 rely on these exact names):
  - `WORKFLOW_NAME: str = "hotl-migration-readiness"`
  - `ALLOWED_CHECKPOINT_TYPES: list[str]` — `"module:qualname"` strings derived from the message classes
  - `gate_checkpoint(checkpoints: list[WorkflowCheckpoint]) -> WorkflowCheckpoint | None`
  - `build_workflow(store, base_dir, scratchpad_path=SCRATCHPAD_PATH, checkpoint_storage=None)`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_checkpoint.py`:

```python
"""Checkpointing: gate-checkpoint selection, allowlist completeness, pause/resume."""
import dataclasses

import pytest

from agent_framework import WorkflowCheckpoint

from hotl_demo.pipeline import ALLOWED_CHECKPOINT_TYPES, WORKFLOW_NAME, gate_checkpoint


def _cp(iteration_count, pending_ids=()):
    """Build a real WorkflowCheckpoint with only the fields selection reads."""
    return WorkflowCheckpoint(
        workflow_name=WORKFLOW_NAME,
        graph_signature_hash="test-hash",
        iteration_count=iteration_count,
        pending_request_info_events={rid: object() for rid in pending_ids},
    )


def test_gate_checkpoint_picks_pending_with_highest_iteration():
    # list_checkpoints is glob-ordered (UUID filenames), so feed a shuffled
    # list: selection must be semantic, never positional.
    cps = [_cp(9), _cp(4, pending_ids=("r1",)), _cp(2), _cp(6, pending_ids=("r1", "r2"))]
    assert gate_checkpoint(cps) is cps[3]      # pending beats higher bare iteration
    assert gate_checkpoint(list(reversed(cps))) is cps[3]


def test_gate_checkpoint_none_when_nothing_pending():
    assert gate_checkpoint([]) is None
    assert gate_checkpoint([_cp(1), _cp(7)]) is None


def test_allowlist_covers_every_message_dataclass():
    # Trap: checkpoints are pickled behind this allowlist, and a missing type
    # makes list_checkpoints silently return [] - resume just "stops working".
    # Force a conscious decision whenever a dataclass is added.
    from hotl_demo import phases, review
    found = set()
    for mod in (phases, review):
        for name in dir(mod):
            obj = getattr(mod, name)
            if (isinstance(obj, type) and dataclasses.is_dataclass(obj)
                    and obj.__module__ == mod.__name__):
                found.add(f"{obj.__module__}:{obj.__qualname__}")
    non_messages = {"hotl_demo.phases:PhaseSpec"}   # executor config, never routed
    assert found - non_messages == set(ALLOWED_CHECKPOINT_TYPES)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `poetry run pytest tests/test_checkpoint.py -v`
Expected: FAIL — `ImportError: cannot import name 'ALLOWED_CHECKPOINT_TYPES' from 'hotl_demo.pipeline'`.

- [ ] **Step 3: Add the wiring to `pipeline.py`**

Extend the `agent_framework` import line and the `.review` import:

```python
from agent_framework import CheckpointStorage, Executor, WorkflowBuilder, WorkflowCheckpoint, WorkflowContext, handler
```

and below the existing imports add (note: `LedgerQuestionRequest` comes from `.review`; importing it here is cycle-free because `review.py` imports only from `phases.py`):

```python
from .review import LedgerQuestionRequest, ReviewExecutor
```

(replace the existing `from .review import ReviewExecutor` line). Then, after the imports:

```python
WORKFLOW_NAME = "hotl-migration-readiness"

# Every type that crosses the graph or sits in a pending request. Checkpoints
# are PICKLED behind this allowlist; a type missing here does not raise on
# load - list_checkpoints logs, skips the file, and returns [], which is
# indistinguishable from "no checkpoints exist". Derived from the classes so
# the module:qualname strings can never drift; test_checkpoint.py asserts
# completeness against every dataclass in phases.py/review.py.
_MESSAGE_TYPES = (PhaseDone, AnalysisDone, RevisionDone, RevisionTrigger,
                  ReportTrigger, LedgerQuestionRequest)
ALLOWED_CHECKPOINT_TYPES = [f"{t.__module__}:{t.__qualname__}" for t in _MESSAGE_TYPES]


def gate_checkpoint(checkpoints: list[WorkflowCheckpoint]) -> WorkflowCheckpoint | None:
    """Select the review-gate pause point from a run's checkpoints.

    The gate checkpoint is BY DEFINITION the one idle with pending
    ``request_info`` events; among those, the latest superstep wins.
    Never select positionally: ``list_checkpoints`` globs UUID filenames,
    so its order is meaningless - resuming from an arbitrary "latest" was
    measured to skip the human entirely (the file-backed review_completed
    latch told the re-entered gate "already reviewed").

    Args:
        checkpoints: Whatever ``storage.list_checkpoints`` returned.

    Returns:
        The gate checkpoint, or ``None`` when the run never paused (or every
        checkpoint file failed to decode - see ALLOWED_CHECKPOINT_TYPES).

    Example:
        >>> gate_checkpoint([]) is None
        True
    """
    pending = [c for c in checkpoints if c.pending_request_info_events]
    return max(pending, key=lambda c: c.iteration_count) if pending else None
```

- [ ] **Step 4: Thread `checkpoint_storage` through `build_workflow`**

Change the signature and the builder call:

```python
def build_workflow(store: ArtifactStore, base_dir: Path,
                   scratchpad_path: Path = SCRATCHPAD_PATH,
                   checkpoint_storage: CheckpointStorage | None = None):
```

Add to the docstring's Args block:

```
        checkpoint_storage: When provided, the framework checkpoints every
            superstep into it (the --pause/--resume flows). ``None`` - the
            default and the interactive path - changes nothing.
```

and replace the builder line:

```python
    builder = WorkflowBuilder(start_executor=discovery)
```

with:

```python
    builder = WorkflowBuilder(name=WORKFLOW_NAME, start_executor=discovery,
                              checkpoint_storage=checkpoint_storage)
```

(`name` is required for `list_checkpoints(workflow_name=...)` to find the run's checkpoints; harmless when checkpointing is off.)

- [ ] **Step 5: Run the tests**

Run: `poetry run pytest tests/test_checkpoint.py tests/test_pipeline.py -v`
Expected: PASS — the 3 new tests plus all pre-existing pipeline tests (the new kwarg defaults to `None`, and naming the builder does not change the graph).

- [ ] **Step 6: Run the whole suite and commit**

Run: `poetry run pytest`
Expected: all green.

```bash
git add src/hotl_demo/pipeline.py tests/test_checkpoint.py
git commit -m "feat: checkpoint wiring - workflow name, pickle allowlist, semantic gate selection

gate_checkpoint() picks the checkpoint idle with pending request_info events;
list_checkpoints is glob-ordered so positional selection is meaningless (and
was measured to skip the human entirely). ALLOWED_CHECKPOINT_TYPES is derived
from the message classes and completeness-tested, because a missing type makes
checkpoint loads fail SILENTLY (list_checkpoints returns [])."
```

---

### Task 3: CLI — `--pause`, `--resume`, and `review.jsonl`

**Files:**
- Modify: `src/hotl_demo/main.py`
- Test: `tests/test_main.py`

**Interfaces:**
- Consumes (from Task 2): `WORKFLOW_NAME`, `ALLOWED_CHECKPOINT_TYPES`, `gate_checkpoint`, `build_workflow(..., checkpoint_storage=)`.
- Produces: `render_review_lines(open_questions: list[dict]) -> str`, `parse_review_answers(text: str) -> dict[str, str]` (Task 4 round-trips them), constants `REVIEW_FILENAME = "review.jsonl"` and `CHECKPOINT_DIRNAME = "checkpoints"`.

- [ ] **Step 1: Write the failing tests**

Replace the import block at the top of `tests/test_main.py` with:

```python
"""CLI runner helpers: preflight matching, host normalization, review.jsonl."""
import json

import pytest

from hotl_demo.main import (
    _prompt_human,
    model_present,
    normalize_host,
    parse_review_answers,
    render_review_lines,
)
from hotl_demo.review import LedgerQuestionRequest
```

and append:

```python
def _q(qid):
    return {"id": qid, "phase": "discovery", "unit": None, "question": "Q?",
            "context": "c", "default_assumption": "d", "status": "open",
            "human_answer": None, "asked_at": "t"}


def test_render_review_lines_seeds_id_and_answer_only():
    # The questions stay in ledger.jsonl (agent-curated, read-only); the
    # answer sheet carries ONLY the human's input, joined on id.
    lines = [json.loads(l) for l in render_review_lines([_q("q-1"), _q("q-2")]).splitlines()]
    assert lines == [{"id": "q-1", "answer": ""}, {"id": "q-2", "answer": ""}]


def test_parse_review_answers_round_trip_and_blank_lines():
    text = '{"id": "q-1", "answer": "yes, in scope"}\n\n{"id": "q-2", "answer": ""}\n'
    assert parse_review_answers(text) == {"q-1": "yes, in scope", "q-2": ""}


@pytest.mark.parametrize("bad, hint", [
    ('{"id": "q-1", "answer": "ok"}\nnot json\n', "line 2"),
    ('["q-1", "ok"]\n', "line 1"),                          # not an object
    ('{"answer": "ok"}\n', "line 1"),                       # missing id
    ('{"id": "q-1", "answer": null}\n', "line 1"),          # non-string answer
    ('{"id": "q-1", "answer": "a"}\n{"id": "q-1", "answer": "b"}\n', "line 2"),
])
def test_parse_review_answers_is_loud_on_malformed_input(bad, hint):
    # A parse error must NEVER degrade into "decline" - that would silently
    # discard the human's gathered answers and ship a defaults-only report.
    with pytest.raises(ValueError, match=hint):
        parse_review_answers(bad)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `poetry run pytest tests/test_main.py -v`
Expected: FAIL — `ImportError: cannot import name 'parse_review_answers' from 'hotl_demo.main'`.

- [ ] **Step 3: Add constants and the two pure helpers to `main.py`**

Below `DEFAULT_MODEL = "gemma4:31b"`:

```python
REVIEW_FILENAME = "review.jsonl"
CHECKPOINT_DIRNAME = "checkpoints"
```

After `_prompt_human`, add:

```python
def render_review_lines(open_questions: list[dict]) -> str:
    """Seed the answer sheet: one ``{"id", "answer": ""}`` line per open question.

    Only the human's input lives in review.jsonl - the question text stays in
    ledger.jsonl (agent-curated, read-only); a frontend joins on ``id``.
    Unedited lines decline naturally at resume time.

    Args:
        open_questions: Ledger entries with ``status == "open"``, in ledger
            order (as returned by ``ArtifactStore.open_questions``).

    Returns:
        JSONL text, one seeded record per question.

    Example:
        >>> render_review_lines([{"id": "q-1"}])
        '{"id": "q-1", "answer": ""}\\n'
    """
    return "".join(json.dumps({"id": q["id"], "answer": ""}) + "\n" for q in open_questions)


def parse_review_answers(text: str) -> dict[str, str]:
    """Parse review.jsonl into ``{question_id: answer}``.

    Loud on ANY malformed line: a parse error must never degrade into
    "decline", which would silently discard the human's gathered answers and
    proceed on defaults. Blank lines are allowed (editors add them).

    Args:
        text: Full review.jsonl content.

    Returns:
        Mapping of question id to raw answer text (``""`` = decline).

    Raises:
        ValueError: Malformed JSON, non-object line, missing/duplicate id, or
            a non-string answer - always naming the offending line number.

    Example:
        >>> parse_review_answers('{"id": "q-1", "answer": "yes"}\\n')
        {'q-1': 'yes'}
    """
    answers: dict[str, str] = {}
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{REVIEW_FILENAME} line {lineno}: invalid JSON ({exc})")
        if not isinstance(record, dict) or not isinstance(record.get("id"), str):
            raise ValueError(
                f"{REVIEW_FILENAME} line {lineno}: expected an object with a string 'id'")
        if not isinstance(record.get("answer"), str):
            raise ValueError(
                f'{REVIEW_FILENAME} line {lineno}: "answer" must be a string (use "" to decline)')
        if record["id"] in answers:
            raise ValueError(f"{REVIEW_FILENAME} line {lineno}: duplicate id {record['id']!r}")
        answers[record["id"]] = record["answer"]
    return answers


def map_answers(pending: dict[str, "LedgerQuestionRequest"],
                answers: dict[str, str]) -> tuple[dict[str, str], list[str]]:
    """Join the answer sheet onto the gate's pending requests.

    Missing id = decline (""), consistent with "empty answer = decline".

    Args:
        pending: ``request_id -> LedgerQuestionRequest`` collected from the
            resumed workflow's re-emitted events.
        answers: ``question_id -> answer`` from review.jsonl.

    Returns:
        ``(responses, unknown_ids)`` - responses keyed by request id, ready
        for ``workflow.run(responses=...)``, plus any sheet ids that match no
        pending question (sorted; the caller warns about them).
    """
    responses = {rid: answers.get(q.question_id, "") for rid, q in pending.items()}
    unknown = sorted(set(answers) - {q.question_id for q in pending.values()})
    return responses, unknown
```

Add these tests to the Step 1 block in `tests/test_main.py` as well (they fail with the same ImportError; add `map_answers` to the import):

```python
def test_map_answers_missing_id_declines_and_unknown_ids_surface():
    pending = {"r1": LedgerQuestionRequest("q-1", "discovery", None, "Q?", "c", "d"),
               "r2": LedgerQuestionRequest("q-2", "discovery", None, "Q?", "c", "d")}
    responses, unknown = map_answers(pending, {"q-1": "yes", "q-99": "ghost"})
    assert responses == {"r1": "yes", "r2": ""}   # q-2 unanswered -> decline
    assert unknown == ["q-99"]                    # warned by the caller, ignored
```

- [ ] **Step 4: Run the new tests**

Run: `poetry run pytest tests/test_main.py -v`
Expected: PASS.

- [ ] **Step 5: Restructure `_amain` into the three flows**

Replace `main.py`'s module docstring with:

```python
"""CLI runner: preflight, run the workflow, adjudicate the review gate.

Three flows (see the checkpointing spec):

* default        - interactive: stream events, prompt on stdin at the gate,
                   resume with ``run(responses={...})`` in the same process.
* ``--pause``    - checkpointing on: at the gate, seed ``review.jsonl`` with
                   one ``{"id", "answer"}`` line per open question and EXIT;
                   the human answers at leisure (question text stays in
                   ``ledger.jsonl``).
* ``--resume``   - restore the gate checkpoint of a --pause run, apply the
                   answers from ``review.jsonl``, drive revisions to the
                   final report. Re-runs zero phases.
"""
```

Replace the whole `_amain` function with:

```python
async def _amain() -> None:
    """Parse args, preflight, then dispatch to the right flow."""
    parser = argparse.ArgumentParser(description="HOTL cloud migration readiness demo")
    parser.add_argument("--model", default=os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL),
                        help="Ollama model tag (default: %(default)s)")
    parser.add_argument("--data", type=Path, default=Path("sample_data"),
                        help="sample data directory")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--pause", action="store_true",
                      help="checkpoint and exit at the review gate instead of prompting; "
                           "fill in <run_dir>/review.jsonl, then rerun with --resume. "
                           "A run that raises no questions never pauses.")
    mode.add_argument("--resume", type=Path, metavar="RUN_DIR", default=None,
                      help="resume a --pause run from its review-gate checkpoint, "
                           "applying the answers in RUN_DIR/review.jsonl. "
                           "Pass the same --model the run was paused with.")
    args = parser.parse_args()
    os.environ["OLLAMA_MODEL"] = args.model  # OllamaChatClient reads this
    base_url = normalize_host(os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    preflight(base_url, args.model)
    ensure_scratchpad(SCRATCHPAD_PATH)

    # import here so --help and preflight failures never touch the framework
    from agent_framework import FileCheckpointStorage, WorkflowCheckpointException
    from .pipeline import (
        ALLOWED_CHECKPOINT_TYPES,
        WORKFLOW_NAME,
        build_workflow,
        gate_checkpoint,
    )

    if args.resume is not None:
        run_dir = args.resume
        review_path = run_dir / REVIEW_FILENAME
        if not review_path.exists():
            raise SystemExit(
                f"{review_path} not found - only runs started with --pause can be resumed.")
        answers = parse_review_answers(review_path.read_text(encoding="utf-8"))
        store = ArtifactStore(run_dir, REPOS)  # reopening preserves memory + ledger
        open_ids = {q["id"] for q in store.open_questions()}
        if not (set(answers) & open_ids):
            # Resume is NOT idempotent (a second pass would re-run every
            # revision), so refuse loudly - distinguishing the two causes.
            report = run_dir / "final_report.md"
            if report.exists():
                raise SystemExit(f"Already resumed - final report at {report}")
            raise SystemExit(
                "Answers were already applied but no final report exists - the previous "
                "resume likely crashed mid-revision; start a fresh run "
                "(mid-revision recovery is out of scope).")
        storage = FileCheckpointStorage(run_dir / CHECKPOINT_DIRNAME,
                                        allowed_checkpoint_types=ALLOWED_CHECKPOINT_TYPES)
        workflow = build_workflow(store, args.data, checkpoint_storage=storage)
        gate = gate_checkpoint(await storage.list_checkpoints(workflow_name=WORKFLOW_NAME))
        if gate is None:
            raise SystemExit(
                f"No review-gate checkpoint under {run_dir / CHECKPOINT_DIRNAME}.\n"
                "Either this run was not started with --pause, or a message type is "
                "missing from ALLOWED_CHECKPOINT_TYPES - unreadable checkpoint files "
                "are SILENTLY skipped, so check for decode warnings above.")
        try:
            await _drive(workflow, store, checkpoint_id=gate.checkpoint_id, answers=answers)
        except WorkflowCheckpointException as exc:
            # e.g. graph_signature_hash mismatch: prompts/repos edited since pause
            raise SystemExit(
                f"The pipeline changed since this run was paused ({exc}).\n"
                "A checkpoint only fits the graph that wrote it - start a fresh run.")
        return

    run_dir = Path("output") / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    store = ArtifactStore(run_dir, REPOS)
    storage = None
    if args.pause:
        storage = FileCheckpointStorage(run_dir / CHECKPOINT_DIRNAME,
                                        allowed_checkpoint_types=ALLOWED_CHECKPOINT_TYPES)
    # storage=None on the default path: the interactive flow is byte-for-byte
    # unchanged; checkpointing risk stays opt-in.
    workflow = build_workflow(store, args.data, checkpoint_storage=storage)
    await _drive(workflow, store, message="start", pause=args.pause)
```

Then add `_drive` and `_write_pause_files` after `_amain`:

```python
async def _drive(workflow, store: ArtifactStore, *, message: str | None = None,
                 checkpoint_id: str | None = None,
                 answers: dict[str, str] | None = None, pause: bool = False) -> None:
    """Shared run loop for all three flows.

    First iteration: fresh start (``message``) or checkpoint restore
    (``checkpoint_id``). Later iterations resume the SAME workflow instance
    with the collected responses - the framework's pause/resume contract.
    Verdict source: ``answers`` (--resume; missing id = decline), stdin
    (interactive), or nobody - ``pause`` seeds review.jsonl and exits.

    Args:
        workflow: The built workflow.
        store: The run's artifact store (pause files + final print).
        message: Start message for a fresh run.
        checkpoint_id: Gate checkpoint to restore instead of starting fresh.
        answers: ``{question_id: answer}`` from review.jsonl, or ``None``.
        pause: Seed the answer sheet and exit when the gate opens.
    """
    responses: dict[str, str] | None = None
    first = True
    while True:
        if not first:
            stream = workflow.run(stream=True, responses=responses)
        elif checkpoint_id is not None:
            stream = workflow.run(checkpoint_id=checkpoint_id, stream=True)
        else:
            stream = workflow.run(message, stream=True)
        first = False
        pending: dict[str, LedgerQuestionRequest] = {}
        async for event in stream:
            if event.type == "request_info" and isinstance(event.data, LedgerQuestionRequest):
                pending[event.request_id] = event.data
            elif event.type == "output":
                print(f"\nFinal report: {event.data}")
        if not pending:
            break  # the run finished without pausing: we are done
        if pause:
            _write_pause_files(store, len(pending))
            return
        if answers is not None:
            responses, unknown = map_answers(pending, answers)
            for qid in unknown:
                print(f"  warning: {REVIEW_FILENAME} id {qid!r} matches no pending "
                      "question - ignored")
        else:
            responses = {rid: _prompt_human(q) for rid, q in pending.items()}
    print(f"Run artifacts: {store.run_dir}")


def _write_pause_files(store: ArtifactStore, pending_count: int) -> None:
    """Seed the answer sheet and tell the human how to continue.

    Args:
        store: The run's store; the gate is idle, so ``open_questions()`` is
            exactly the presented set, in ledger order.
        pending_count: Number of pending gate requests (for the banner).
    """
    open_qs = store.open_questions()
    review_path = store.run_dir / REVIEW_FILENAME
    review_path.write_text(render_review_lines(open_qs), encoding="utf-8")
    print(f"\n== PAUSED at the review gate - {pending_count} open questions ==")
    for q in open_qs:
        where = f"{q['phase']}[{q['unit']}]" if q["unit"] else q["phase"]
        print(f"\n[{q['id']}] ({where}) {q['question']}")
        print(f"      Evidence: {q['context']}")
        print(f"      Default if declined: {q['default_assumption']}")
    print(f"\nFill in the answers in {review_path}")
    print('(one {"id", "answer"} JSON line per question; empty answer = decline; '
          f"question text lives in {store.run_dir / 'ledger.jsonl'})")
    print(f"Then: poetry run demo --resume {store.run_dir}")
```

- [ ] **Step 6: Run the whole suite and commit**

Run: `poetry run pytest`
Expected: all green — the interactive flow through `_drive(message="start", pause=False)` is behaviorally identical to the old loop.

```bash
git add src/hotl_demo/main.py tests/test_main.py
git commit -m "feat: demo --pause / --resume with a review.jsonl answer sheet

--pause checkpoints and exits at the gate, seeding one {id, answer} line per
open question (questions stay in ledger.jsonl - the sheet carries only human
input). --resume restores the gate checkpoint, applies the answers (missing or
empty = decline), and refuses loudly when the run was already resumed or the
checkpoint cannot be found. Malformed answer lines abort with a line number
rather than degrading to decline."
```

---

### Task 4: Full pause/resume regression test against the real graph

**Files:**
- Modify: `tests/conftest.py` (move `DriveAgent` + `DRIVE_TARGETS` here from `test_pipeline.py`)
- Modify: `tests/test_pipeline.py` (import them instead of defining them)
- Test: `tests/test_checkpoint.py` (append the integration test)

**Interfaces:**
- Consumes: everything from Tasks 1–3 (`gate_checkpoint`, `ALLOWED_CHECKPOINT_TYPES`, `WORKFLOW_NAME`, `build_workflow(checkpoint_storage=)`, `render_review_lines`, `parse_review_answers`, the ledger-derived gate guard).
- Produces: `conftest.DriveAgent(name, store, calls)` and `conftest.DRIVE_TARGETS` shared by `test_pipeline.py` and `test_checkpoint.py`.

**Why this test exists:** it is the executable form of the spike that found the 5x bug. If anyone reintroduces executor-local gate state, this test fails with five `discovery` revisions instead of one.

- [ ] **Step 1: Move the graph-drive double into `conftest.py`**

Cut `_TARGETS` and `_DriveAgent` out of `tests/test_pipeline.py` (lines defining them, currently between the `# -- LLM-free drive...` comment and the async graph test) and append to `tests/conftest.py`, renamed public:

```python
DRIVE_TARGETS = {
    "discovery": ("discovery", None),
    "analyze_oms-monolith": ("deep_analysis", "oms-monolith"),
    "analyze_oms-batch-recon": ("deep_analysis", "oms-batch-recon"),
    "enterprise_context": ("enterprise_context", None),
    "questionnaire": ("questionnaire", None),
}


class DriveAgent:
    """Stands in for every Agent when driving the REAL assembled graph LLM-free:
    raises one question per phase on the initial pass, one extra during
    discovery's revision, records call order. Shared by test_pipeline.py and
    test_checkpoint.py (the pause/resume cycle)."""

    def __init__(self, name, store, calls):
        self.name, self.store, self.calls = name, store, calls

    def create_session(self, *, session_id=None):
        """Mirror the real Agent's session API; PhaseExecutor mints one per cycle."""
        return f"{self.name}-session"

    async def run(self, prompt, *, session=None):
        if self.name == "final_report":
            self.calls.append((self.name, "report", prompt))
            return FakeAgentResult("FINAL-VERDICT")
        kind = "revision" if "## HUMAN ANSWERS" in prompt else "initial"
        self.calls.append((self.name, kind, prompt))
        phase, unit = DRIVE_TARGETS[self.name]
        if kind == "initial":
            self.store.update_memory(phase, unit, f"finding_{len(self.calls)}", "v")
            self.store.raise_question(phase, unit, f"Q from {self.name}?", "ctx", "default")
        elif self.name == "discovery":
            self.store.raise_question(phase, unit, "Raised during revision?", "ctx", "post-gate")
        return FakeAgentResult(f"REPORT[{self.name}][{kind}]")
```

(Note it now returns `FakeAgentResult` — already defined at the top of `conftest.py` — instead of the inline `type("R", ...)` trick.)

In `tests/test_pipeline.py`, change the conftest import to `from conftest import DRIVE_TARGETS, DriveAgent, FakeCtx`, delete the moved definitions, and update the two references: the factory in `test_workflow_graph_drive_gate_revisions_report` becomes `return DriveAgent(name, store, calls)` and `set(_TARGETS.values())` becomes `set(DRIVE_TARGETS.values())` (also `_TARGETS[self.name]` etc. disappear with the moved class).

- [ ] **Step 2: Verify the move broke nothing**

Run: `poetry run pytest tests/test_pipeline.py -v`
Expected: PASS — pure relocation.

- [ ] **Step 3: Write the integration test**

(It passes with Tasks 1–3 in place; Step 4 shows how to watch it catch the bug it guards.) Append to `tests/test_checkpoint.py` (add `import json` to its imports):

```python
from pathlib import Path

from agent_framework import FileCheckpointStorage

from conftest import DriveAgent

from hotl_demo.artifacts import REPOS, ArtifactStore
from hotl_demo.main import map_answers, parse_review_answers, render_review_lines
from hotl_demo.pipeline import build_workflow
from hotl_demo.review import LedgerQuestionRequest


@pytest.mark.asyncio
async def test_pause_resume_cycle_revises_each_target_exactly_once(tmp_path, monkeypatch):
    """The spike that found the 5x bug, as a permanent regression guard.

    Process 1 runs to the gate with checkpointing on. Process 2 is simulated
    with a FRESH store and a FRESH workflow over the same run dir: restore the
    gate checkpoint, answer everything via the review.jsonl round-trip, and
    assert the resumed gate dispatches ONE ordered revision queue - not one
    per answer.
    """
    from hotl_demo import phases, report
    from hotl_demo.pipeline import (ALLOWED_CHECKPOINT_TYPES, WORKFLOW_NAME,
                                    gate_checkpoint)

    monkeypatch.setenv("OLLAMA_MODEL", "test-model")
    run_dir, cp_dir, data = tmp_path / "run", tmp_path / "checkpoints", Path("sample_data")
    storage = FileCheckpointStorage(cp_dir, allowed_checkpoint_types=ALLOWED_CHECKPOINT_TYPES)

    def patch_agents(store, calls):
        def agent_factory(*_, name="", **__):
            return DriveAgent(name, store, calls)
        for mod in (phases, report):
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
    assert len(pending1) == 5

    # the pause artifact: seed the sheet, then play the human filling it in
    sheet = render_review_lines(store1.open_questions())
    answered = "".join(
        line.replace('"answer": ""', f'"answer": "answer to {json.loads(line)["id"]}"') + "\n"
        for line in sheet.splitlines()
    )
    answers = parse_review_answers(answered)
    assert set(answers) == {q.question_id for q in pending1.values()}
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
    assert calls2 == []                        # restore re-ran ZERO phases

    outputs = []
    responses, unknown = map_answers(pending2, answers)
    assert unknown == ["q-99"]                 # the ghost id surfaced, not applied
    async for ev in wf2.run(stream=True, responses=responses):
        assert ev.type != "request_info"       # review-once survives the restart
        if ev.type == "output":
            outputs.append(ev.data)

    # THE regression assertions: one ordered queue, not one queue per answer.
    # questionnaire was declined by omission, so exactly four targets revise.
    revisions = [name for name, kind, _ in calls2 if kind == "revision"]
    assert revisions == ["discovery", "analyze_oms-monolith", "analyze_oms-batch-recon",
                         "enterprise_context"]
    assert [name for name, kind, _ in calls2 if kind == "report"] == ["final_report"]
    assert outputs == [str(store2.run_dir / "final_report.md")]
    statuses = [e["status"] for e in store2.read_ledger()]
    assert statuses == ["answered"] * 4 + ["declined", "open"]  # open: raised mid-revision
```

Also add `import json` to `tests/test_checkpoint.py`'s imports.

- [ ] **Step 4: Run it**

Run: `poetry run pytest tests/test_checkpoint.py -v`
Expected: PASS with Tasks 1–3 in place. To watch it catch the bug it guards against, temporarily restore an `_awaiting`-style counter in `review.py` (init to 0, decrement per answer, dispatch when `<= 0`) and observe the assertion fail with overlapping queues — `discovery` revised once per answer instead of once. Then revert.

- [ ] **Step 5: Run the whole suite and commit**

Run: `poetry run pytest`
Expected: all green.

```bash
git add tests/conftest.py tests/test_pipeline.py tests/test_checkpoint.py
git commit -m "test: pause/resume cycle against the real graph - the 5x-bug regression guard

Fresh store + fresh workflow over the same run dir restore the gate
checkpoint, answer via the review.jsonl round-trip, and must dispatch exactly
one ordered revision queue with zero phase re-runs on restore. DriveAgent
moves to conftest so test_pipeline and test_checkpoint share it."
```

---

### Task 5: Documentation — README, CLAUDE.md

**Files:**
- Modify: `README.md` (Run the demo section)
- Modify: `CLAUDE.md` (Commands, Architecture review-gate bullet, Rules and gotchas)

Both files are in the markdown lint gate (`tests/test_markdown_lint.py`); README uses ASCII hyphens (` - `), no em-dashes.

- [ ] **Step 1: README — document the pause/resume flow**

In `README.md`, after the paragraph ending "or press ENTER to decline (the stated default stands)." (in `## Run the demo`), insert:

```markdown
### Pausing for days: `--pause` / `--resume`

Gathering real answers can take days, and holding a live process open for
that is the wrong shape. With `--pause` the run checkpoints at the gate and
exits:

```bash
poetry run demo --pause
# == PAUSED at the review gate - 5 open questions ==
# Fill in the answers in output/run_<ts>/review.jsonl
# Then: poetry run demo --resume output/run_<ts>
```

`review.jsonl` carries only the human's input - one `{"id", "answer"}` line
per question, seeded empty. The question text stays in `ledger.jsonl`
(agent-curated, read-only); a frontend joins the two on `id`. Leave an answer
empty to decline. Resuming restores the exact gate state and re-runs **zero**
phases - only the answered phases revise, then the report lands. Resume with
the same `--model` the run was paused with; a run that raises no questions
never pauses. A malformed answer line aborts loudly rather than silently
becoming a decline.
```

- [ ] **Step 2: CLAUDE.md — commands and architecture**

In the Commands code block, after the `poetry run demo` line, add:

```bash
poetry run demo --pause                        # checkpoint + exit at the review gate
poetry run demo --resume output/run_<ts>       # apply review.jsonl answers, finish the run
```

In the Architecture section's review-gate bullet, append:

```markdown
  With `--pause` the gate checkpoints and exits (answers land in
  `review.jsonl` - id + answer only); `--resume` restores via
  `gate_checkpoint()`, which selects the checkpoint holding pending
  request_info events - `list_checkpoints` is glob-ordered, never trust
  "latest".
```

- [ ] **Step 3: CLAUDE.md — record the two traps as gotchas**

Add to "Rules and gotchas":

```markdown
- Review-gate progress must stay LEDGER-derived. The framework only persists
  executor state registered via `ctx.set_executor_state` (we use none), so a
  resumed gate is a fresh instance: an in-memory counter made every answer
  look like the last (discovery revised 5x). `test_checkpoint.py`'s cycle
  test guards this.
- Checkpoints are pickled behind `ALLOWED_CHECKPOINT_TYPES` (pipeline.py). A
  type missing from it does NOT raise - `list_checkpoints` returns `[]`, so
  resume silently "finds no checkpoints". The allowlist is derived from the
  message classes and completeness-tested; keep new message dataclasses in
  `_MESSAGE_TYPES`.
```

- [ ] **Step 4: Lint and full suite**

Run: `poetry run pytest tests/test_markdown_lint.py -v && poetry run pytest`
Expected: lint clean, all green. If lint fails: `poetry run pymarkdown --config .pymarkdown.json scan README.md CLAUDE.md` for rule numbers.

- [ ] **Step 5: Commit**

```bash
git add README.md CLAUDE.md
git commit -m "docs: pause/resume flow (README) and the two checkpointing traps (CLAUDE.md)"
```

---

## Verification

- [ ] `poetry run pytest` — full suite green, LLM-free.
- [ ] Manual live check (needs Ollama): `poetry run demo --pause`, confirm exit-0 with a seeded `review.jsonl`; type answers into it; `poetry run demo --resume output/run_<ts>`; confirm only answered phases revise and `final_report.md` lands. Then `--resume` again: expect the loud "Already resumed" refusal.
- [ ] `poetry run demo` (no flags) still behaves exactly as before — interactive gate, no `checkpoints/` directory created.
- [ ] `OLLAMA_E2E=1 poetry run pytest -m ollama -s` still passes (~10 min).
