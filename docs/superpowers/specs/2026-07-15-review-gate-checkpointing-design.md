# Review-Gate Checkpointing — Design

**Date:** 2026-07-15
**Status:** Approved
**Extends:** `2026-07-14-hotl-pipeline-design.md` (§8 review gate, §10 CLI, §12 testing) — authoritative for everything else.
**Verified against:** `agent-framework` 1.11.0. Every claim below was proven by an executable spike against the real assembled graph (LLM-free, via `test_pipeline.py`'s `_DriveAgent` technique), not read from documentation.
**Sibling spec (later):** bounded ledger — importance flag + slot competition at the gate. Independent of this one; see §8.

## 1. Purpose

Answering the review gate's questions is a **days-long human process**: the operator must go and ask people. Today the pipeline holds a live process open on a blocking `input()` for that entire time, pinning a workflow, an Ollama connection, and every agent object in memory.

This spec adds a real pause: at the review gate the run **checkpoints and exits**. The human answers at leisure — by hand, or through a frontend (out of scope) — and `demo --resume <run_dir>` restores the run and continues. Restoring re-runs **zero phases**; no LLM work is repeated.

Interactive prompting remains the **default** — it is the better live demo. Checkpointing is opt-in via `--pause`.

## 2. What the spike established

The design rests on facts, each verified by running code:

| Question | Finding |
|---|---|
| Can the framework serialize our types? | **Only with an allowlist.** Checkpoints are *pickled*; all six of our dataclasses are blocked by default |
| Does resume re-emit pending requests? | **Yes** — all 5, and `request_id`s are **stable** across restore |
| Does resume re-run phases? | **No.** Zero phase re-executions; restore is instant |
| Is `list_checkpoints` ordered? | **No** — it is `storage_path.glob("*.json")` over UUID filenames |
| Does `ReviewExecutor._awaiting` survive? | **No, and it fails catastrophically** — see §3 |

### The three silent traps

Every one of these fails *quietly*, producing a plausible-looking run rather than an error. That is what makes them worth designing against explicitly.

1. **A missing allowlist entry is indistinguishable from "no checkpoints exist."** `list_checkpoints` catches each read failure, logs it, and returns `[]`. Add a message type later, forget the allowlist, and resume simply stops working with no exception.
2. **Resuming from the wrong checkpoint skips the human entirely.** Observed: selecting `cps[-1]` re-ran `questionnaire`; the `review_completed` latch — which lives in `memory.json` and therefore survives the restart — told the re-entered gate "already reviewed", which emitted `ReportTrigger` and produced a final report with **all five questions unanswered**. The review-once latch and checkpointing actively fight each other.
3. **Resume is not idempotent.** Resuming twice would re-run every revision and rewrite the report.

### The `_awaiting` bug, measured

`ReviewExecutor` tracks `self._awaiting` and `self._queue` as plain Python attributes. The framework only checkpoints executor state registered via `ctx.set_executor_state()`, which this codebase never calls. On resume the executor is a fresh object, so `_awaiting == 0`; the `if self._awaiting > 0: return` guard fails on the *first* answer, and every answer then believes it is the last.

Spike output, resuming with 5 answers:

```text
Re-running affected: discovery
Re-running affected: discovery, deep_analysis[oms-monolith]
Re-running affected: discovery, ... , questionnaire      <- five overlapping queues
  revised: discovery      (x5)
  final_report: written   (x5)
```

`discovery` re-ran **five times**, `final_report` was written **five times**, and the "exactly one revision in flight" guarantee was destroyed. No exception was raised; the ledger still ended all-`answered`, so the run looks successful. On a real model this is 5x the cost plus a race.

Everything else in the codebase is checkpoint-safe *by accident*: `ArtifactStore` already put all durable state on disk. `_awaiting` is the one exception.

## 3. Design

### 3.1 The `_awaiting` fix is a deletion

The ledger already knows the answer:

```python
self._store.resolve_question(original.question_id, "answered" if text else "declined", text or None)
if self._store.open_questions():
    return                      # verdicts still outstanding
```

File-backed, so it survives any restart, and `_awaiting` disappears. `resolve_question` moves an entry off `open`, so `open_questions()` empties exactly once — on the last verdict. Precisely the semantics `_awaiting == 0` intended.

**Verified**: with this single change, the spike run above becomes one queue, one revision per target in pipeline order, one final report.

It also composes with the sibling spec: once questions can be *deferred* at the gate, deferred entries are not `open`, so this guard keeps meaning "all **presented** questions are resolved" with no edit.

`self._queue` stays in memory deliberately. It is built by `on_answer` *after* the resume and consumed by `on_revision_done` within the same process, so it never needs to cross a restart. See §7.

### 3.2 Checkpoint wiring

`WorkflowBuilder` takes `checkpoint_storage` as a **constructor argument**. (`with_checkpointing()` appears in the published docs but **does not exist** in 1.11.0.) `list_checkpoints` requires a `workflow_name`, which the builder does not currently set.

```python
WORKFLOW_NAME = "hotl-migration-readiness"

ALLOWED_CHECKPOINT_TYPES = [
    "hotl_demo.phases:PhaseDone",
    "hotl_demo.phases:AnalysisDone",
    "hotl_demo.phases:RevisionDone",
    "hotl_demo.phases:RevisionTrigger",
    "hotl_demo.phases:ReportTrigger",
    "hotl_demo.review:LedgerQuestionRequest",
]
```

`build_workflow` gains `checkpoint_storage: CheckpointStorage | None = None`, passed straight through to `WorkflowBuilder(name=WORKFLOW_NAME, start_executor=discovery, checkpoint_storage=checkpoint_storage)`. Default `None` keeps the interactive path byte-for-byte unchanged.

The allowlist lives next to the message types it names, and a test asserts every `@dataclass` message in `phases.py`/`review.py` appears in it — otherwise trap 1 lands on a future maintainer.

### 3.3 Selecting the gate checkpoint semantically

Never "the latest". The review-gate checkpoint is *by definition* the one idle with pending requests:

```python
def gate_checkpoint(checkpoints: list[WorkflowCheckpoint]) -> WorkflowCheckpoint | None:
    """The review-gate pause: the checkpoint idle with pending request_info events.

    list_checkpoints() is glob-ordered, NOT chronological, so positional
    selection is meaningless.
    """
    pending = [c for c in checkpoints if c.pending_request_info_events]
    return max(pending, key=lambda c: c.iteration_count) if pending else None
```

A pure function over a list — directly unit-testable with hand-built checkpoint objects, no framework run required.

### 3.4 `review.jsonl` — the human's inbox

Written to `output/run_<timestamp>/review.jsonl` at pause, one JSON object per line:

```json
{"id": "q-1", "phase": "discovery", "unit": null, "question": "Is the 'OMS Batch Reconciliation' tool in scope?", "context": "The architecture document does not mention oms-batch-recon...", "default_assumption": "oms-batch-recon is included in migration scope", "answer": ""}
```

JSONL rather than markdown so a frontend gets **structured fields** rather than prose to re-parse, and to match `ledger.jsonl`'s existing convention.

**Deliberately a separate file from `ledger.jsonl`.** The ledger is append-only and machine-owned — CLAUDE.md's rule is that every mutation goes through `ArtifactStore`. A human or frontend writing into it invites corruption against a concurrent writer. The ledger is the system's record; `review.jsonl` is the human's inbox/outbox, applied back through the existing `resolve_question` path.

- `id` is the join key (our stable `q-N`, not the framework's `request_id`).
- `answer` is the only field the human or frontend writes. Empty or whitespace = **decline**.
- Forward-compatible: when the sibling spec lands, `importance` joins the record, since the file is rendered from ledger entries.

### 3.5 The three flows

| Command | Behaviour |
|---|---|
| `demo` | Interactive, unchanged. **No checkpointing** — zero new risk to the existing demo or the `OLLAMA_E2E` test |
| `demo --pause` | Checkpointing on. At the gate: write `review.jsonl`, print the resume command, exit 0 without prompting |
| `demo --resume <run_dir>` | Rebuild → select gate checkpoint → read `review.jsonl` → resume → revisions → final report |

`--resume` only works for runs started with `--pause`, because only those wrote checkpoints. Stated in `--help` and in the error path.

Two consequences to make explicit rather than leave implied:

- **A run with no open questions never pauses.** The gate releases `ReportTrigger` immediately, so `--pause` completes the run in one go and writes no `review.jsonl`. `--pause` is a request to pause *if the gate opens*, not an unconditional stop.
- **`--resume` takes `--model` exactly as a fresh run does**, and preflights it. The model is an environment concern, not workflow state, so it is deliberately not recorded in the checkpoint — pass the same model you paused with. Resuming a `gemma4:31b` pause under a different model will produce revisions from that other model.

`main.py`'s existing loop already does *run → collect pending → respond → run(responses=)*. Resume changes only the first call and the source of the answers:

```python
stream = (workflow.run(checkpoint_id=gate.checkpoint_id, stream=True) if resuming
          else workflow.run("start", stream=True) if responses is None
          else workflow.run(stream=True, responses=responses))
```

Answers are mapped from `review.jsonl` by `question_id` off the re-emitted `LedgerQuestionRequest` payloads. Request ids proved stable across restore, so `run(checkpoint_id=..., responses=...)` in a single call would also work — but keying off the re-emitted events costs nothing and does not depend on that framework-internal detail holding.

## 4. Changes to existing code

| File | Change |
|---|---|
| `review.py` | Delete `_awaiting`; guard on `self._store.open_questions()` |
| `pipeline.py` | `WORKFLOW_NAME`, `ALLOWED_CHECKPOINT_TYPES`, `gate_checkpoint()`; `build_workflow(..., checkpoint_storage=None)`; name the builder |
| `main.py` | `--pause` / `--resume`; render + parse `review.jsonl`; resume branch in the run loop |
| `artifacts.py` | None. It already does the hard part |

## 5. Error handling

Every failure here must be **loud**, because the failure mode is a plausible-looking report built without the human's input.

- **Malformed `review.jsonl` line** → abort with the line number. A parse error must **never** degrade to "decline"; that would silently discard days of gathered answers and proceed on defaults.
- **No gate checkpoint found** → actionable message distinguishing the two real causes: the run was not started with `--pause`, or a message type is missing from `ALLOWED_CHECKPOINT_TYPES` (trap 1 — remember `list_checkpoints` returns `[]` rather than raising).
- **Ledger already fully resolved** → "this run has already been resumed"; exit without re-running. Closes trap 3, and is file-derived like everything else.
- **`id` in `review.jsonl` with no matching pending request** → warn and ignore.
- **Pending request with no `id` in the file** → decline (consistent with "empty = decline").
- **Graph changed between pause and resume** (prompt files edited, a repo added) → the checkpoint carries a `graph_signature_hash`; surface the framework's mismatch error as "the pipeline changed since this run was paused; start a fresh run."

## 6. Testing (LLM-free)

The spike becomes the regression test — `_DriveAgent` drives the real assembled graph with no LLM, so a full pause/resume cycle runs in the default suite.

- **`gate_checkpoint()`** — pure: prefers pending-request checkpoints, picks highest `iteration_count`, returns `None` when none pending, and ignores list order (feed it shuffled).
- **`review.jsonl` render/parse** — round-trip; blank answer = decline; malformed line raises with a line number; unknown id warns; missing id declines.
- **Full resume cycle** (the important one) — run to the gate, build a **fresh workflow instance with a fresh `ArtifactStore` over the same run dir**, resume from the gate checkpoint, answer all five, then assert **exactly one revision per target, in pipeline order, and exactly one `final_report` write**. This is the precise regression guard for the 5x `_awaiting` bug; it fails loudly if the guard ever regresses.
- **Allowlist completeness** — assert every `@dataclass` message type in `phases.py` and `review.py` is listed in `ALLOWED_CHECKPOINT_TYPES`.

## 7. Non-goals and known limits

- **Resume only from the gate checkpoint.** An earlier checkpoint would restore a fresh `JoinAnalyses` whose `_seen` set is empty (also plain Python state), so the fan-in barrier would never release and the run would hang. Only the gate checkpoint is offered, and `gate_checkpoint()` cannot select any other.
- **No mid-revision crash recovery.** `_queue` is in-memory; a checkpoint taken between revisions would restore an empty queue and jump to the report. Out of scope: the requirement is a *planned* pause at the gate, not fault tolerance. Fixing it would mean moving the queue into `ctx.set_executor_state()` or deriving it from the ledger.
- **Checkpointing stays off by default**, so a crash during an interactive run is still unrecoverable. Deliberate: it keeps the default path and the live E2E test untouched.
- **Checkpoint storage is a trust boundary.** It is pickle behind an allowlist. `output/` is gitignored and local; do not load a checkpoint from anywhere you would not run code from.

## 8. Relationship to the bounded-ledger spec

Independent — verified, not assumed. The §3.1 guard is `open_questions()`, which means "presented and unresolved" both today (presented == all open) and after the sibling spec introduces a non-`open` *deferred* status for questions that lose the slot competition. No ordering dependency, and no edit to this design when that lands.

## 9. Decisions log

| Decision | Choice | Alternatives considered |
|---|---|---|
| Pause mechanism | Framework checkpoint + resume | Re-run the pipeline from artifacts (re-invokes every LLM call — the whole cost this avoids); inject a synthetic `PhaseDone` at `review` (fights the `review_completed` latch; bypasses the framework's own mechanism) |
| `_awaiting` | Delete; derive from `open_questions()` | `ctx.set_executor_state()` (works, but adds state where deletion suffices — the ledger is already the source of truth) |
| Checkpoint selection | Semantic: `pending_request_info_events` | `cps[-1]` / `get_latest()` — **empirically produced a final report with zero questions answered** |
| Default review UX | Interactive; `--pause` opt-in | Pause by default (worse live demo, rewrites the E2E's scripted stdin); pause-only (loses the interactive path) |
| Checkpointing on by default | No — only with `--pause` | Always on (adds pickle/serialization risk and disk churn to the default path for a benefit nobody asked for) |
| Answers file format | `review.jsonl` | Markdown (nicer to hand-edit, no escaping — but a frontend would have to re-parse prose); YAML (breaks on a colon in free text) |
| Answers file identity | Separate from `ledger.jsonl` | Reuse the ledger (human/frontend writes would race `ArtifactStore` and break append-only ownership) |
| Parse failure | Abort loudly | Treat as decline (silently discards the human's work and ships a defaults-only report) |
