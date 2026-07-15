# Bounded Ledger — Design

**Date:** 2026-07-15
**Status:** Approved
**Extends:** `2026-07-14-hotl-pipeline-design.md` (§5 ledger schema, §6 tools, §8 review gate) and `2026-07-15-review-gate-checkpointing-design.md` (§3.1 guard, §5 already-resumed refusal — amended by §5 below). Those specs stay authoritative for everything else.

## 1. Purpose

Unbounded question accumulation defeats the point of attended automation: if agents can escalate everything, the human ends up doing the pipeline's thinking. This spec bounds what the gate presents.

- Every question now carries an **importance** flag (`high` / `medium` / `low`) and an **impact** statement — *how the human's answer would change the migration decision*.
- The gate has a configurable number of **slots** (`--max-questions`, default **3**). Open questions compete; only the most influential are presented. The rest are **deferred**: their default assumptions stand, and the final report says so explicitly.

Competition happens **at the gate** (decided in the checkpointing brainstorm): the ledger stays append-only and machine-owned; bounding applies to what the human sees, not to what agents may record.

## 2. The competition model

### Selection is a pure function

```python
_IMPORTANCE_RANK = {"high": 0, "medium": 1, "low": 2}

def select_for_review(open_questions: list[dict],
                      max_questions: int) -> tuple[list[dict], list[dict]]:
    """Split the open ledger into (presented, deferred), most influential first.

    Sort key is (importance tier, ledger position): high before medium before
    low, raise order breaking ties. Deterministic - no LLM judge; the raising
    agent already encoded its judgment in ``importance``.
    """
    ranked = sorted(enumerate(open_questions),
                    key=lambda iq: (_IMPORTANCE_RANK[iq[1]["importance"]], iq[0]))
    chosen = {i for i, _ in ranked[:max_questions]}
    presented = [q for i, q in enumerate(open_questions) if i in chosen]
    deferred = [q for i, q in enumerate(open_questions) if i not in chosen]
    return presented, deferred
```

(Illustrative; the implementation may simplify, but the observable contract — tier order, raise-order tie-break, both outputs in ledger order — is fixed.)

### Deferral is terminal

The gate runs exactly once, so losing the competition is permanent for the run: the question's stated default assumption applies, and the adjudication log gains an explicit branch — `deferred (over slot limit) - default applied: <default>`. Deferred is distinct from post-gate `open` (raised during a revision, never eligible to compete) and both appear in the log under their own wording.

### The degenerate case is a feature

`--max-questions 0` defers everything: the gate latches, marks all questions deferred, emits no `request_info`, and releases the report immediately. That is the fully autonomous, defaults-only closed pipeline as a special case of the HOTL one. Allowed and documented. Negative values are rejected by the CLI.

## 3. Schema changes

### The tool

```python
raise_question(question, context, default_assumption, importance, impact)
```

- `importance` must be exactly one of `high` / `medium` / `low`; anything else (including omission) returns an ERROR string the framework feeds back for a retry — the existing validation pattern. No silent default: a lazy `medium`-everywhere would quietly degrade ranking to raise order.
- `impact` must be non-empty: one or two sentences on how the human's answer would change the migration decision. It is the justification for the claimed importance, and it is what the human reads when deciding whether to engage.
- `context` keeps its existing meaning: the evidence that motivated the question.

Prompt guidance (`_duties.md`) gains the criteria: **high** = the answer materially changes migration approach, scope, or cost; **medium** = affects one workstream or sequencing; **low** = clarification that tightens the report but changes no decision.

### The ledger entry

`importance` and `impact` join the entry, agent-supplied but validated; `status` gains the value `deferred` (with `human_answer` remaining `null`):

```json
{
  "id": "q-2",
  "phase": "deep_analysis",
  "unit": "oms-batch-recon",
  "question": "Which upstream system produces the GL extracts?",
  "context": "recon_job.py reads /mnt/nfs/finance/gl_extracts; no doc mentions it.",
  "impact": "If the producer is also migrating, the feed can move to object storage in one step; if not, an NFS bridge must be budgeted.",
  "importance": "medium",
  "default_assumption": "External finance system on the shared NFS estate.",
  "status": "open | answered | declined | deferred",
  "human_answer": null,
  "asked_at": "<iso8601>"
}
```

The frontend reads `importance`/`impact` from `ledger.jsonl` exactly as it reads the other display fields; `review.jsonl` stays id+answer only and — because deferral is marked *before* the pause seeds it — automatically contains only the presented questions.

### The request payload

`LedgerQuestionRequest` gains `importance` and `impact` fields so `_prompt_human` and any event consumer can display them. It is already in `ALLOWED_CHECKPOINT_TYPES`; no new message types are introduced (deferral is a ledger status, not a graph message).

### New store methods

- `defer_questions(ids)` — mark each id `deferred` in **one** lock acquisition and one atomic rewrite. `KeyError` on an unknown id, like `resolve_question`.
- `unresolved_questions()` — entries with status `open` **or** `deferred`, ledger order. Exists for duplicate suppression (§5); `open_questions()` keeps its exact current meaning, which the gate, the pause seeding, and the `on_answer` guard all rely on.

## 4. Gate flow

`ReviewExecutor(store, revision_order, max_questions=3)`; `build_workflow` threads it through; `main.py` supplies `--max-questions`.

`on_questionnaire_done` becomes: latch → read open questions → `select_for_review(open_qs, max_questions)` → `defer_questions([...])` for the losers → emit `request_info` per winner. Ordering matters: deferral is written **before** the workflow idles, so a `--pause` checkpoint and its seeded `review.jsonl` already reflect the competition.

Banner when deferral occurs:

```text
== REVIEW - presenting 3 of 6 open questions (3 deferred to defaults) ==
```

(unchanged wording when nothing is deferred). `_prompt_human` and the pause banner both gain two lines per question:

```text
      Importance: high
      Impact if answered: <the agent's impact statement>
```

## 5. Cross-feature amendments (verified, not assumed)

- **`already_resumed()` must tighten.** It currently treats *any* non-open entry as proof a resume ran. Deferral writes non-open entries **at pause time**, before any resume exists — the current predicate would falsely refuse every first resume of a competitive run. Amend to `any(e["status"] in ("answered", "declined") ...)`: verdicts, specifically. (A deferred entry genuinely is not a verdict; only a resume produces those.)
- **Checkpointing spec §8 correction.** That section claims Spec B lands "with no edit to this design". True for the `on_answer` guard it was written about; false for `already_resumed`, which was added later by the adversarial-review fix. This spec supersedes that sentence.
- **Duplicate suppression must include deferred questions.** Phase prompts currently suppress re-raising via the open ledger; a deferred question vanishes from that list, so a revising agent could re-raise it as new — and post-gate it would sit `open` forever, cluttering the report with a duplicate. The prompt builders (`build_initial_prompt` / `build_revision_prompt`) switch to `unresolved_questions()`.
- **Unaffected, verified:** `on_answer`'s `open_questions()` guard (presented-and-unresolved is exactly right after deferral); `review.jsonl` seeding (reads `open_questions()` post-deferral); `ALLOWED_CHECKPOINT_TYPES` (no new message types); the review-once latch; revision routing.

## 6. Changes to existing code

| File | Change |
|---|---|
| `artifacts.py` | `raise_question` gains `importance`/`impact` (validated at the tool layer, stored here); `defer_questions`; `unresolved_questions` |
| `tools.py` | `raise_question` tool: two new params, enum + non-empty validation, docstring teaching the criteria |
| `review.py` | `select_for_review`; `ReviewExecutor(max_questions=3)`; gate flow marks deferrals before prompting; banner; `LedgerQuestionRequest` + `importance`/`impact` |
| `main.py` | `--max-questions` (default 3, non-negative); `already_resumed` tightened; display lines in `_prompt_human`/`_write_pause_files` |
| `pipeline.py` | `build_workflow(..., max_questions=3)` threaded to the gate |
| `report.py` | `render_adjudication_log`: explicit `deferred` branch |
| `phases.py` | `PhaseExecutor`'s two prompt-assembly call sites (`_run_initial`, `on_revision`) feed `store.unresolved_questions()` into the builders instead of `open_questions()` (dup suppression; builder signatures unchanged) |
| `prompts/_duties.md` | Importance criteria + impact requirement (markdown lint gate applies) |
| `tests/conftest.py` | `DriveAgent.raise_question` calls gain the two args |

## 7. Error handling

- Invalid/missing `importance` or empty `impact` → ERROR string, model retries (never raises; existing tool contract).
- `defer_questions` with an unknown id → `KeyError` (programming error, loud).
- `--max-questions` negative → `parser.error` before any work.
- Old paused runs do **not** survive this upgrade: a pre-Spec-B checkpoint pickles `LedgerQuestionRequest` instances without the new fields, and display code would `AttributeError` on restore. Start a fresh run; one sentence in the README says so. (Demo-scale honesty beats a migration shim.)

## 8. Testing (LLM-free)

- `select_for_review`: tier ordering; raise-order tie-break inside a tier; fewer open than slots (all presented, none deferred); `max_questions=0` (all deferred); output lists preserve ledger order.
- `defer_questions`: statuses flip atomically; unknown id raises; `human_answer` untouched.
- Tool validation: bad importance → ERROR string naming the allowed values; empty impact → ERROR; valid call records both fields.
- `already_resumed`: deferred-only ledger → `False` (the regression this spec exists to prevent); answered or declined → `True`.
- `render_adjudication_log`: deferred branch wording; deferred vs post-gate-open are distinct rows.
- Duplicate suppression: `build_revision_prompt` includes a deferred question's id in the suppression list.
- **Integration tests embrace the default**: the pipeline drive and pause/resume cycle tests keep raising 5 questions and now assert 3 presented + 2 deferred (by tier/raise-order), deferred entries in the final adjudication log, and — in the cycle test — `review.jsonl` seeded with exactly the 3 presented ids. The regression suite itself demonstrates the feature.
- Live E2E: unchanged in structure; the scripted loop answers whatever the gate presents.

## 9. Non-goals

- **No eviction at raise time.** Settled in the checkpointing brainstorm: the ledger stays append-only; bounding is presentation-side.
- **No re-ranking after revisions.** The gate runs once; questions raised during re-runs never compete and are listed as open-with-default, exactly as today.
- **No LLM-judged importance.** Deterministic ranking from the agent-declared flag; a judge call would add cost and non-determinism for a demo-invisible gain.
- **No cross-run memory of deferrals.** Each run competes fresh.

## 10. Decisions log

| Decision | Choice | Alternatives considered |
|---|---|---|
| Competition site | At the gate (rank + defer) | Eviction at raise time (user-decided against in the checkpointing brainstorm: order-dependent, breaks append-only) |
| Default slots | 3 (`--max-questions`) | 5 (feature invisible in a typical 5-question demo run); unlimited-unless-flagged (invisible by default) |
| Ranking | `(importance tier, raise order)`, pure function | LLM judge at the gate (nondeterministic, extra call); recency-weighted (later ≠ more influential) |
| `importance` handling | Required, enum-validated, ERROR retry | Optional default `medium` (lazy models flatten ranking to raise order, silently) |
| Impact capture | Separate structured `impact` field | Enriched `context` prose (frontend would re-parse prose; agents blur evidence with consequence) |
| Deferral mechanics | Ledger status `deferred`, marked pre-pause in one atomic rewrite | Presentation-only filtering with no status (report cannot distinguish deferred from open; resume seeding would leak deferred ids into `review.jsonl`) |
| `--max-questions 0` | Allowed: fully autonomous defaults-only run | Rejecting it (forecloses the closed-pipeline degenerate case for free) |
| Old paused runs | Not resumable across the upgrade; documented | Pickle-compat shim for `LedgerQuestionRequest` (migration machinery for a demo) |
