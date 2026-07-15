# Bounded Ledger — Design

**Date:** 2026-07-15
**Status:** Approved (rev 2, 2026-07-16 — semantic LLM ranking at the gate replaces raise-order tie-breaks; importance/status/phase become typed enums)
**Extends:** `2026-07-14-hotl-pipeline-design.md` (§5 ledger schema, §6 tools, §8 review gate) and `2026-07-15-review-gate-checkpointing-design.md` (§3.1 guard, §5 already-resumed refusal — amended by §5 below). Those specs stay authoritative for everything else.

## 1. Purpose

Unbounded question accumulation defeats the point of attended automation: if agents can escalate everything, the human ends up doing the pipeline's thinking. This spec bounds what the gate presents.

- Every question now carries an **importance** flag (`high` / `medium` / `low`) and an **impact** statement — *how the human's answer would change the migration decision*.
- The gate has a configurable number of **slots** (`--max-questions`, default **3**). Open questions compete; only the most influential are presented. The rest are **deferred**: their default assumptions stand, and the final report says so explicitly.

Competition happens **at the gate** (decided in the checkpointing brainstorm): the ledger stays append-only and machine-owned; bounding applies to what the human sees, not to what agents may record.

## 2. The competition model

### Ranking is semantic; raise order carries zero signal

The most profound questions — the ones whose answers would swing the final report's verdict hardest — must win the slots. That is a judgment over the *content* of `question`, `impact`, and `context` (with the agent-declared `importance` as one input among them, not a hard tier), so ranking is performed by **one LLM call at the gate**. Raise order and `asked_at` are explicitly **not** ranking inputs: when a question was asked says nothing about whether it matters, and the ranker's prompt states that ids carry no ordering signal.

The ranker is a tool-less `Agent` owned by `ReviewExecutor` (test seam `ranker=` like every other executor's `agent=`), prompted via a new `prompts/rank_questions.md` template (prompts-are-data convention; the lint gate covers it). It receives every open question's `id`, `question`, `impact`, `context`, `importance`, and `default_assumption` — never `asked_at` — and must return the ids ordered by expected swing on the final report, biggest first, one id per line.

**The ranker cannot kill the run.** Its output is validated by a pure function (`validate_ranking(candidate_ids, text) -> list[str] | None`: every candidate exactly once, nothing else); an invalid response gets one corrective retry; if that also fails, the gate falls back to a deterministic degraded order — importance tier, then id — and says so on stdout. The fallback exists only for ranker failure; the normal path is fully semantic.

**Ranking only happens when there is real competition.** `len(open) <= max_questions` presents everything with no LLM call; `max_questions == 0` defers everything with no LLM call.

### Splitting stays pure

```python
def split_ranked(ranked_ids: list[str], open_questions: list[dict],
                 max_questions: int) -> tuple[list[dict], list[dict]]:
    """(presented, deferred) - winners are the first max_questions ranked ids;
    both output lists are returned in LEDGER order for stable display."""
```

(Illustrative; the observable contract — winners by ranked prefix, outputs in ledger order — is fixed.)

### Deferral is terminal

The gate runs exactly once, so losing the competition is permanent for the run: the question's stated default assumption applies, and the adjudication log gains an explicit branch — `deferred (over slot limit) - default applied: <default>`. Deferred is distinct from post-gate `open` (raised during a revision, never eligible to compete) and both appear in the log under their own wording.

### The degenerate case is a feature

`--max-questions 0` defers everything: the gate latches, marks all questions deferred, emits no `request_info`, and releases the report immediately. That is the fully autonomous, defaults-only closed pipeline as a special case of the HOTL one. Allowed and documented. Negative values are rejected by the CLI.

## 3. Schema changes

### Typed enums, not string literals

`artifacts.py` gains three `str`-based enums — the exact form pydantic enum fields consume (pydantic is already in the dependency tree via `agent-framework`), while keeping JSONL on disk and every existing `==` comparison working unchanged. `enum.StrEnum` is 3.11+; the project floor is 3.10, hence `(str, Enum)`:

```python
class Phase(str, Enum):
    DISCOVERY = "discovery"
    DEEP_ANALYSIS = "deep_analysis"
    ENTERPRISE_CONTEXT = "enterprise_context"
    QUESTIONNAIRE = "questionnaire"

class QuestionStatus(str, Enum):
    OPEN = "open"
    ANSWERED = "answered"
    DECLINED = "declined"
    DEFERRED = "deferred"

class Importance(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

PHASES: tuple[str, ...] = tuple(p.value for p in Phase)   # existing consumers unchanged
```

Values are stored in the JSONL as plain strings (`.value`), so the frontend and every existing artifact reader are unaffected. Code replaces bare literals (`"open"`, `"answered"`, …) with enum members; validation at the tool layer becomes enum membership (`Importance(value)` raising `ValueError` → the ERROR-retry path).

### The tool

```python
raise_question(question, context, default_assumption, importance, impact)
```

- `importance` must be exactly one of `high` / `medium` / `low` (the `Importance` enum values); anything else (including omission) returns an ERROR string the framework feeds back for a retry — the existing validation pattern. No silent default: a lazy `medium`-everywhere would blind one of the ranker's input signals and hollow out the degraded fallback entirely.
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

`on_questionnaire_done` becomes: latch → read open questions → if competition is real, run the ranker (`rank_questions.md` prompt → `validate_ranking` → retry once → degraded fallback) → `split_ranked(...)` → `defer_questions([...])` for the losers → emit `request_info` per winner. Ordering matters: deferral is written **before** the workflow idles, so a `--pause` checkpoint and its seeded `review.jsonl` already reflect the competition.

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
| `artifacts.py` | `Phase`/`QuestionStatus`/`Importance` enums (`PHASES` derived); `raise_question` gains `importance`/`impact`; `defer_questions`; `unresolved_questions`; literals → enum members |
| `tools.py` | `raise_question` tool: two new params, enum-membership + non-empty validation, docstring teaching the criteria |
| `review.py` | Ranker `Agent` (tool-less, `ranker=` test seam) + `validate_ranking` + `split_ranked` + degraded fallback; `ReviewExecutor(max_questions=3)`; gate marks deferrals before prompting; banner; `LedgerQuestionRequest` + `importance`/`impact` |
| `prompts/rank_questions.md` | New ranking prompt template: candidates rendered without `asked_at`; instructs that ids/raise order carry no signal; output one id per line, biggest expected report-swing first |
| `main.py` | `--max-questions` (default 3, non-negative); `already_resumed` tightened; display lines in `_prompt_human`/`_write_pause_files` |
| `pipeline.py` | `build_workflow(..., max_questions=3)` threaded to the gate |
| `report.py` | `render_adjudication_log`: explicit `deferred` branch |
| `phases.py` | `PhaseExecutor`'s two prompt-assembly call sites (`_run_initial`, `on_revision`) feed `store.unresolved_questions()` into the builders instead of `open_questions()` (dup suppression; builder signatures unchanged) |
| `prompts/_duties.md` | Importance criteria + impact requirement (markdown lint gate applies) |
| `tests/conftest.py` | `DriveAgent.raise_question` calls gain the two args |

## 7. Error handling

- Invalid/missing `importance` or empty `impact` → ERROR string, model retries (never raises; existing tool contract).
- Ranker output invalid (missing/duplicate/foreign ids) → one corrective retry carrying the validation complaint; still invalid → deterministic degraded fallback (importance tier, then id), announced on stdout. The gate never crashes on a bad ranking.
- `defer_questions` with an unknown id → `KeyError` (programming error, loud).
- `--max-questions` negative → `parser.error` before any work.
- Old paused runs do **not** survive this upgrade: a pre-Spec-B checkpoint pickles `LedgerQuestionRequest` instances without the new fields, and display code would `AttributeError` on restore. Start a fresh run; one sentence in the README says so. (Demo-scale honesty beats a migration shim.)

## 8. Testing (LLM-free)

- `validate_ranking`: accepts a permutation of the candidates (whitespace/blank-line tolerant); rejects missing, duplicate, and foreign ids → `None`.
- `split_ranked`: winners are the ranked prefix; both outputs in ledger order; fewer open than slots (all presented, none deferred); `max_questions=0` (all deferred).
- Gate ranking flow (scripted ranker via the `ranker=` seam, `FakeAgent` pattern): valid ranking → winners presented; first response invalid → retry consulted; both invalid → degraded fallback order used and announced; no ranker call when `len(open) <= slots` or `slots == 0`.
- Enums: JSONL round-trips plain strings; `Importance("mid")` raises → tool returns the ERROR string naming allowed values.
- `defer_questions`: statuses flip atomically; unknown id raises; `human_answer` untouched.
- Tool validation: bad importance → ERROR string naming the allowed values; empty impact → ERROR; valid call records both fields.
- `already_resumed`: deferred-only ledger → `False` (the regression this spec exists to prevent); answered or declined → `True`.
- `render_adjudication_log`: deferred branch wording; deferred vs post-gate-open are distinct rows.
- Duplicate suppression: `build_revision_prompt` includes a deferred question's id in the suppression list.
- **Integration tests embrace the default**: the pipeline drive and pause/resume cycle tests keep raising 5 questions; the review ranker is scripted (same monkeypatch pattern as `phases.Agent`/`report.Agent` — `DriveAgent` learns to answer the ranking prompt with a fixed order). Assert: exactly the scripted top-3 are presented, 2 deferred, deferred rows in the adjudication log, and — in the cycle test — `review.jsonl` seeded with exactly the 3 presented ids. The regression suite itself demonstrates the feature.
- Live E2E: unchanged in structure; the scripted loop answers whatever the gate presents (now the ranker's real top 3).

## 9. Non-goals

- **No eviction at raise time.** Settled in the checkpointing brainstorm: the ledger stays append-only; bounding is presentation-side.
- **No re-ranking after revisions.** The gate runs once; questions raised during re-runs never compete and are listed as open-with-default, exactly as today.
- **No determinism guarantee for ranking.** Semantic ranking is an LLM judgment and may vary run to run; the deterministic order exists only as the degraded fallback. Accepted deliberately (rev 2): profundity is semantic, and raise order must carry no signal.
- **No cross-run memory of deferrals.** Each run competes fresh.

## 10. Decisions log

| Decision | Choice | Alternatives considered |
|---|---|---|
| Competition site | At the gate (rank + defer) | Eviction at raise time (user-decided against in the checkpointing brainstorm: order-dependent, breaks append-only) |
| Default slots | 3 (`--max-questions`) | 5 (feature invisible in a typical 5-question demo run); unlimited-unless-flagged (invisible by default) |
| Ranking | Semantic LLM ranker at the gate: expected swing on the final report, over `question`/`impact`/`context` with `importance` as one input; raise order and `asked_at` excluded | rev 1's `(importance tier, raise order)` pure function — rejected by user feedback: profundity is semantic and raise order must carry zero signal; recency-weighted (later ≠ more influential) |
| Ranker robustness | `validate_ranking` → one corrective retry → deterministic degraded fallback (tier, then id), announced | Trusting raw output (a flaky local model kills the gate); no fallback (run dies on a formatting error) |
| Ranking scope | Only when `len(open) > slots > 0` | Always rank (wasted LLM call when nothing competes) |
| Field typing | `(str, Enum)` for `Phase`/`QuestionStatus`/`Importance`, plain-string values on disk | Bare literals (rev 1 — user feedback); `enum.StrEnum` (3.11+, floor is 3.10); full pydantic `BaseModel` entries (a far larger refactor than the ask, for a dict-based store) |
| `importance` handling | Required, enum-validated, ERROR retry | Optional default `medium` (lazy models flatten the signal, silently) |
| Impact capture | Separate structured `impact` field | Enriched `context` prose (frontend would re-parse prose; agents blur evidence with consequence) |
| Deferral mechanics | Ledger status `deferred`, marked pre-pause in one atomic rewrite | Presentation-only filtering with no status (report cannot distinguish deferred from open; resume seeding would leak deferred ids into `review.jsonl`) |
| `--max-questions 0` | Allowed: fully autonomous defaults-only run | Rejecting it (forecloses the closed-pipeline degenerate case for free) |
| Old paused runs | Not resumable across the upgrade; documented | Pickle-compat shim for `LedgerQuestionRequest` (migration machinery for a demo) |
