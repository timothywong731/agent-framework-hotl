"""The human review gate: presents the ledger once, routes selective re-runs.

This is the HOTL centerpiece. The gate pauses the workflow with one
``ctx.request_info`` per open ledger question (the framework-native pause),
receives the human's verdicts through ``@response_handler``, then re-runs
only the phases whose questions were answered - sequentially, in phase
order - before releasing the final report.
"""
# NOTE: no `from __future__ import annotations` here: @response_handler validates
# ctx via inspect.signature (not get_type_hints), so PEP 563 string annotations
# leave WorkflowContext[...] unresolved and validation rejects it. Eager
# annotations (all runtime-valid on py3.10+) are required for the response handler.
import re
from dataclasses import dataclass

from agent_framework import Agent, Executor, WorkflowContext, handler, response_handler
from agent_framework.ollama import OllamaChatClient

from .artifacts import ArtifactStore, Importance
from .compaction import resolve_num_ctx
from .phases import PROMPT_ENV, PhaseDone, ReportTrigger, RevisionDone, RevisionTrigger


@dataclass
class LedgerQuestionRequest:
    """The ``request_info`` payload the CLI renders as one review prompt.

    Attributes:
        question_id: Ledger id, e.g. ``"q-1"``.
        phase: Phase that raised the question.
        unit: Raising repo for deep_analysis questions, else ``None``.
        question: The question itself.
        context: Evidence that motivated it.
        default_assumption: What applies if the human declines.
        importance: The raising agent's declared importance (high/medium/low).
        impact: How the human's answer would change the migration decision.
    """

    question_id: str
    phase: str
    unit: str | None
    question: str
    context: str
    default_assumption: str
    importance: str
    impact: str


def affected_targets(ledger: list[dict],
                     revision_order: list[tuple[str, str | None]]) -> list[tuple[str, str | None]]:
    """Compute which ``(phase, unit)`` targets must re-run, in phase order.

    Only ANSWERED questions trigger re-runs; declined ones keep their default
    assumption and cost nothing.

    Args:
        ledger: Full ledger after the human's verdicts were recorded.
        revision_order: Canonical pipeline order of every ``(phase, unit)``;
            the result preserves this order and deduplicates targets that
            raised several answered questions.

    Returns:
        Ordered, deduplicated re-run targets.

    Example:
        >>> affected_targets(
        ...     [{"phase": "discovery", "unit": None, "status": "answered"},
        ...      {"phase": "questionnaire", "unit": None, "status": "declined"}],
        ...     [("discovery", None), ("questionnaire", None)])
        [('discovery', None)]
    """
    answered = {(e["phase"], e["unit"]) for e in ledger if e["status"] == "answered"}
    # Iterate revision_order (not the ledger) so re-runs happen in pipeline
    # order regardless of the order questions were raised or answered.
    return [t for t in revision_order if t in answered]


def answers_for(ledger: list[dict], phase: str, unit: str | None) -> list[dict]:
    """Return the answered ledger entries belonging to one re-run target.

    Args:
        ledger: Full ledger.
        phase: Target phase name.
        unit: Target repo (``None`` for non-analyzer phases).

    Returns:
        Answered entries whose ``(phase, unit)`` matches, in ledger order.
    """
    return [e for e in ledger
            if e["status"] == "answered" and e["phase"] == phase and e["unit"] == unit]


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


class ReviewExecutor(Executor):
    """Single-shot human gate between ``questionnaire`` and ``final_report``.

    State machine:

    1. ``on_questionnaire_done`` - latch ``review_completed`` (the
       review-once rule); when open questions exceed the slot budget, one
       semantic ranking call picks the winners (validated, retried once,
       degraded fallback) and the losers are marked ``deferred`` in the
       ledger BEFORE pausing; then one ``request_info`` per presented
       question idles the run until the CLI resumes it with responses.
    2. ``on_answer`` (once per question) - record the verdict; when the ledger
       has no open questions left (all verdicts in - a FILE-backed check, so a
       checkpoint-resumed gate, which is a fresh instance, behaves the same),
       build the ordered re-run queue and dispatch its head.
    3. ``on_revision_done`` - dispatch the next queued re-run, or
       ``ReportTrigger`` when the queue is empty.

    Questions raised DURING re-runs stay in the ledger but are never
    prompted (the latch is already set); the final report lists them as
    "open - default assumption applied".
    """

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
            # Single-turn agent: no history to compact, but the server must
            # honor the same window the phase agents budget for.
            default_options={"num_ctx": resolve_num_ctx()},
        )
        self._queue: list[RevisionTrigger] = []

    @handler
    async def on_questionnaire_done(
        self, done: PhaseDone,
        ctx: WorkflowContext[ReportTrigger | RevisionTrigger],
    ) -> None:
        """Open the gate: pause the workflow with one request per open question.

        Args:
            done: The questionnaire phase's initial completion.
            ctx: Workflow context; used for ``request_info`` (pause) or, when
                there is nothing to ask, to release the report immediately.
        """
        if self._store.review_completed():
            # Review runs exactly once per run; defensive guard.
            await ctx.send_message(ReportTrigger())
            return
        self._store.set_review_completed()  # set on ENTRY, before prompting (spec 8)
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

    @response_handler
    async def on_answer(
        self, original: LedgerQuestionRequest, answer: str,
        ctx: WorkflowContext[ReportTrigger | RevisionTrigger],
    ) -> None:
        """Record one human verdict; on the last one, start the re-run queue.

        Args:
            original: The request this answer belongs to (carries the
                question id).
            answer: Raw human input - any non-whitespace text is an
                authoritative answer; empty/whitespace means decline.
            ctx: Workflow context for dispatching re-runs/report.
        """
        text = (answer or "").strip()
        self._store.resolve_question(
            original.question_id, "answered" if text else "declined", text or None
        )
        if self._store.open_questions():
            return  # verdicts still outstanding - ledger-derived, so a resumed
            # run (fresh executor instance) behaves identically to this one
        ledger = self._store.read_ledger()
        targets = affected_targets(ledger, self._revision_order)
        self._queue = [
            RevisionTrigger(phase, unit, answers_for(ledger, phase, unit))
            for phase, unit in targets
        ]
        if self._queue:
            pretty = ", ".join(f"{p}[{u}]" if u else p for p, u in targets)
            print(f"Re-running affected: {pretty}")
        await self._dispatch_next(ctx)

    @handler
    async def on_revision_done(
        self, done: RevisionDone,
        ctx: WorkflowContext[ReportTrigger | RevisionTrigger],
    ) -> None:
        """Advance the queue when a phase finishes its re-run.

        Args:
            done: The revision completion (contents unused; arrival is the
                signal).
            ctx: Workflow context for the next dispatch.
        """
        await self._dispatch_next(ctx)

    async def _dispatch_next(self, ctx) -> None:
        """Send the next queued ``RevisionTrigger``, or release the report.

        Sequential by construction: exactly one revision is in flight at a
        time, so re-runs happen in pipeline order and never race each other.

        Args:
            ctx: Workflow context to send the next message on.
        """
        if self._queue:
            await ctx.send_message(self._queue.pop(0))
        else:
            await ctx.send_message(ReportTrigger())
