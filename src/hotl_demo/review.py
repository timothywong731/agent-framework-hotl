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
from dataclasses import dataclass

from agent_framework import Executor, WorkflowContext, handler, response_handler

from .artifacts import ArtifactStore
from .phases import PhaseDone, ReportTrigger, RevisionDone, RevisionTrigger


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
    """

    question_id: str
    phase: str
    unit: str | None
    question: str
    context: str
    default_assumption: str


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


class ReviewExecutor(Executor):
    """Single-shot human gate between ``questionnaire`` and ``final_report``.

    State machine:

    1. ``on_questionnaire_done`` - latch ``review_completed`` (the
       review-once rule), then emit one ``request_info`` per open question;
       the workflow run goes idle until the CLI resumes it with responses.
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
                 revision_order: list[tuple[str, str | None]]) -> None:
        """Remember the store and the canonical re-run ordering.

        Deliberately no adjudication counters here: gate progress must be
        derived from the ledger so that a checkpoint resume (fresh instance)
        cannot diverge from a live run. See the checkpointing spec.

        Args:
            store: The run's shared artifact store.
            revision_order: Every ``(phase, unit)`` in pipeline order,
                exactly as built from the phase specs.
        """
        super().__init__(id="review")
        self._store = store
        self._revision_order = revision_order
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
        print(f"\n== REVIEW - {len(open_qs)} open questions ==")
        for q in open_qs:
            # One request_info per question: the workflow idles after this
            # handler returns, until the runner calls run(responses={...}).
            await ctx.request_info(
                request_data=LedgerQuestionRequest(
                    question_id=q["id"], phase=q["phase"], unit=q["unit"],
                    question=q["question"], context=q["context"],
                    default_assumption=q["default_assumption"],
                ),
                response_type=str,
            )

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
