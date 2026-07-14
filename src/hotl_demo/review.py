"""The human review gate: presents the ledger once, routes selective re-runs."""
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
    question_id: str
    phase: str
    unit: str | None
    question: str
    context: str
    default_assumption: str


def affected_targets(ledger: list[dict],
                     revision_order: list[tuple[str, str | None]]) -> list[tuple[str, str | None]]:
    answered = {(e["phase"], e["unit"]) for e in ledger if e["status"] == "answered"}
    return [t for t in revision_order if t in answered]


def answers_for(ledger: list[dict], phase: str, unit: str | None) -> list[dict]:
    return [e for e in ledger
            if e["status"] == "answered" and e["phase"] == phase and e["unit"] == unit]


class ReviewExecutor(Executor):
    def __init__(self, store: ArtifactStore,
                 revision_order: list[tuple[str, str | None]]) -> None:
        super().__init__(id="review")
        self._store = store
        self._revision_order = revision_order
        self._awaiting = 0
        self._queue: list[RevisionTrigger] = []

    @handler
    async def on_questionnaire_done(
        self, done: PhaseDone,
        ctx: WorkflowContext[ReportTrigger | RevisionTrigger],
    ) -> None:
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
        self._awaiting = len(open_qs)
        for q in open_qs:
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
        text = (answer or "").strip()
        self._store.resolve_question(
            original.question_id, "answered" if text else "declined", text or None
        )
        self._awaiting -= 1
        if self._awaiting > 0:
            return
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
        await self._dispatch_next(ctx)

    async def _dispatch_next(self, ctx) -> None:
        if self._queue:
            await ctx.send_message(self._queue.pop(0))
        else:
            await ctx.send_message(ReportTrigger())
