"""Reflexion workflow: messages, verdict schema, and the two executors.

Routing convention (same as the HOTL pipeline): message TYPES encode meaning.
``DraftReady`` only ever flows worker -> reviewer; ``ReviewVerdict`` only ever
flows reviewer -> worker. No mode flags anywhere.

This module deliberately avoids ``from __future__ import annotations``: the
framework inspects handler signatures, and string annotations are a known
trap (see the warning at the top of hotl_demo/review.py).
"""
import json
import re
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ValidationError

from .prompting import render_reviewer_prompt, render_worker_prompt
from .tools import atomic_write

REPORT_FILENAME = "report.md"
LOG_FILENAME = "review_log.jsonl"


@dataclass
class DraftReady:
    """Worker -> reviewer: draft ``cycle`` is on disk, review it."""

    cycle: int
    topic: str


@dataclass
class ReviewVerdict:
    """Reviewer -> worker: boolean verdict plus steering feedback.

    ``reviewer_tool_calls`` rides along so the worker executor - the sole
    review-log writer - can log complete cycle lines.
    """

    approved: bool
    feedback: str
    cycle: int
    reviewer_tool_calls: int


class ReviewOutput(BaseModel):
    """The reviewer's structured verdict (``response_format`` schema)."""

    approved: bool
    feedback: str


_FENCE_OPEN = re.compile(r"^```[a-zA-Z]*\s*")
_FENCE_CLOSE = re.compile(r"```\s*$")

_VERDICT_PROMPT = (
    "Based on your review, return your verdict now as a JSON object with "
    'exactly two fields: "approved" (boolean) and "feedback" (string). '
    "Approve only if accuracy, coverage, and actionability all hold. On "
    "rejection, the feedback must name what is missing or wrong and which "
    "angle to pursue next."
)

_VERDICT_RETRY = (
    "\nYour previous reply was not valid JSON with boolean \"approved\" and "
    "string \"feedback\". Return ONLY that JSON object, nothing else."
)

_UNPARSEABLE_FEEDBACK = (
    "The reviewer could not produce a valid verdict this cycle. Improve the "
    "report's evidence citations and completeness, then resubmit."
)


def parse_verdict(text: str) -> "ReviewOutput | None":
    """Parse the model's verdict text; ``None`` when it does not validate.

    Tolerates markdown code fences - local models add them even when told
    not to.
    """
    cleaned = _FENCE_CLOSE.sub("", _FENCE_OPEN.sub("", (text or "").strip())).strip()
    try:
        return ReviewOutput.model_validate_json(cleaned)
    except ValidationError:
        return None


# Executor/handler imports live below the pure helpers so tests of the pure
# parts stay importable even if the framework changes.
from agent_framework import Executor, WorkflowBuilder, WorkflowContext, handler  # noqa: E402


class ReviewerExecutor(Executor):
    """Independent verification: explore, then emit a structured verdict.

    Two calls in ONE session: the exploration turn builds context (read the
    report, spot-check sources), the verdict turn extracts the boolean +
    feedback under ``response_format``. Splitting them keeps schema forcing
    away from the tool-calling turn - local models handle each half better
    than both at once.
    """

    def __init__(self, agent_factory, id: str = "reviewer") -> None:
        """Args:
            agent_factory: Zero-arg callable returning ``(agent, budget)``
                fresh for this turn; the agent carries the corpus read tools,
                ``read_report``, and the budget middleware.
            id: Workflow node id.
        """
        super().__init__(id=id)
        self._agent_factory = agent_factory

    @handler
    async def on_draft(self, draft: DraftReady, ctx: WorkflowContext[ReviewVerdict]) -> None:
        agent, budget = self._agent_factory()
        session = agent.create_session()
        print(f"  [reviewer] cycle {draft.cycle}: verifying against the corpus...")
        await agent.run(
            render_reviewer_prompt(topic=draft.topic, cycle=draft.cycle),
            session=session,
        )
        verdict = None
        prompt = _VERDICT_PROMPT
        for _ in range(2):  # one attempt + one retry
            result = await agent.run(
                prompt, session=session,
                options={"response_format": ReviewOutput},
            )
            verdict = parse_verdict(result.text)
            if verdict is not None:
                break
            prompt = _VERDICT_PROMPT + _VERDICT_RETRY
        if verdict is None:
            # Fail closed: an unverifiable draft must never ship as approved.
            verdict = ReviewOutput(approved=False, feedback=_UNPARSEABLE_FEEDBACK)
        print(f"  [reviewer] cycle {draft.cycle}: "
              f"{'APPROVED' if verdict.approved else 'REJECTED'}")
        if not verdict.approved:
            print(f"  [reviewer] feedback: {verdict.feedback}")
        await ctx.send_message(ReviewVerdict(
            approved=verdict.approved, feedback=verdict.feedback,
            cycle=draft.cycle, reviewer_tool_calls=budget.spent,
        ))


_WRITE_REPORT_NUDGE = (
    "You finished without saving the report. Call the write_report tool NOW "
    "with the complete report markdown."
)


class WorkerExecutor(Executor):
    """Draft/revise/finalize the report; sole writer of the review log.

    Cycle state lives in executor memory - this demo has no checkpoint or
    resume, so in-memory counters are safe here (unlike the HOTL gate, whose
    progress must be ledger-derived).
    """

    def __init__(self, agent_factory, run_dir: Path, max_cycles: int,
                 id: str = "worker") -> None:
        """Args:
            agent_factory: ``factory(finalize: bool) -> (agent, budget, flag)``,
                fresh per turn. ``finalize=True`` must construct the agent
                with ``write_report`` only - stripping expressed at
                construction time.
            run_dir: This run's artifact directory (report + review log).
            max_cycles: Review-cycle budget; cycle ``max_cycles`` rejecting
                triggers the forced finalize.
            id: Workflow node id.
        """
        super().__init__(id=id)
        self._agent_factory = agent_factory
        self._report_path = run_dir / REPORT_FILENAME
        self._log_path = run_dir / LOG_FILENAME
        self._max_cycles = max_cycles
        self._topic = ""
        self._last_spent = 0

    @handler
    async def on_topic(self, topic: str, ctx: WorkflowContext[DraftReady]) -> None:
        """First draft, then hand to the reviewer."""
        self._topic = topic
        print("== cycle 1: drafting ==")
        await self._draft(render_worker_prompt(
            mode="initial", topic=topic, cycle=1, max_cycles=self._max_cycles))
        await ctx.send_message(DraftReady(cycle=1, topic=topic))

    @handler
    async def on_verdict(self, verdict: ReviewVerdict,
                         ctx: WorkflowContext[DraftReady, str]) -> None:
        """Approve -> ship; reject -> revise; budget exhausted -> forced finalize."""
        self._append_log({
            "cycle": verdict.cycle, "approved": verdict.approved,
            "feedback": verdict.feedback, "forced": False,
            "worker_tool_calls": self._last_spent,
            "reviewer_tool_calls": verdict.reviewer_tool_calls,
        })
        if verdict.approved:
            self._append_log({"outcome": "approved", "cycles": verdict.cycle,
                              "report": str(self._report_path)})
            await ctx.yield_output(
                f"Report approved after {verdict.cycle} cycle(s): {self._report_path}")
            return
        if verdict.cycle >= self._max_cycles:
            print("== review budget exhausted: forced finalize (read tools stripped) ==")
            await self._draft(render_worker_prompt(
                mode="finalize", topic=self._topic, cycle=verdict.cycle + 1,
                max_cycles=self._max_cycles, feedback=verdict.feedback,
                previous_report=self._previous_report()), finalize=True)
            self._append_log({
                "cycle": verdict.cycle + 1, "approved": False,
                "feedback": verdict.feedback, "forced": True,
                "worker_tool_calls": self._last_spent,
                "reviewer_tool_calls": None,
            })
            self._append_log({"outcome": "forced", "cycles": verdict.cycle,
                              "report": str(self._report_path)})
            await ctx.yield_output(
                f"Report shipped unapproved after {verdict.cycle} cycle(s) "
                f"(forced finalize): {self._report_path}")
            return
        next_cycle = verdict.cycle + 1
        print(f"== cycle {next_cycle}: revising ==")
        await self._draft(render_worker_prompt(
            mode="revision", topic=self._topic, cycle=next_cycle,
            max_cycles=self._max_cycles, feedback=verdict.feedback,
            previous_report=self._previous_report()))
        await ctx.send_message(DraftReady(cycle=next_cycle, topic=self._topic))

    async def _draft(self, prompt: str, finalize: bool = False) -> None:
        """One drafting turn: run the agent, ensure the report landed.

        One session covers the turn and its nudge retry, so the retry sees
        the exploration (``Agent.run(session=None)`` is stateless per call -
        same idiom and same reason as hotl_demo/phases.py).
        """
        agent, budget, flag = self._agent_factory(finalize)
        session = agent.create_session()
        draft = await agent.run(prompt, session=session)
        if not flag.written:
            retry = await agent.run(_WRITE_REPORT_NUDGE, session=session)
            if not flag.written:
                # Last resort: the turn's LONGEST text is the report - not the
                # latest. The model often emits the full report as plain chat
                # text in the draft run and answers the nudge with filler
                # ("Done."), which must not clobber the real content.
                text = max(((draft.text or "").strip(), (retry.text or "").strip()),
                           key=len) or "(no report produced)"
                atomic_write(self._report_path, text)
                print("  [worker] write_report never called - persisted final text instead")
        self._last_spent = budget.spent

    def _previous_report(self) -> str:
        """Prior report text for revision/finalize prompts (worker has no
        read access to the report file - the prompt carries it)."""
        if not self._report_path.exists():
            return "(no previous report)"
        return self._report_path.read_text(encoding="utf-8", errors="replace")

    def _append_log(self, record: dict) -> None:
        with self._log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


def build_reflexion_workflow(worker: WorkerExecutor, reviewer: ReviewerExecutor):
    """Assemble the cyclic worker <-> reviewer graph.

    Both edges are unconditional: each direction carries exactly one message
    type and each executor handles exactly that type, so isinstance dispatch
    does the routing.
    """
    return (
        WorkflowBuilder(start_executor=worker)
        .add_edge(worker, reviewer)
        .add_edge(reviewer, worker)
        .build()
    )
