"""Reflexion workflow: messages, verdict schema, and the two executors.

Routing convention (same as the HOTL pipeline): message TYPES encode meaning.
``DraftReady`` only ever flows worker -> reviewer; ``ReviewVerdict`` only ever
flows reviewer -> worker. No mode flags anywhere.

This module deliberately avoids ``from __future__ import annotations``: the
framework inspects handler signatures, and string annotations are a known
trap (see the warning at the top of hotl_demo/review.py).
"""
import re
from dataclasses import dataclass

from pydantic import BaseModel, ValidationError

from .prompting import render_reviewer_prompt

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
from agent_framework import Executor, WorkflowContext, handler  # noqa: E402


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
