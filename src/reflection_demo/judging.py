"""The tool-less judge: verdict reading, the run log, and the loop predicate.

This module deliberately avoids ``from __future__ import annotations`` - the
framework introspects callables handed to it, and string annotations are a
known trap in this repo.
"""
import json
from dataclasses import dataclass
from pathlib import Path

from agent_framework import JudgeVerdict

# The framework's own fallback markers, reused verbatim so a judge that
# cannot honour response_format still lands on the same contract.
JUDGE_VERDICT_DONE = "VERDICT: DONE"
JUDGE_VERDICT_MORE = "VERDICT: MORE"


@dataclass
class Verdict:
    """One judged pass."""

    pass_no: int
    answered: bool
    reasoning: str


def read_verdict(value, text) -> tuple:
    """Normalize the judge's reply to ``(answered, reasoning)``.

    ``value`` is the parsed :class:`JudgeVerdict` when the client honoured
    ``response_format``; otherwise fall back to the explicit markers, with
    ``MORE`` winning whenever the reply is ambiguous or marker-less.

    This fails OPEN - an unreadable verdict keeps the loop running and costs
    one pass. That is the opposite of the reflexion reviewer, which fails
    CLOSED and rejects. Both are right for their pattern: reflexion must
    never ship unverified work as approved, whereas here the pass cap is
    what guarantees termination.

    Args:
        value: ``ChatResponse.value`` - a ``JudgeVerdict`` or anything else.
        text: ``ChatResponse.text`` - the raw reply, used for the fallback.
    """
    if isinstance(value, JudgeVerdict):
        return value.answered, value.reasoning
    raw = (text or "").strip()
    upper = raw.upper()
    answered = False if JUDGE_VERDICT_MORE in upper else JUDGE_VERDICT_DONE in upper
    return answered, raw


def summarize(verdicts, max_passes: int) -> tuple:
    """Terminal outcome of a finished run: ``("answered"|"unjudged", passes)``.

    ``AgentLoopMiddleware`` checks ``max_iterations`` BEFORE evaluating
    ``should_continue`` (``agent_framework/_harness/_loop.py``), so the judge
    is never consulted on the capped pass: a capped run has one more pass
    than it has verdicts, and that last pass ships unjudged. The reflexion
    parallel is the forced finalize shipping unapproved.

    Args:
        verdicts: Every verdict recorded this run, in pass order.
        max_passes: The ``--max-passes`` cap the loop was built with.
    """
    if verdicts and verdicts[-1].answered:
        return "answered", len(verdicts)
    return "unjudged", max_passes


class RunLog:
    """Append-only ``reflection_log.jsonl``; the sole writer.

    Deliberately the same line shape as the reflexion demo's
    ``review_log.jsonl`` so the two runs can be read side by side.
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self.verdicts = []

    def record(self, verdict: Verdict) -> None:
        """Append one judged pass."""
        self.verdicts.append(verdict)
        self._append({"pass": verdict.pass_no, "answered": verdict.answered,
                      "reasoning": verdict.reasoning, "judged": True})

    def finish(self, max_passes: int, report_path: Path) -> tuple:
        """Write the unjudged pass (if any) and the outcome line.

        Returns:
            ``(outcome, passes)`` as computed by :func:`summarize`.
        """
        outcome, passes = summarize(self.verdicts, max_passes)
        if outcome == "unjudged":
            self._append({"pass": passes, "answered": None,
                          "reasoning": None, "judged": False})
        self._append({"outcome": outcome, "passes": passes,
                      "report": str(report_path)})
        return outcome, passes

    def _append(self, record: dict) -> None:
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
