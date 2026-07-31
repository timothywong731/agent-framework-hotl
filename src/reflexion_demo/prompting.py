"""Prompt rendering: Jinja2 templates in ``prompts/``, one per agent.

The worker template carries all three variants (initial/revision/finalize)
selected by ``mode`` - explicit variant files would repeat the shared
delivery contract three times.
"""
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

PROMPTS_DIR = Path(__file__).parent / "prompts"
_ENV = Environment(loader=FileSystemLoader(str(PROMPTS_DIR)), keep_trailing_newline=True)

_WORKER_MODES = ("initial", "revision", "finalize")


def render_worker_prompt(*, mode: str, topic: str, cycle: int, max_cycles: int,
                         max_tool_calls: int, feedback: str = "",
                         previous_report: str = "") -> str:
    """Render the worker's turn prompt.

    Args:
        mode: ``"initial"``, ``"revision"``, or ``"finalize"``.
        topic: The migration topic under assessment.
        cycle: 1-based draft number this turn produces.
        max_cycles: The review-cycle budget (for the model's situational
            awareness).
        max_tool_calls: The read-tool budget this turn runs under. Stated as a
            number, in a paragraph byte-identical to the reflection worker's:
            a worker that knows it has 12 can plan a 12-file sweep and one
            told only "a limited number" cannot, so naming it for one demo and
            not the other would make evidence-gathering differ for a reason
            unrelated to the critic. Required rather than defaulted - a
            plausible-looking default would render a silently wrong number.
            The ``finalize`` variant ignores it: that agent is constructed
            with ``write_report`` only, so it has no budget to spend.
        feedback: Reviewer feedback (revision/finalize only).
        previous_report: Prior report text (revision/finalize only).

    Raises:
        ValueError: Unknown ``mode`` - programmer error, fail loud.
    """
    if mode not in _WORKER_MODES:
        raise ValueError(f"unknown worker mode: {mode!r}")
    return _ENV.get_template("worker.md").render(
        mode=mode, topic=topic, cycle=cycle, max_cycles=max_cycles,
        max_tool_calls=max_tool_calls, feedback=feedback,
        previous_report=previous_report,
    ).strip()


def render_reviewer_prompt(*, topic: str, cycle: int) -> str:
    """Render the reviewer's evaluation brief for one cycle."""
    return _ENV.get_template("reviewer.md").render(topic=topic, cycle=cycle).strip()
