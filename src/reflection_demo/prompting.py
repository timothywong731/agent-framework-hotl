"""Prompt rendering: Jinja2 templates in ``prompts/``, one per participant.

Unlike the reflexion worker there is no revision or finalize variant: the
loop middleware injects the judge's feedback as the next pass's input, so
the worker prompt is rendered exactly once per run.
"""
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

PROMPTS_DIR = Path(__file__).parent / "prompts"
_ENV = Environment(loader=FileSystemLoader(str(PROMPTS_DIR)), keep_trailing_newline=True)


def render_worker_prompt(*, topic: str, max_passes: int) -> str:
    """Render the worker's single prompt for the whole run.

    Args:
        topic: The migration topic under assessment.
        max_passes: The loop cap, for the model's situational awareness.
    """
    return _ENV.get_template("worker.md").render(
        topic=topic, max_passes=max_passes).strip()


def render_judge_instructions(*, topic: str) -> str:
    """Render the judge's system instructions.

    The rubric is deliberately identical to the reflexion reviewer's. If the
    two critics were given different standards the A/B would confound two
    variables; only the evidence channel may differ.
    """
    return _ENV.get_template("judge.md").render(topic=topic).strip()
