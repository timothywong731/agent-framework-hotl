"""Prompt rendering: Jinja2 templates in ``prompts/``, one per participant.

There is no revision variant - the loop middleware injects the judge's
feedback as the next pass's input, so the worker prompt is rendered once per
run. There IS a finalize message: the final pass gets a different instruction
(write now, no further review), delivered through ``next_message`` because the
worker prompt cannot be re-rendered mid-run.
"""
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

PROMPTS_DIR = Path(__file__).parent / "prompts"
_ENV = Environment(loader=FileSystemLoader(str(PROMPTS_DIR)), keep_trailing_newline=True)


def render_worker_prompt(*, topic: str, max_passes: int, max_tool_calls: int) -> str:
    """Render the worker's opening prompt for the whole run.

    Args:
        topic: The migration topic under assessment.
        max_passes: The loop cap, for the model's situational awareness.
        max_tool_calls: The per-pass tool budget, so the countdown lines that
            arrive later are expected rather than a surprise.
    """
    return _ENV.get_template("worker.md").render(
        topic=topic, max_passes=max_passes, max_tool_calls=max_tool_calls).strip()


def render_finalize_message(*, topic: str, max_passes: int) -> str:
    """Render the final pass's instruction.

    Delivered via ``next_message``, not as a re-render of the worker prompt:
    ``AgentLoopMiddleware`` owns the agent for the whole run, so the system
    prompt is fixed once and only the per-pass input can change.
    """
    return _ENV.get_template("finalize.md").render(
        topic=topic, max_passes=max_passes).strip()


def render_judge_instructions(*, topic: str) -> str:
    """Render the judge's system instructions.

    Coverage and Actionability are identical to the reflexion reviewer's.
    Accuracy is necessarily weaker: the judge has no corpus access and
    cannot verify that claims match their sources. This asymmetry is
    the documented difference the demo exists to demonstrate.
    """
    return _ENV.get_template("judge.md").render(topic=topic).strip()
