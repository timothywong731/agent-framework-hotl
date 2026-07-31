"""Prompt rendering: the three worker variants and the reviewer brief."""
import pytest

from reflexion_demo.prompting import render_reviewer_prompt, render_worker_prompt


def test_initial_variant_explores_and_delivers_via_write_report():
    text = render_worker_prompt(mode="initial", topic="NFS to S3", cycle=1,
                                max_cycles=3, max_tool_calls=12)
    assert "NFS to S3" in text
    assert "write_report" in text
    assert "read_file" in text            # told to explore
    assert "cycle 1 of at most 3" in text
    assert "REJECTED" not in text         # no revision leakage
    assert "12 tool calls" in text        # the number, as the reflection worker gets


def test_revision_variant_carries_feedback_and_previous_report():
    text = render_worker_prompt(
        mode="revision", topic="NFS to S3", cycle=2, max_cycles=3, max_tool_calls=12,
        feedback="Missing the Azure mandate conflict.",
        previous_report="# Draft 1 with {braces}",
    )
    assert "Missing the Azure mandate conflict." in text
    assert "# Draft 1 with {braces}" in text   # Jinja2 leaves literal braces alone
    assert "REJECTED" in text
    assert "12 tool calls" in text        # coached same as the initial turn


def test_finalize_variant_says_tools_are_gone():
    text = render_worker_prompt(
        mode="finalize", topic="NFS to S3", cycle=4, max_cycles=3, max_tool_calls=12,
        feedback="Still missing residency analysis.", previous_report="# Draft 3",
    )
    assert "reasoning for a long time" in text
    assert "exploration tools have been removed" in text
    assert "# Draft 3" in text
    assert "read_file" not in text        # must not tell it to explore
    assert "tool calls" not in text       # no read tools to budget


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        render_worker_prompt(mode="bogus", topic="t", cycle=1, max_cycles=3,
                             max_tool_calls=12)


def test_reviewer_prompt_demands_independent_verification():
    text = render_reviewer_prompt(topic="NFS to S3", cycle=2)
    assert "NFS to S3" in text
    assert "read_report" in text
    assert "read_file" in text            # spot-check against sources
    assert "cycle 2" in text
