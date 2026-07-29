"""Prompt rendering: both templates render, and carry their contracts."""
from reflection_demo.prompting import render_judge_instructions, render_worker_prompt

TOPIC = "Assess migrating OMS file storage from NFS to S3."


def test_worker_prompt_carries_topic_and_delivery_contract():
    out = render_worker_prompt(topic=TOPIC, max_passes=3)
    assert TOPIC in out
    assert "write_report" in out
    assert "list_files" in out and "read_file" in out
    assert "3" in out


def test_worker_prompt_has_no_revision_variant():
    # One template, one variant: the loop injects feedback, the prompt does not.
    out = render_worker_prompt(topic=TOPIC, max_passes=2)
    assert "{{" not in out and "{%" not in out


def test_judge_instructions_carry_the_reviewer_rubric():
    out = render_judge_instructions(topic=TOPIC)
    assert TOPIC in out
    # Rubric parity with the reflexion reviewer - only the evidence differs.
    for word in ("Accuracy", "Coverage", "Actionability"):
        assert word in out


def test_judge_instructions_carry_the_verdict_contract():
    out = render_judge_instructions(topic=TOPIC)
    assert "answered" in out and "reasoning" in out
    assert "VERDICT: DONE" in out and "VERDICT: MORE" in out


def test_judge_instructions_state_it_has_no_tools():
    out = render_judge_instructions(topic=TOPIC)
    assert "no tools" in out.lower()
