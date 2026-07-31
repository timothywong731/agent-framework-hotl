"""Prompt rendering: both templates render, and carry their contracts."""
from reflection_demo.prompting import render_finalize_message, render_judge_instructions, render_worker_prompt

TOPIC = "Assess migrating OMS file storage from NFS to S3."


def test_worker_prompt_carries_topic_and_delivery_contract():
    out = render_worker_prompt(topic=TOPIC, max_passes=3, max_tool_calls=12)
    assert TOPIC in out
    assert "write_report" in out
    assert "list_files" in out and "read_file" in out
    assert "3" in out


def test_worker_prompt_has_no_revision_variant():
    # One template, one variant: the loop injects feedback, the prompt does not.
    out = render_worker_prompt(topic=TOPIC, max_passes=2, max_tool_calls=12)
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


def test_finalize_message_states_it_is_the_last_pass_and_demands_delivery():
    out = render_finalize_message(topic=TOPIC, max_passes=3)
    assert TOPIC in out
    assert "write_report" in out
    assert "3" in out
    lowered = out.lower()
    assert "final" in lowered
    assert "no further review" in lowered


def test_finalize_message_warns_that_one_read_call_closes_exploration():
    """The only prose in either package describing the finalize-pass strip.

    ``remove_tools`` is reachable only from inside a tool call, so the final
    pass cannot be stripped pre-emptively - it strips after the first one. This
    sentence is what stops that from reading as a malfunction, and is the only
    guard against the prompt and the middleware drifting apart.
    """
    out = " ".join(render_finalize_message(topic=TOPIC, max_passes=3).split())
    assert ("Do not start new exploration: if you call a read tool on this "
            "pass, your exploration tools close immediately afterwards.") in out


def test_finalize_message_fully_renders():
    out = render_finalize_message(topic=TOPIC, max_passes=2)
    assert "{{" not in out and "{%" not in out


def test_worker_prompt_states_the_tool_budget():
    out = render_worker_prompt(topic=TOPIC, max_passes=3, max_tool_calls=12)
    assert "12" in out
    # The old "There is no tool budget here" claim must be gone.
    assert "no tool budget" not in out.lower()
