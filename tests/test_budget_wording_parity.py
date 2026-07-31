"""The two demos' budget wording must not drift.

Both packages are standalone by design, so the budget logic is duplicated
rather than shared. Drift is that duplication's only real failure mode: if one
worker is coached differently from the other, evidence-gathering differs for a
reason unrelated to the critic - a confound in the A/B the demos exist to run.

A test importing both packages does not violate the standalone rule; that
rule binds package code. Precedent: tests/test_reflection_main.py imports
reflexion_demo.main.DEFAULT_TOPIC to pin topic identity.
"""
from reflection_demo import budget as reflection_budget
from reflection_demo import prompting as reflection_prompting
from reflexion_demo import budget as reflexion_budget
from reflexion_demo import prompting as reflexion_prompting

TOPIC = "Assess migrating OMS file storage from NFS to S3."
MAX_TOOL_CALLS = 12


def _budget_paragraph(prompt: str):
    """The rendered paragraph stating the tool budget, or ``None``.

    Paragraph-level, not constant-level: the constants-only tests below passed
    for the whole life of the drift they were meant to catch - reflection's
    worker was told "{{ max_tool_calls }} tool calls per pass ... spend them on
    the gaps that matter" while reflexion's was told only "a limited number",
    which is the same coaching asymmetry the constants exist to prevent, one
    layer up.
    """
    found = [p.strip() for p in prompt.split("\n\n") if "tool calls" in p]
    assert len(found) <= 1, "more than one budget paragraph - the extractor is lying"
    return found[0] if found else None


def test_budget_paragraph_identical_in_both_worker_prompts():
    reflection = _budget_paragraph(reflection_prompting.render_worker_prompt(
        topic=TOPIC, max_passes=3, max_tool_calls=MAX_TOOL_CALLS))
    # Not None first: two prompts that both stopped mentioning the budget
    # would satisfy the equality below while coaching nobody.
    assert reflection is not None
    assert str(MAX_TOOL_CALLS) in reflection
    for mode in ("initial", "revision"):
        reflexion = _budget_paragraph(reflexion_prompting.render_worker_prompt(
            mode=mode, topic=TOPIC, cycle=2, max_cycles=3,
            max_tool_calls=MAX_TOOL_CALLS, feedback="thin on residency",
            previous_report="# Draft 1"))
        assert reflexion == reflection, f"{mode} worker prompt drifted"


def test_the_reflexion_finalize_variant_claims_no_budget():
    """That agent is constructed with write_report only - it has no read tools
    to budget, so promising it calls it cannot make would be a lie."""
    assert _budget_paragraph(reflexion_prompting.render_worker_prompt(
        mode="finalize", topic=TOPIC, cycle=4, max_cycles=3,
        max_tool_calls=MAX_TOOL_CALLS, feedback="still thin",
        previous_report="# Draft 3")) is None


def test_countdown_wording_identical_across_demos():
    assert reflection_budget.COUNTDOWN == reflexion_budget.COUNTDOWN


def test_budget_spent_wording_identical_across_demos():
    assert reflection_budget.BUDGET_SPENT == reflexion_budget.BUDGET_SPENT


def test_both_demos_exempt_write_report_from_the_budget():
    assert "write_report" not in reflection_budget.BUDGETED_TOOL_NAMES
    assert "write_report" not in reflexion_budget.BUDGETED_TOOL_NAMES
