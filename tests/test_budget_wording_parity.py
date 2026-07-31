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


def _budget_paragraph(prompt: str, max_tool_calls: int = MAX_TOOL_CALLS):
    """The rendered paragraph stating the tool-call NUMBER, or ``None``.

    Paragraph-level, not constant-level: the constants-only tests below passed
    for the whole life of the drift they were meant to catch - reflection's
    worker was told "{{ max_tool_calls }} tool calls per pass ... spend them on
    the gaps that matter" while reflexion's was told only "a limited number",
    which is the same coaching asymmetry the constants exist to prevent, one
    layer up.

    Matches on the exact rendered opening ("You have N tool calls.") rather
    than a bare "tool calls" substring: each worker.md now ALSO carries a
    per-demo reset-boundary sentence next to this paragraph (see
    ``test_each_prompt_states_its_own_reset_boundary`` below), and a substring
    match would either swallow that sentence into "the" budget paragraph or
    silently start matching two paragraphs instead of one. Anchoring on the
    literal number keeps this test pinned to the paragraph it was written to
    guard, regardless of what prose sits next to it.
    """
    needle = f"You have {max_tool_calls} tool calls."
    found = [p.strip() for p in prompt.split("\n\n") if p.strip().startswith(needle)]
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
    prompt = reflexion_prompting.render_worker_prompt(
        mode="finalize", topic=TOPIC, cycle=4, max_cycles=3,
        max_tool_calls=MAX_TOOL_CALLS, feedback="still thin",
        previous_report="# Draft 3")
    assert _budget_paragraph(prompt) is None
    # It must not be told a reset boundary for a budget it does not have
    # either - that would coach it to plan around calls it cannot make.
    assert "renews at the start of every" not in prompt


def test_each_prompt_states_its_own_reset_boundary():
    """The shared paragraph above names no unit - "per pass" and "per turn"
    are the one fact that legitimately differs between the demos, so a
    byte-identical paragraph cannot carry both. Each worker.md instead states
    its own reset boundary, next to the shared paragraph, in its own unit:
    reflection's budget is minted per PASS (``PassBudget.start_pass``, called
    from ``next_message`` at every pass boundary); reflexion's is minted per
    TURN (a fresh ``ToolBudget`` built by the worker factory on every call).
    Same fact, different vocabulary - the sentences must therefore differ by
    exactly that word and no other.
    """
    reflection = reflection_prompting.render_worker_prompt(
        topic=TOPIC, max_passes=3, max_tool_calls=MAX_TOOL_CALLS)
    assert "renews at the start of every pass" in reflection
    assert "every turn" not in reflection, "reflection prompt borrowed reflexion's unit"

    for mode in ("initial", "revision"):
        reflexion = reflexion_prompting.render_worker_prompt(
            mode=mode, topic=TOPIC, cycle=2, max_cycles=3,
            max_tool_calls=MAX_TOOL_CALLS, feedback="thin on residency",
            previous_report="# Draft 1")
        assert "renews at the start of every turn" in reflexion, \
            f"{mode} worker prompt lost its reset boundary"
        assert "every pass" not in reflexion, \
            f"{mode} worker prompt borrowed reflection's unit"


def test_countdown_wording_identical_across_demos():
    assert reflection_budget.COUNTDOWN == reflexion_budget.COUNTDOWN


def test_budget_spent_wording_identical_across_demos():
    assert reflection_budget.BUDGET_SPENT == reflexion_budget.BUDGET_SPENT


def test_both_demos_exempt_write_report_from_the_budget():
    assert "write_report" not in reflection_budget.BUDGETED_TOOL_NAMES
    assert "write_report" not in reflexion_budget.BUDGETED_TOOL_NAMES
