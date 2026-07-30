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
from reflexion_demo import budget as reflexion_budget


def test_countdown_wording_identical_across_demos():
    assert reflection_budget.COUNTDOWN == reflexion_budget.COUNTDOWN


def test_budget_spent_wording_identical_across_demos():
    assert reflection_budget.BUDGET_SPENT == reflexion_budget.BUDGET_SPENT


def test_both_demos_exempt_write_report_from_the_budget():
    assert "write_report" not in reflection_budget.BUDGETED_TOOL_NAMES
    assert "write_report" not in reflexion_budget.BUDGETED_TOOL_NAMES
