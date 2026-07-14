"""Human-on-the-loop pipeline demo on Microsoft Agent Framework.

A multi-phase agent pipeline (discovery -> per-repo deep analysis ->
enterprise context -> questionnaire) that accumulates a ledger of questions,
pauses exactly once at a human review gate, selectively re-runs affected
phases with the human's answers, and writes a final readiness report.
See ``docs/superpowers/specs/`` for the full design.
"""
