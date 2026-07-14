"""Workflow graph assembly. Message TYPES encode routing; edges carry conditions."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from agent_framework import Executor, WorkflowBuilder, WorkflowContext, handler

from .artifacts import REPOS, ArtifactStore
from .phases import (
    AnalysisDone,
    PhaseDone,
    PhaseExecutor,
    ReportTrigger,
    RevisionDone,
    RevisionTrigger,
    build_phase_specs,
)
from .report import FinalReportExecutor
from .review import ReviewExecutor
from .tools import SCRATCHPAD_PATH


def is_type(message_type: type) -> Callable[[Any], bool]:
    return lambda message: isinstance(message, message_type)


def revision_for(phase: str, unit: str | None) -> Callable[[Any], bool]:
    return lambda m: isinstance(m, RevisionTrigger) and m.phase == phase and m.unit == unit


class JoinAnalyses(Executor):
    """Fan-in: wait for every repo analyzer, then advance. Initial mode only -
    revision completions are typed RevisionDone and never routed here."""

    def __init__(self, expected: int) -> None:
        super().__init__(id="join")
        self._expected = expected
        self._seen: set[str] = set()

    @handler
    async def on_analysis(self, done: AnalysisDone, ctx: WorkflowContext[PhaseDone]) -> None:
        self._seen.add(done.unit)
        if len(self._seen) == self._expected:
            await ctx.send_message(PhaseDone("deep_analysis"))


def build_workflow(store: ArtifactStore, base_dir: Path,
                   scratchpad_path: Path = SCRATCHPAD_PATH):
    specs = build_phase_specs(base_dir)
    phase_execs = {s.executor_id: PhaseExecutor(s, store, scratchpad_path) for s in specs}
    discovery = phase_execs["discovery"]
    analyzers = [phase_execs[f"analyze:{repo}"] for repo in REPOS]
    enterprise = phase_execs["enterprise_context"]
    questionnaire = phase_execs["questionnaire"]
    join = JoinAnalyses(expected=len(analyzers))
    review = ReviewExecutor(store, revision_order=[(s.name, s.unit) for s in specs])
    report = FinalReportExecutor(store)

    builder = WorkflowBuilder(start_executor=discovery)
    # initial forward flow
    for analyzer in analyzers:
        builder.add_edge(discovery, analyzer, condition=is_type(PhaseDone))
        builder.add_edge(analyzer, join, condition=is_type(AnalysisDone))
    builder.add_edge(join, enterprise, condition=is_type(PhaseDone))
    builder.add_edge(enterprise, questionnaire, condition=is_type(PhaseDone))
    builder.add_edge(questionnaire, review)  # carries PhaseDone AND RevisionDone
    # revision completions back to review (bypassing join / forward chain)
    builder.add_edge(discovery, review, condition=is_type(RevisionDone))
    for analyzer in analyzers:
        builder.add_edge(analyzer, review, condition=is_type(RevisionDone))
    builder.add_edge(enterprise, review, condition=is_type(RevisionDone))
    # review dispatches revisions to exactly one target, and finally the report
    for spec in specs:
        builder.add_edge(review, phase_execs[spec.executor_id],
                         condition=revision_for(spec.name, spec.unit))
    builder.add_edge(review, report, condition=is_type(ReportTrigger))
    return builder.build()
