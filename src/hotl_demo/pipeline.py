"""Workflow graph assembly. Message TYPES encode routing; edges carry conditions.

The shape (see the README's mermaid rendering):

* initial flow: ``discovery`` fans out to one analyzer per repo (they run in
  the same superstep, i.e. concurrently); analyzers converge on ``join``;
  then ``enterprise_context`` -> ``questionnaire`` -> ``review`` ->
  ``final_report``.
* revision flow: ``review`` targets exactly one ``(phase, unit)`` at a time
  with a ``RevisionTrigger``; the revised phase answers straight back to
  ``review`` with a ``RevisionDone``, bypassing the join and the forward
  chain entirely.

Because initial completions (``PhaseDone``/``AnalysisDone``) and revision
completions (``RevisionDone``) are different TYPES, every edge condition is a
plain ``isinstance`` check and the join can never be polluted by re-runs.
"""
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
    """Edge-condition factory: pass only messages of one type.

    Args:
        message_type: The message class to admit.

    Returns:
        A predicate suitable for ``WorkflowBuilder.add_edge(condition=...)``.

    Example:
        >>> is_type(PhaseDone)(PhaseDone("discovery"))
        True
        >>> is_type(PhaseDone)(RevisionDone("discovery"))
        False
    """
    return lambda message: isinstance(message, message_type)


def revision_for(phase: str, unit: str | None) -> Callable[[Any], bool]:
    """Edge-condition factory: pass only the ``RevisionTrigger`` for one target.

    The review gate has an outbound edge per phase executor; these predicates
    make each trigger reach exactly its own target.

    Args:
        phase: Target phase name.
        unit: Target repo for deep_analysis, else ``None``.

    Returns:
        A predicate matching only that target's revision trigger.

    Example:
        >>> revision_for("deep_analysis", "oms-monolith")(
        ...     RevisionTrigger("deep_analysis", "oms-monolith", []))
        True
    """
    return lambda m: isinstance(m, RevisionTrigger) and m.phase == phase and m.unit == unit


class JoinAnalyses(Executor):
    """Fan-in barrier: wait for every repo analyzer, then advance the chain.

    Initial mode only - revision completions are typed ``RevisionDone`` and
    never routed here, so the barrier cannot be confused by re-runs. A
    self-owned 4-liner instead of the framework's fan-in edge group: its
    barrier semantics under mixed message types are ours to guarantee and
    unit-test.
    """

    def __init__(self, expected: int) -> None:
        """Set the barrier width.

        Args:
            expected: Number of distinct analyzer units to wait for.
        """
        super().__init__(id="join")
        self._expected = expected
        self._seen: set[str] = set()

    @handler
    async def on_analysis(self, done: AnalysisDone, ctx: WorkflowContext[PhaseDone]) -> None:
        """Collect one analyzer completion; release once all repos reported.

        Args:
            done: The analyzer's completion, carrying its repo name.
            ctx: Workflow context; emits the chain-advancing ``PhaseDone``.
        """
        # A set keyed by unit: duplicate completions from one repo (never
        # expected, but harmless) cannot release the barrier early.
        self._seen.add(done.unit)
        if len(self._seen) == self._expected:
            await ctx.send_message(PhaseDone("deep_analysis"))


def build_workflow(store: ArtifactStore, base_dir: Path,
                   scratchpad_path: Path = SCRATCHPAD_PATH):
    """Assemble the full HOTL workflow graph.

    Args:
        store: The run's shared artifact store, injected into every executor.
        base_dir: Sample-data root passed to :func:`build_phase_specs`.
        scratchpad_path: Steering file handed to every phase agent's tools.

    Returns:
        The built (not yet running) agent-framework workflow. Drive it with
        ``workflow.run("start", stream=True)`` and resume the review pause
        with ``workflow.run(stream=True, responses={...})``.

    Example:
        >>> workflow = build_workflow(store, Path("sample_data"))  # doctest: +SKIP
    """
    specs = build_phase_specs(base_dir)
    phase_execs = {s.executor_id: PhaseExecutor(s, store, scratchpad_path) for s in specs}
    discovery = phase_execs["discovery"]
    analyzers = [phase_execs[f"analyze:{repo}"] for repo in REPOS]
    enterprise = phase_execs["enterprise_context"]
    questionnaire = phase_execs["questionnaire"]
    join = JoinAnalyses(expected=len(analyzers))
    # revision_order preserves spec order so re-runs happen in pipeline order.
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
