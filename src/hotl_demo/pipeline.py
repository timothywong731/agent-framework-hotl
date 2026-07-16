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

from agent_framework import CheckpointStorage, Executor, WorkflowBuilder, WorkflowCheckpoint, WorkflowContext, handler

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
from .review import LedgerQuestionRequest, ReviewExecutor
from .tools import SCRATCHPAD_PATH

WORKFLOW_NAME = "hotl-migration-readiness"

# Every type that crosses the graph or sits in a pending request. Checkpoints
# are PICKLED behind this allowlist; a type missing here does not raise on
# load - list_checkpoints logs, skips the file, and returns [], which is
# indistinguishable from "no checkpoints exist". Derived from the classes so
# the module:qualname strings can never drift; test_checkpoint.py asserts
# completeness against every dataclass in phases.py/review.py.
_MESSAGE_TYPES = (PhaseDone, AnalysisDone, RevisionDone, RevisionTrigger,
                  ReportTrigger, LedgerQuestionRequest)
ALLOWED_CHECKPOINT_TYPES = [f"{t.__module__}:{t.__qualname__}" for t in _MESSAGE_TYPES]


def gate_checkpoint(checkpoints: list[WorkflowCheckpoint]) -> WorkflowCheckpoint | None:
    """Select the review-gate pause point from a run's checkpoints.

    The gate checkpoint is BY DEFINITION the one idle with pending
    ``request_info`` events; among those, the latest superstep wins.
    Never select positionally: ``list_checkpoints`` globs UUID filenames,
    so its order is meaningless - resuming from an arbitrary "latest" was
    measured to skip the human entirely (the file-backed review_completed
    latch told the re-entered gate "already reviewed").

    Args:
        checkpoints: Whatever ``storage.list_checkpoints`` returned.

    Returns:
        The gate checkpoint, or ``None`` when the run never paused (or every
        checkpoint file failed to decode - see ALLOWED_CHECKPOINT_TYPES).

    Example:
        >>> gate_checkpoint([]) is None
        True
    """
    pending = [c for c in checkpoints if c.pending_request_info_events]
    return max(pending, key=lambda c: c.iteration_count) if pending else None


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
                   scratchpad_path: Path = SCRATCHPAD_PATH,
                   checkpoint_storage: CheckpointStorage | None = None,
                   max_questions: int = 3):
    """Assemble the full HOTL workflow graph.

    Args:
        store: The run's shared artifact store, injected into every executor.
        base_dir: Sample-data root passed to :func:`build_phase_specs`.
        scratchpad_path: Steering file handed to every phase agent's tools.
        checkpoint_storage: When provided, the framework checkpoints every
            superstep into it (the --pause/--resume flows). ``None`` - the
            default and the interactive path - changes nothing.
        max_questions: Review-gate slot budget (``0`` = defer everything and
            never pause); threaded to :class:`ReviewExecutor`.

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
    review = ReviewExecutor(store, revision_order=[(s.name, s.unit) for s in specs],
                            max_questions=max_questions)
    report = FinalReportExecutor(store)

    builder = WorkflowBuilder(name=WORKFLOW_NAME, start_executor=discovery,
                              checkpoint_storage=checkpoint_storage)
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
