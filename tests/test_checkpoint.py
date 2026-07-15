"""Checkpointing: gate-checkpoint selection, allowlist completeness, pause/resume."""
import dataclasses

import pytest

from agent_framework import WorkflowCheckpoint

from hotl_demo.pipeline import ALLOWED_CHECKPOINT_TYPES, WORKFLOW_NAME, gate_checkpoint


def _cp(iteration_count, pending_ids=()):
    """Build a real WorkflowCheckpoint with only the fields selection reads."""
    return WorkflowCheckpoint(
        workflow_name=WORKFLOW_NAME,
        graph_signature_hash="test-hash",
        iteration_count=iteration_count,
        pending_request_info_events={rid: object() for rid in pending_ids},
    )


def test_gate_checkpoint_picks_pending_with_highest_iteration():
    # list_checkpoints is glob-ordered (UUID filenames), so feed a shuffled
    # list: selection must be semantic, never positional.
    cps = [_cp(9), _cp(4, pending_ids=("r1",)), _cp(2), _cp(6, pending_ids=("r1", "r2"))]
    assert gate_checkpoint(cps) is cps[3]      # pending beats higher bare iteration
    assert gate_checkpoint(list(reversed(cps))) is cps[3]


def test_gate_checkpoint_none_when_nothing_pending():
    assert gate_checkpoint([]) is None
    assert gate_checkpoint([_cp(1), _cp(7)]) is None


def test_allowlist_covers_every_message_dataclass():
    # Trap: checkpoints are pickled behind this allowlist, and a missing type
    # makes list_checkpoints silently return [] - resume just "stops working".
    # Force a conscious decision whenever a dataclass is added.
    from hotl_demo import phases, review
    found = set()
    for mod in (phases, review):
        for name in dir(mod):
            obj = getattr(mod, name)
            if (isinstance(obj, type) and dataclasses.is_dataclass(obj)
                    and obj.__module__ == mod.__name__):
                found.add(f"{obj.__module__}:{obj.__qualname__}")
    non_messages = {"hotl_demo.phases:PhaseSpec"}   # executor config, never routed
    assert found - non_messages == set(ALLOWED_CHECKPOINT_TYPES)
