"""Checkpointing: gate-checkpoint selection, allowlist completeness, pause/resume."""
import dataclasses
import json

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


from pathlib import Path

from agent_framework import FileCheckpointStorage

from conftest import DRIVE_TARGETS, DriveAgent

from hotl_demo.artifacts import REPOS, ArtifactStore
from hotl_demo.main import map_answers, parse_review_answers, render_review_lines
from hotl_demo.pipeline import build_workflow
from hotl_demo.review import LedgerQuestionRequest


@pytest.mark.asyncio
async def test_pause_resume_cycle_revises_each_target_exactly_once(tmp_path, monkeypatch):
    """The spike that found the 5x bug, as a permanent regression guard.

    Process 1 runs to the gate with checkpointing on. Process 2 is simulated
    with a FRESH store and a FRESH workflow over the same run dir: restore the
    gate checkpoint, answer everything via the review.jsonl round-trip, and
    assert the resumed gate dispatches ONE ordered revision queue - not one
    per answer.
    """
    from hotl_demo import phases, report, review
    from hotl_demo.pipeline import (ALLOWED_CHECKPOINT_TYPES, WORKFLOW_NAME,
                                    gate_checkpoint)

    monkeypatch.setenv("OLLAMA_MODEL", "test-model")
    run_dir, cp_dir, data = tmp_path / "run", tmp_path / "checkpoints", Path("sample_data")
    storage = FileCheckpointStorage(cp_dir, allowed_checkpoint_types=ALLOWED_CHECKPOINT_TYPES)

    def patch_agents(store, calls):
        def agent_factory(*_, name="", **__):
            return DriveAgent(name, store, calls)
        for mod in (phases, report, review):
            monkeypatch.setattr(mod, "Agent", agent_factory)
            monkeypatch.setattr(mod, "OllamaChatClient", lambda: None)

    # ---- process 1: run to the gate, checkpointing on ----
    store1, calls1 = ArtifactStore(run_dir, repos=REPOS), []
    patch_agents(store1, calls1)
    wf1 = build_workflow(store1, data, scratchpad_path=tmp_path / "pad.md",
                         checkpoint_storage=storage)
    pending1 = {}
    async for ev in wf1.run("start", stream=True):
        if ev.type == "request_info" and isinstance(ev.data, LedgerQuestionRequest):
            pending1[ev.request_id] = ev.data
    # 5 raised; reversed-order ranking presents the LAST three; 2 deferred
    assert {q.question_id for q in pending1.values()} == {"q-3", "q-4", "q-5"}

    # the pause artifact: seeded with EXACTLY the presented set
    sheet = render_review_lines(store1.open_questions())
    assert [json.loads(l)["id"] for l in sheet.splitlines()] == ["q-3", "q-4", "q-5"]
    answered = "".join(
        line.replace('"answer": ""', f'"answer": "answer to {json.loads(line)["id"]}"') + "\n"
        for line in sheet.splitlines()
    )
    answers = parse_review_answers(answered)
    # decline-by-omission (q-5 is deterministically questionnaire's - it raises
    # last) and an unknown id, exercising the sheet's edge semantics end to end
    del answers["q-5"]
    answers["q-99"] = "ghost"

    # ---- process 2: fresh store, fresh workflow, same run dir ----
    store2, calls2 = ArtifactStore(run_dir, repos=REPOS), []
    patch_agents(store2, calls2)
    wf2 = build_workflow(store2, data, scratchpad_path=tmp_path / "pad.md",
                         checkpoint_storage=storage)
    gate = gate_checkpoint(await storage.list_checkpoints(workflow_name=WORKFLOW_NAME))
    assert gate is not None

    pending2 = {}
    async for ev in wf2.run(checkpoint_id=gate.checkpoint_id, stream=True):
        if ev.type == "request_info" and isinstance(ev.data, LedgerQuestionRequest):
            pending2[ev.request_id] = ev.data
    assert set(pending2) == set(pending1)      # stable request ids
    assert calls2 == []                        # restore re-ran ZERO phases (and no re-rank)

    outputs = []
    responses, unknown = map_answers(pending2, answers)
    assert unknown == ["q-99"]                 # the ghost id surfaced, not applied
    async for ev in wf2.run(stream=True, responses=responses):
        assert ev.type != "request_info"       # review-once survives the restart
        if ev.type == "output":
            outputs.append(ev.data)

    # one ordered queue; expectations DERIVED (analyzer attribution is racy);
    # q-5 declined by omission means the questionnaire never revises
    ledger = store2.read_ledger()
    answered_targets = {(e["phase"], e["unit"]) for e in ledger if e["status"] == "answered"}
    expected = [n for n, t in DRIVE_TARGETS.items() if t in answered_targets]
    assert [name for name, kind, _ in calls2 if kind == "revision"] == expected
    assert [name for name, kind, _ in calls2 if kind == "rank"] == []   # gate never re-ranks
    assert [name for name, kind, _ in calls2 if kind == "report"] == ["final_report"]
    assert outputs == [str(store2.run_dir / "final_report.md")]
    assert [e["status"] for e in ledger] == (
        ["deferred", "deferred", "answered", "answered", "declined"])
