import json
import threading

import pytest

from hotl_demo.artifacts import PHASES, REPOS, ArtifactStore


@pytest.fixture()
def store(tmp_path):
    return ArtifactStore(tmp_path / "run_x", repos=REPOS)


def test_initial_memory_shape(store):
    mem = store.read_memory()
    assert mem["run_id"] == "run_x"
    assert mem["review_completed"] is False
    assert set(mem["sections"]) == set(PHASES)
    assert set(mem["sections"]["deep_analysis"]) == set(REPOS)
    assert mem["sections"]["discovery"] == {}


def test_reopening_existing_run_dir_preserves_memory(store, tmp_path):
    store.update_memory("discovery", None, "purpose", "order management")
    again = ArtifactStore(tmp_path / "run_x", repos=REPOS)
    assert again.read_memory()["sections"]["discovery"]["purpose"] == "order management"


def test_update_memory_unit_nesting(store):
    store.update_memory("deep_analysis", "oms-monolith", "runtime", "Python 2.7")
    mem = store.read_memory()
    assert mem["sections"]["deep_analysis"]["oms-monolith"]["runtime"] == "Python 2.7"
    assert store.memory_key_count("deep_analysis", "oms-monolith") == 1
    assert store.memory_key_count("deep_analysis", "oms-batch-recon") == 0
    assert store.memory_key_count("discovery", None) == 0


def test_update_memory_rejects_unknown_phase_or_unit(store):
    with pytest.raises(KeyError):
        store.update_memory("nope", None, "k", "v")
    with pytest.raises(KeyError):
        store.update_memory("deep_analysis", "nope-repo", "k", "v")


def test_review_completed_flag(store):
    assert store.review_completed() is False
    store.set_review_completed()
    assert store.review_completed() is True


def test_raise_question_assigns_sequential_ids_and_appends(store):
    q1 = store.raise_question("discovery", None, "Scope?", "recon repo undocumented", "in scope")
    q2 = store.raise_question("deep_analysis", "oms-monolith", "RTO?", "not stated", "4h")
    assert (q1, q2) == ("q-1", "q-2")
    entries = store.read_ledger()
    assert [e["id"] for e in entries] == ["q-1", "q-2"]
    assert entries[0]["status"] == "open"
    assert entries[0]["unit"] is None
    assert entries[1]["unit"] == "oms-monolith"
    assert entries[1]["asked_at"]  # iso timestamp present
    # file is genuine JSONL
    lines = (store.run_dir / "ledger.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2 and json.loads(lines[0])["id"] == "q-1"


def test_open_questions_and_resolve(store):
    store.raise_question("discovery", None, "Scope?", "ctx", "in scope")
    store.raise_question("enterprise_context", None, "Region?", "ctx", "EU")
    resolved = store.resolve_question("q-1", "answered", "yes, in scope")
    assert resolved["human_answer"] == "yes, in scope"
    assert [e["id"] for e in store.open_questions()] == ["q-2"]
    store.resolve_question("q-2", "declined", None)
    assert store.open_questions() == []
    assert store.read_ledger()[1]["status"] == "declined"


def test_resolve_unknown_id_raises(store):
    with pytest.raises(KeyError):
        store.resolve_question("q-99", "answered", "x")


def test_concurrent_raises_get_unique_ids(store):
    ids: list[str] = []

    def worker(i: int) -> None:
        ids.append(store.raise_question("deep_analysis", REPOS[i % 2], f"Q{i}?", "ctx", "d"))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(set(ids)) == 20
    assert len(store.read_ledger()) == 20


def test_concurrent_memory_readers_and_writers(store):
    """Windows: an unlocked open colliding with a locked os.replace raises
    PermissionError; both paths must go through the store lock."""
    errors: list[Exception] = []

    def writer() -> None:
        try:
            for i in range(150):
                store.update_memory("deep_analysis", "oms-monolith", f"k{i}", "v")
        except Exception as exc:  # noqa: BLE001 - recorded for the assertion
            errors.append(exc)

    def reader() -> None:
        try:
            for _ in range(150):
                store.memory_key_count("deep_analysis", "oms-monolith")
                store.review_completed()
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=f) for f in (writer, writer, reader, reader)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert errors == []
    assert store.memory_key_count("deep_analysis", "oms-monolith") == 150


def test_reports_roundtrip(store):
    assert store.read_report("phase_01_discovery.md") == ""
    p = store.write_report("phase_01_discovery.md", "# Discovery\nfindings")
    assert p.read_text(encoding="utf-8").startswith("# Discovery")
    store.write_report("phase_03_enterprise_context.md", "ec")
    store.write_report("final_report.md", "final")  # not a phase report
    reports = store.read_all_reports()
    assert list(reports) == ["phase_01_discovery.md", "phase_03_enterprise_context.md"]
