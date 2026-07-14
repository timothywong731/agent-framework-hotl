"""Core agent tools (scratchpad/ledger/memory binding) and repo-exploration tools."""
from pathlib import Path

import pytest

from hotl_demo.artifacts import REPOS, ArtifactStore
from hotl_demo.tools import SCRATCHPAD_PATH, ensure_scratchpad, make_repo_tools, make_tools

MONOLITH = Path("sample_data/repos/oms-monolith")


@pytest.fixture()
def store(tmp_path):
    return ArtifactStore(tmp_path / "run", repos=REPOS)


def _tools(store, tmp_path, phase="discovery", unit=None):
    pad = tmp_path / "scratchpad.md"
    read_scratchpad, raise_question, update_memory = make_tools(
        store, phase, unit, scratchpad_path=pad
    )
    return pad, read_scratchpad, raise_question, update_memory


def test_scratchpad_path_is_stable_repo_root():
    # CWD-independent: running the demo from any directory must find the same file.
    assert SCRATCHPAD_PATH.is_absolute()
    assert SCRATCHPAD_PATH == Path(__file__).resolve().parents[1] / "scratchpad.md"


def test_ensure_scratchpad_creates_but_never_truncates(tmp_path):
    pad = tmp_path / "scratchpad.md"
    ensure_scratchpad(pad)
    assert pad.exists() and pad.read_text(encoding="utf-8") == ""
    pad.write_text("steering note", encoding="utf-8")
    ensure_scratchpad(pad)
    assert pad.read_text(encoding="utf-8") == "steering note"


def test_read_scratchpad_missing_and_empty_and_content(store, tmp_path):
    pad, read_scratchpad, _, _ = _tools(store, tmp_path)
    assert "empty" in read_scratchpad().lower()          # missing file
    pad.write_text("   \n", encoding="utf-8")
    assert "empty" in read_scratchpad().lower()          # whitespace-only
    pad.write_text("Focus on the database.", encoding="utf-8")
    assert read_scratchpad() == "Focus on the database."


def test_raise_question_appends_with_phase_and_unit(store, tmp_path):
    _, _, raise_question, _ = _tools(store, tmp_path, phase="deep_analysis", unit="oms-monolith")
    out = raise_question("RTO?", "not stated in PDF 2", "assume 4h")
    assert "q-1" in out
    entry = store.read_ledger()[0]
    assert entry["phase"] == "deep_analysis"
    assert entry["unit"] == "oms-monolith"
    assert entry["default_assumption"] == "assume 4h"


def test_raise_question_validates_args(store, tmp_path):
    _, _, raise_question, _ = _tools(store, tmp_path)
    assert raise_question("", "ctx", "d").startswith("ERROR")
    assert raise_question("Q?", "ctx", "  ").startswith("ERROR")
    assert store.read_ledger() == []


def test_update_memory_bound_to_phase_and_unit(store, tmp_path):
    _, _, _, update_memory = _tools(store, tmp_path, phase="deep_analysis", unit="oms-batch-recon")
    out = update_memory("secrets", "hardcoded Oracle password in config.py")
    assert "secrets" in out
    mem = store.read_memory()
    assert mem["sections"]["deep_analysis"]["oms-batch-recon"]["secrets"].startswith("hardcoded")


def test_update_memory_validates_args(store, tmp_path):
    _, _, _, update_memory = _tools(store, tmp_path)
    assert update_memory(" ", "v").startswith("ERROR")
    assert update_memory("k", "").startswith("ERROR")
    assert store.memory_key_count("discovery", None) == 0


def test_repo_tools_list_and_read():
    list_files, read_file = make_repo_tools(MONOLITH)
    tree = list_files()
    assert "s3_uploader.py" in tree and "README.md" in tree
    assert "boto3" in read_file("s3_uploader.py")


def test_repo_tools_reject_traversal_and_missing():
    _, read_file = make_repo_tools(MONOLITH)
    assert read_file("../oms-batch-recon/config.py").startswith("ERROR")
    assert read_file("nope.py").startswith("ERROR")
