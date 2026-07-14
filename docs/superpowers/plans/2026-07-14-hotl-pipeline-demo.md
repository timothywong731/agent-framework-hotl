# HOTL Pipeline Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the human-on-the-loop (HOTL) cloud-migration-readiness demo pipeline on Microsoft Agent Framework per `docs/superpowers/specs/2026-07-14-hotl-pipeline-design.md`.

**Architecture:** A `WorkflowBuilder` graph — discovery → (fan-out) per-repo deep_analysis analyzers → join → enterprise_context → questionnaire → review (human gate via `ctx.request_info`) → final_report — with revision edges from review back to every phase. Agents run on local Ollama (`gemma4:31b`) with exactly three tools (`read_scratchpad`, `raise_question`, `update_memory`); artifacts (`memory.json`, `ledger.jsonl`, phase reports) are file-backed through a thread-safe `ArtifactStore`. Message *types* encode mode: `PhaseDone`/`AnalysisDone` flow forward on initial runs, `RevisionDone` flows back to review, so no fan-in barrier ever sees a revision message.

**Tech Stack:** Python ≥3.10 (dev machine has 3.13), Poetry 2.x, pytest, `agent-framework ~=1.11`, `agent-framework-ollama ==1.0.0b260709`, `pypdf`, `fpdf2`.

## Global Constraints

- Python `>=3.10`; Poetry 2.x with PEP 621 `[project]` table; src layout (`src/hotl_demo/`).
- Runtime deps ONLY: `agent-framework (~=1.11)`, `agent-framework-ollama (==1.0.0b260709)`, `pypdf (>=5,<7)`, `fpdf2 (>=2.8,<3)`. Dev dep: `pytest (~=8.3)`. No rich/typer/click/dotenv — stdlib `argparse` and `print`.
- Model: `gemma4:31b` by default, always via the `OLLAMA_MODEL` env var + no-arg `OllamaChatClient()` (matches the official sample; never pass the model as a constructor kwarg).
- Agents get exactly three tools: `read_scratchpad`, `raise_question`, `update_memory`. PDFs/repo contents/template/ledger are pre-loaded into prompts, never fetched by tools.
- `scratchpad.md` at repo root: created empty if missing, NEVER truncated or overwritten if present.
- `ledger.jsonl` is append-only for raises; status changes rewrite the file atomically. All `memory.json`/`ledger.jsonl` writes go through one `threading.Lock` and `os.replace` (analyzers run concurrently).
- Review gate runs exactly once per run (`review_completed` flag in `memory.json`, set on gate entry).
- Sample docs are ASCII-only (fpdf2 core fonts are latin-1; no smart quotes/em-dashes in `docs_src`).
- Default test run is LLM-free: `pytest` config excludes the `ollama` marker; live E2E is opt-in via `OLLAMA_E2E=1`.
- Every commit message ends with:
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`
  `Claude-Session: https://claude.ai/code/session_014aYGSgoeKoLtSLC6YU2GgG`
- All commands below run from the repo root `C:\Users\Timothy Wong\Repositories\agent-framework-hotl`. They are written for PowerShell/Git Bash; `poetry run` works in both.

---

### Task 1: Project scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `src/hotl_demo/__init__.py`
- Create: `src/hotl_demo/main.py` (stub entry point only; real runner in Task 10)
- Create: `.gitignore`
- Create: `scratchpad.md` (empty)
- Create: `tests/.gitkeep` (empty dir marker; deliberately NO `tests/__init__.py` — pytest must import tests as top-level modules so `from conftest import ...` works)

**Interfaces:**
- Consumes: nothing (first task).
- Produces: installable package `hotl_demo`; `poetry run demo` entry point calling `hotl_demo.main:run`; pytest configured with the `ollama` marker excluded by default. Later tasks assume `poetry run pytest` works and `import hotl_demo` succeeds.

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "hotl-demo"
version = "0.1.0"
description = "Human-on-the-loop pipeline demo on Microsoft Agent Framework"
requires-python = ">=3.10"
dependencies = [
    "agent-framework (~=1.11)",
    "agent-framework-ollama (==1.0.0b260709)",
    "pypdf (>=5,<7)",
    "fpdf2 (>=2.8,<3)",
]

[project.scripts]
demo = "hotl_demo.main:run"

[tool.poetry]
packages = [{ include = "hotl_demo", from = "src" }]

[tool.poetry.group.dev.dependencies]
pytest = "~=8.3"

[build-system]
requires = ["poetry-core>=2.0"]
build-backend = "poetry.core.masonry.api"

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["ollama: live end-to-end test requiring a running Ollama server"]
addopts = "-m 'not ollama'"
```

- [ ] **Step 2: Write `.gitignore`**

```gitignore
__pycache__/
*.pyc
.pytest_cache/
dist/
output/
.venv/
```

Note: `scratchpad.md` is committed (empty); `output/` (run artifacts) is ignored.

- [ ] **Step 3: Write package skeleton**

`src/hotl_demo/__init__.py`:

```python
"""Human-on-the-loop pipeline demo on Microsoft Agent Framework."""
```

`src/hotl_demo/main.py` (stub — replaced in Task 10):

```python
def run() -> None:
    raise SystemExit("demo runner not implemented yet (see Task 10)")
```

`tests/.gitkeep`: empty file (do NOT create `tests/__init__.py`, now or in any later task).

`scratchpad.md`: empty file (0 bytes). Create it with `New-Item -ItemType File scratchpad.md` (PowerShell) or `touch scratchpad.md` (bash) — NOT with an editor that adds a newline; content must start empty.

- [ ] **Step 4: Install and verify**

Run: `poetry install`
Expected: resolves and installs `agent-framework 1.11.x`, `agent-framework-ollama 1.0.0b260709`, `pypdf`, `fpdf2`, `pytest` without errors (first run takes a few minutes; `poetry.lock` is created).

Run: `poetry run python -c "import agent_framework, agent_framework.ollama, hotl_demo; print('ok')"`
Expected: `ok`

Run: `poetry run pytest`
Expected: exit code 5 / "no tests ran" (nothing collected yet) — that is success for this task.

Run: `poetry run demo`
Expected: `demo runner not implemented yet (see Task 10)` (exit code 1).

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml poetry.lock .gitignore src tests scratchpad.md
git commit -m "chore: scaffold poetry project with agent-framework deps"
```

---

### Task 2: ArtifactStore (memory.json + ledger.jsonl + reports)

**Files:**
- Create: `src/hotl_demo/artifacts.py`
- Test: `tests/test_artifacts.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces (exact signatures later tasks rely on):

```python
PHASES: tuple[str, ...]  # ("discovery", "deep_analysis", "enterprise_context", "questionnaire")
REPOS: tuple[str, ...]   # ("oms-monolith", "oms-batch-recon")

class ArtifactStore:
    def __init__(self, run_dir: Path, repos: tuple[str, ...] = REPOS) -> None
    run_dir: Path
    def read_memory(self) -> dict
    def memory_text(self) -> str                     # json.dumps(read_memory(), indent=2)
    def update_memory(self, phase: str, unit: str | None, key: str, value: str) -> None
    def memory_key_count(self, phase: str, unit: str | None) -> int
    def review_completed(self) -> bool
    def set_review_completed(self) -> None
    def raise_question(self, phase: str, unit: str | None, question: str,
                       context: str, default_assumption: str) -> str   # returns "q-<n>"
    def read_ledger(self) -> list[dict]
    def open_questions(self) -> list[dict]
    def resolve_question(self, question_id: str, status: str, human_answer: str | None) -> dict
    def write_report(self, filename: str, text: str) -> Path
    def read_report(self, filename: str) -> str      # "" if missing
    def read_all_reports(self) -> dict[str, str]     # {filename: text} for phase_*.md, sorted
```

- [ ] **Step 1: Write the failing tests**

`tests/test_artifacts.py`:

```python
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


def test_reports_roundtrip(store):
    assert store.read_report("phase_01_discovery.md") == ""
    p = store.write_report("phase_01_discovery.md", "# Discovery\nfindings")
    assert p.read_text(encoding="utf-8").startswith("# Discovery")
    store.write_report("phase_03_enterprise_context.md", "ec")
    store.write_report("final_report.md", "final")  # not a phase report
    reports = store.read_all_reports()
    assert list(reports) == ["phase_01_discovery.md", "phase_03_enterprise_context.md"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/test_artifacts.py -v`
Expected: FAIL at import time — `ModuleNotFoundError: No module named 'hotl_demo.artifacts'`.

- [ ] **Step 3: Write `src/hotl_demo/artifacts.py`**

```python
"""File-backed run artifacts: shared memory (json), question ledger (jsonl), reports (md).

The files ARE the pipeline's long-term memory story; executors share one
ArtifactStore instance. A single lock serializes writers because the two
deep_analysis analyzers run concurrently.
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

PHASES: tuple[str, ...] = ("discovery", "deep_analysis", "enterprise_context", "questionnaire")
REPOS: tuple[str, ...] = ("oms-monolith", "oms-batch-recon")


class ArtifactStore:
    def __init__(self, run_dir: Path, repos: tuple[str, ...] = REPOS) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._memory_path = self.run_dir / "memory.json"
        self._ledger_path = self.run_dir / "ledger.jsonl"
        if not self._memory_path.exists():
            sections: dict[str, dict] = {
                p: ({r: {} for r in repos} if p == "deep_analysis" else {}) for p in PHASES
            }
            self._write_memory({
                "run_id": self.run_dir.name,
                "review_completed": False,
                "sections": sections,
            })

    # -- memory ---------------------------------------------------------

    def read_memory(self) -> dict:
        return json.loads(self._memory_path.read_text(encoding="utf-8"))

    def memory_text(self) -> str:
        return json.dumps(self.read_memory(), indent=2)

    def update_memory(self, phase: str, unit: str | None, key: str, value: str) -> None:
        with self._lock:
            mem = self.read_memory()
            target = mem["sections"][phase]
            if phase == "deep_analysis":
                target = target[unit]  # KeyError for unknown repo is intentional
            target[key] = value
            self._write_memory(mem)

    def memory_key_count(self, phase: str, unit: str | None) -> int:
        section = self.read_memory()["sections"][phase]
        if phase == "deep_analysis":
            section = section[unit]
        return len(section)

    def review_completed(self) -> bool:
        return bool(self.read_memory()["review_completed"])

    def set_review_completed(self) -> None:
        with self._lock:
            mem = self.read_memory()
            mem["review_completed"] = True
            self._write_memory(mem)

    def _write_memory(self, data: dict) -> None:
        _atomic_write(self._memory_path, json.dumps(data, indent=2))

    # -- ledger ---------------------------------------------------------

    def raise_question(self, phase: str, unit: str | None, question: str,
                       context: str, default_assumption: str) -> str:
        if phase not in PHASES:
            raise KeyError(phase)
        with self._lock:
            entry = {
                "id": f"q-{len(self._read_ledger_unlocked()) + 1}",
                "phase": phase,
                "unit": unit,
                "question": question,
                "context": context,
                "default_assumption": default_assumption,
                "status": "open",
                "human_answer": None,
                "asked_at": datetime.now(timezone.utc).isoformat(),
            }
            with self._ledger_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
            return entry["id"]

    def read_ledger(self) -> list[dict]:
        with self._lock:
            return self._read_ledger_unlocked()

    def open_questions(self) -> list[dict]:
        return [e for e in self.read_ledger() if e["status"] == "open"]

    def resolve_question(self, question_id: str, status: str, human_answer: str | None) -> dict:
        with self._lock:
            entries = self._read_ledger_unlocked()
            found: dict | None = None
            for e in entries:
                if e["id"] == question_id:
                    e["status"] = status
                    e["human_answer"] = human_answer
                    found = e
            if found is None:
                raise KeyError(question_id)
            _atomic_write(self._ledger_path, "".join(json.dumps(e) + "\n" for e in entries))
            return found

    def _read_ledger_unlocked(self) -> list[dict]:
        if not self._ledger_path.exists():
            return []
        text = self._ledger_path.read_text(encoding="utf-8")
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    # -- reports --------------------------------------------------------

    def write_report(self, filename: str, text: str) -> Path:
        path = self.run_dir / filename
        _atomic_write(path, text)
        return path

    def read_report(self, filename: str) -> str:
        path = self.run_dir / filename
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def read_all_reports(self) -> dict[str, str]:
        return {
            p.name: p.read_text(encoding="utf-8")
            for p in sorted(self.run_dir.glob("phase_*.md"))
        }


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
```

Note the design points a reviewer should check: every mutation holds `self._lock`; every write lands via `_atomic_write` (temp + `os.replace`) because a human may have the files open mid-run; ledger raise APPENDS, only `resolve_question` rewrites; `update_memory("deep_analysis", unit, ...)` requires a valid repo unit while other phases ignore `unit`.

Known ceiling, fine for a demo: `raise_question` re-reads the whole ledger to number the next id.

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/test_artifacts.py -v`
Expected: all 10 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hotl_demo/artifacts.py tests/test_artifacts.py
git commit -m "feat: thread-safe file-backed artifact store (memory, ledger, reports)"
```

---
### Task 3: Agent tools (scratchpad, raise_question, update_memory)

**Files:**
- Create: `src/hotl_demo/tools.py`
- Test: `tests/test_tools.py`

**Interfaces:**
- Consumes: `ArtifactStore` from Task 2 (`raise_question`, `update_memory`).
- Produces:

```python
SCRATCHPAD_PATH: Path                          # Path("scratchpad.md") — repo-root default
def ensure_scratchpad(path: Path = SCRATCHPAD_PATH) -> None   # create empty if missing, never truncate
def make_tools(store: ArtifactStore, phase: str, unit: str | None = None,
               scratchpad_path: Path = SCRATCHPAD_PATH) -> list
# returns [read_scratchpad, raise_question, update_memory] — @tool-decorated, bound to (store, phase, unit)
```

Note for the implementer: `@tool` comes from `agent_framework`. Docstrings on the tool functions ARE the tool descriptions the LLM sees — write them for the model, not for humans. Tools return plain strings; validation failures return `"ERROR: …"` strings (the framework feeds them back to the model), they never raise.

- [ ] **Step 1: Write the failing tests**

`tests/test_tools.py`:

```python
import pytest

from hotl_demo.artifacts import REPOS, ArtifactStore
from hotl_demo.tools import ensure_scratchpad, make_tools


@pytest.fixture()
def store(tmp_path):
    return ArtifactStore(tmp_path / "run", repos=REPOS)


def _tools(store, tmp_path, phase="discovery", unit=None):
    pad = tmp_path / "scratchpad.md"
    read_scratchpad, raise_question, update_memory = make_tools(
        store, phase, unit, scratchpad_path=pad
    )
    return pad, read_scratchpad, raise_question, update_memory


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/test_tools.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'hotl_demo.tools'`.

- [ ] **Step 3: Write `src/hotl_demo/tools.py`**

```python
"""The three agent tools. Docstrings below are the tool descriptions the LLM sees."""
from __future__ import annotations

from pathlib import Path

from agent_framework import tool

from .artifacts import ArtifactStore

SCRATCHPAD_PATH = Path("scratchpad.md")


def ensure_scratchpad(path: Path = SCRATCHPAD_PATH) -> None:
    if not path.exists():
        path.write_text("", encoding="utf-8")


def make_tools(store: ArtifactStore, phase: str, unit: str | None = None,
               scratchpad_path: Path = SCRATCHPAD_PATH) -> list:
    @tool(approval_mode="never_require")
    def read_scratchpad() -> str:
        """Read the human operator's scratchpad. It may contain steering guidance,
        priorities, or constraints for this assessment run. Always consult it
        before starting your work and follow any guidance it contains."""
        if scratchpad_path.exists():
            text = scratchpad_path.read_text(encoding="utf-8")
            if text.strip():
                return text
        return "The scratchpad is empty. No operator guidance provided."

    @tool(approval_mode="never_require")
    def raise_question(question: str, context: str, default_assumption: str) -> str:
        """Raise a question that requires human clarification or adjudication.
        Use when evidence conflicts or a decision-critical fact is missing.
        Provide the question, the evidence context, and the default assumption
        you will proceed with until a human answers. Returns the question id."""
        if not question.strip() or not default_assumption.strip():
            return "ERROR: question and default_assumption must both be non-empty. Retry with both."
        qid = store.raise_question(
            phase, unit, question.strip(), context.strip(), default_assumption.strip()
        )
        return f"Recorded {qid}. Proceed using your stated default assumption."

    @tool(approval_mode="never_require")
    def update_memory(key: str, value: str) -> str:
        """Record one finding in the shared long-term memory for this assessment.
        Call this once per key finding (3-8 times per phase). Use a short
        snake_case key (e.g. 'runtime', 'data_store', 'blockers') and a concise
        factual value."""
        if not key.strip() or not value.strip():
            return "ERROR: key and value must both be non-empty."
        store.update_memory(phase, unit, key.strip(), value.strip())
        return f"Memory updated: {key.strip()}"

    return [read_scratchpad, raise_question, update_memory]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/test_tools.py -v`
Expected: all 6 tests PASS.

Note: if `@tool` wraps functions into non-callable objects and the direct calls in the tests fail with `TypeError`, the framework's tool wrapper is invocable some other way — check `python -c "from agent_framework import tool; help(tool)"`. In `agent-framework 1.11` decorated tools remain directly callable; if that changed, call the underlying function via the wrapper's `.func`/`.invoke` attribute in `_tools()` (adjust the test helper only, not the production code).

- [ ] **Step 5: Commit**

```bash
git add src/hotl_demo/tools.py tests/test_tools.py
git commit -m "feat: scratchpad/raise_question/update_memory agent tools"
```

---

### Task 4: Sample data (EA docs, PDFs, legacy repos, questionnaire template)

**Files:**
- Create: `sample_data/docs_src/01_oms_application_architecture.md`
- Create: `sample_data/docs_src/02_enterprise_cloud_strategy.md`
- Create: `sample_data/docs_src/03_cybersecurity_standards.md`
- Create: `scripts/make_pdfs.py`
- Create: `sample_data/docs/*.pdf` (generated by the script, committed)
- Create: `sample_data/questionnaire_template.md`
- Create: `sample_data/repos/oms-monolith/{README.md,requirements.txt,order_processor.py,db.py,s3_uploader.py,file_store.py,crontab.txt}`
- Create: `sample_data/repos/oms-batch-recon/{README.md,recon_job.py,sftp_feed.py,config.py,crontab.txt}`
- Test: `tests/test_sample_data.py`

**Interfaces:**
- Consumes: nothing from other tasks (fpdf2/pypdf from Task 1's deps).
- Produces: the exact paths above. Task 5's source loaders read `sample_data/docs/*.pdf`, `sample_data/repos/<name>/`, `sample_data/questionnaire_template.md`. The planted strings tested here are what make the demo's ledger questions arise.

- [ ] **Step 1: Write `sample_data/docs_src/01_oms_application_architecture.md`**

ASCII only (fpdf core fonts). Plantings: vague "supporting batch processes", NO RTO/RPO anywhere, Oracle 11g + PL/SQL.

```markdown
# OMS Application Architecture and Business Context

Document ID: MR-ARCH-014 | Owner: Application Architecture | Status: Approved 2019

## Business context

The Order Management System (OMS) is the system of record for retail orders
at Meridian Retail. It captures orders from the e-commerce front end and the
store network, allocates inventory, prices and invoices orders, and feeds
downstream fulfilment. Supporting batch processes assist back-office
operations. Peak observed load is approximately 40 orders per minute during
seasonal trading.

## Application overview

The OMS is a monolithic application written in Python 2.7, deployed on a
pair of virtual machines on the on-premises VMware estate. There is no
containerisation. Releases are quarterly, deployed by the operations team
using shell scripts.

## Data architecture

All persistent state resides in an Oracle Database 11g Release 2 instance.
Significant business logic is implemented in approximately 120 PL/SQL
packages, including inventory allocation (OMS_PKG.ALLOCATE_INVENTORY),
pricing, and invoice generation. The application connects via cx_Oracle.

Order documents received from partners are dropped onto an NFS share
mounted at /mnt/nfs/orders. Batch jobs poll this location.

## Integration

- Order intake: SOAP web service exposed to the e-commerce platform.
- Warehouse events: published to IBM MQ (queue manager OMSQM01).
- Invoice archive: nightly upload of generated invoices to object storage.

## Operations

Batch jobs are scheduled with cron on the primary VM. Backups are taken
nightly to the enterprise backup service. A warm disaster recovery copy is
maintained at the secondary data centre. Recovery objectives for this
application are defined in the service catalogue.

## Known constraints

- The Python 2.7 runtime is end of life and no longer receives patches.
- Oracle 11gR2 is on extended support.
- The NFS share is a single point of failure for order intake.
```

(The "service catalogue" reference is a dead end on purpose — the RTO/RPO numbers exist nowhere in the corpus.)

- [ ] **Step 2: Write `sample_data/docs_src/02_enterprise_cloud_strategy.md`**

Plantings: Azure mandate, MQ retirement "date TBC".

```markdown
# Enterprise Cloud Strategy and Patterns

Document ID: MR-STRAT-002 | Owner: Enterprise Architecture | Status: Approved 2023

## Strategic direction

Meridian Retail is cloud-first. Microsoft Azure is the approved strategic
cloud platform for all new and migrated workloads. Exceptions require a
formal waiver from the Architecture Review Board.

## Landing zone

Workloads deploy into the enterprise landing zone: hub-and-spoke network
topology, centralised identity via Entra ID, platform logging, and policy
enforced via Azure Policy. All infrastructure must be defined as code.

## Approved patterns

- Compute: Azure App Service or AKS for containerised workloads.
  Plain IaaS virtual machines require a waiver.
- Data: Azure SQL Managed Instance or Azure Database for PostgreSQL.
  Flexible Server are the approved relational targets.
- Messaging: Azure Service Bus is the strategic messaging platform.
- File transfer: managed SFTP on Azure Blob Storage.
- Batch: Azure Container Apps jobs or Azure Functions timer triggers
  replace VM cron.

## Legacy middleware

IBM MQ is scheduled for retirement (date TBC). No new integrations may be
built on IBM MQ. Migrating applications should plan a path to Azure Service
Bus unless the retirement schedule dictates otherwise.

## Migration approach

Applications are assessed using the 6R model (rehost, replatform,
refactor, repurchase, retire, retain). PaaS-first: replatform is preferred
over rehost where feasible within the migration window.

## FinOps

All resources carry mandatory cost-centre and owner tags. Non-production
environments shut down outside business hours.
```

- [ ] **Step 3: Write `sample_data/docs_src/03_cybersecurity_standards.md`**

Plantings: "remain in-region" with region unspecified, no-credentials-in-code policy.

```markdown
# Cybersecurity and Data Protection Standards

Document ID: MR-SEC-009 | Owner: Information Security | Status: Approved 2024

## Data classification

Data is classified as Public, Internal, Confidential, or Restricted.
Customer personal data and order history are Confidential.

## Data residency

Confidential and Restricted data must remain in-region at rest and in
transit. Cross-region replication of Confidential data requires an approved
data transfer assessment.

## Secrets management

Credentials, API keys, and certificates must never be stored in source
code or configuration files under version control. All secrets must be
held in the enterprise vault service and rotated at least every 90 days.

## Encryption

Data in transit must use TLS 1.2 or higher. Data at rest must use AES-256
or platform-managed equivalent. Database connections must be encrypted.

## Access control

Production access follows least privilege with quarterly access reviews.
Service accounts must be non-interactive and individually owned.

## Logging and monitoring

Security-relevant events must be forwarded to the enterprise SIEM within
five minutes. Log retention is 13 months.
```

- [ ] **Step 4: Write `sample_data/questionnaire_template.md`**

```markdown
# Cloud Migration Readiness Questionnaire

Fill in every slot. Cite evidence (document or repository) for each answer.
Where an answer rests on a default assumption recorded in the question
ledger, reference the question id (e.g. "assumed - see q-3").

## 1. Business purpose and criticality

> (to be completed)

## 2. Migration scope

> (to be completed)

## 3. Target platform and landing zone fit

> (to be completed)

## 4. Migration approach (6R)

> (to be completed)

## 5. Data store and licensing

> (to be completed)

## 6. Recovery objectives (RTO / RPO)

> (to be completed)

## 7. Data residency and classification

> (to be completed)

## 8. Integrations and messaging

> (to be completed)

## 9. Security posture gaps

> (to be completed)

## 10. Timeline constraints and dependencies

> (to be completed)
```

- [ ] **Step 5: Write the `oms-monolith` repo files**

`sample_data/repos/oms-monolith/README.md`:

```markdown
# OMS Monolith

Core Order Management System. Python 2.7. Runs on VMOMS01/VMOMS02.
Deployed by ops via deploy.sh (not in this repo). See crontab.txt for
batch schedule.
```

`sample_data/repos/oms-monolith/requirements.txt`:

```
cx_Oracle==5.3
boto3==1.4.4
suds==0.4
pymqi==1.5.4
```

`sample_data/repos/oms-monolith/order_processor.py`:

```python
# -*- coding: utf-8 -*-
"""Order intake and allocation. Python 2.7 - do not run under Python 3."""
import ConfigParser
import os
import time

import db
import file_store
import s3_uploader

POLL_SECONDS = 30


def process_pending_orders():
    config = ConfigParser.ConfigParser()
    config.read("/etc/oms/oms.ini")
    for filename in file_store.list_order_files():
        print "processing %s" % filename
        order = file_store.read_order(filename)
        order_id = db.insert_order(order)
        db.allocate_inventory(order_id)
        invoice_path = db.generate_invoice(order_id)
        s3_uploader.archive_invoice(invoice_path)
        file_store.mark_done(filename)
        print "order %s complete" % order_id


if __name__ == "__main__":
    while True:
        process_pending_orders()
        time.sleep(POLL_SECONDS)
```

`sample_data/repos/oms-monolith/db.py`:

```python
# -*- coding: utf-8 -*-
"""Oracle access layer. Business logic lives in PL/SQL packages."""
import cx_Oracle

DSN = cx_Oracle.makedsn("dboms01.meridian.local", 1521, "OMSPRD")


def _connect():
    # credentials come from /etc/oms/oms.ini [oracle] section
    return cx_Oracle.connect("oms_app", _password_from_config(), DSN)


def insert_order(order):
    conn = _connect()
    cur = conn.cursor()
    cur.execute("INSERT INTO ORDERS (PAYLOAD) VALUES (:1) RETURNING ORDER_ID INTO :2",
                [order, cur.var(cx_Oracle.NUMBER)])
    conn.commit()
    return cur.fetchone()[0]


def allocate_inventory(order_id):
    conn = _connect()
    cur = conn.cursor()
    # 120+ PL/SQL packages like this one carry the core business logic
    cur.callproc("OMS_PKG.ALLOCATE_INVENTORY", [order_id])
    conn.commit()


def generate_invoice(order_id):
    conn = _connect()
    cur = conn.cursor()
    cur.callproc("OMS_INVOICE_PKG.GENERATE", [order_id])
    return "/var/oms/invoices/%s.pdf" % order_id


def _password_from_config():
    import ConfigParser
    c = ConfigParser.ConfigParser()
    c.read("/etc/oms/oms.ini")
    return c.get("oracle", "password")
```

`sample_data/repos/oms-monolith/s3_uploader.py`:

```python
# -*- coding: utf-8 -*-
"""Invoice archive. Uploads to AWS S3."""
import boto3

BUCKET = "meridian-oms-invoice-archive"


def archive_invoice(path):
    client = boto3.client("s3", region_name="us-east-1")
    key = "invoices/" + path.split("/")[-1]
    client.upload_file(path, BUCKET, key)
    print "archived %s to s3://%s/%s" % (path, BUCKET, key)
```

`sample_data/repos/oms-monolith/file_store.py`:

```python
# -*- coding: utf-8 -*-
"""Order file drops arrive on the NFS share."""
import os
import shutil

ORDER_DIR = "/mnt/nfs/orders/incoming"
DONE_DIR = "/mnt/nfs/orders/processed"


def list_order_files():
    return [f for f in os.listdir(ORDER_DIR) if f.endswith(".xml")]


def read_order(filename):
    with open(os.path.join(ORDER_DIR, filename)) as f:
        return f.read()


def mark_done(filename):
    shutil.move(os.path.join(ORDER_DIR, filename), os.path.join(DONE_DIR, filename))
```

`sample_data/repos/oms-monolith/crontab.txt`:

```
# OMS batch schedule (VMOMS01)
*/1 * * * *  /usr/bin/python /opt/oms/order_processor.py
0 2 * * *    /usr/bin/python /opt/oms/nightly_pricing_refresh.py
30 2 * * *   /opt/oms/scripts/backup_to_tape.sh
```

- [ ] **Step 6: Write the `oms-batch-recon` repo files**

`sample_data/repos/oms-batch-recon/README.md`:

```markdown
# OMS Batch Reconciliation

Nightly financial reconciliation between OMS order totals and the general
ledger extract. Produces discrepancy reports for the finance team and
uploads the daily reconciliation CSV to VendorCo via SFTP.
```

`sample_data/repos/oms-batch-recon/config.py`:

```python
# -*- coding: utf-8 -*-
"""Connection settings for the reconciliation batch."""

ORACLE_HOST = "dboms01.meridian.local"
ORACLE_SID = "OMSPRD"
ORACLE_USER = "recon_batch"
ORACLE_PASSWORD = "Rec0n#2011!"  # TODO move to vault someday

SFTP_HOST = "sftp.vendorco.example"
SFTP_USER = "meridian"
SFTP_KEY_PATH = "/home/recon/.ssh/id_rsa"

GL_EXTRACT_DIR = "/mnt/nfs/finance/gl_extracts"
REPORT_DIR = "/var/recon/reports"
```

`sample_data/repos/oms-batch-recon/recon_job.py`:

```python
# -*- coding: utf-8 -*-
"""Nightly reconciliation: OMS order totals vs general ledger extract."""
import csv
import datetime
import os

import cx_Oracle

import config
import sftp_feed


def fetch_oms_totals(business_date):
    dsn = cx_Oracle.makedsn(config.ORACLE_HOST, 1521, config.ORACLE_SID)
    conn = cx_Oracle.connect(config.ORACLE_USER, config.ORACLE_PASSWORD, dsn)
    cur = conn.cursor()
    cur.execute(
        "SELECT STORE_ID, SUM(TOTAL_AMOUNT) FROM ORDERS "
        "WHERE TRUNC(ORDER_DATE) = :1 GROUP BY STORE_ID", [business_date])
    return dict(cur.fetchall())


def load_gl_totals(business_date):
    path = os.path.join(config.GL_EXTRACT_DIR, "gl_%s.csv" % business_date.strftime("%Y%m%d"))
    totals = {}
    with open(path) as f:
        for row in csv.reader(f):
            totals[int(row[0])] = float(row[1])
    return totals


def run():
    business_date = datetime.date.today() - datetime.timedelta(days=1)
    oms = fetch_oms_totals(business_date)
    gl = load_gl_totals(business_date)
    report_path = os.path.join(config.REPORT_DIR, "recon_%s.csv" % business_date)
    with open(report_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["store_id", "oms_total", "gl_total", "delta"])
        for store_id in sorted(set(oms) | set(gl)):
            a, b = oms.get(store_id, 0.0), gl.get(store_id, 0.0)
            writer.writerow([store_id, a, b, round(a - b, 2)])
    sftp_feed.upload(report_path)
    print "reconciliation complete: %s" % report_path


if __name__ == "__main__":
    run()
```

`sample_data/repos/oms-batch-recon/sftp_feed.py`:

```python
# -*- coding: utf-8 -*-
"""Upload reconciliation output to VendorCo."""
import subprocess

import config


def upload(path):
    # ops insisted on shelling out to sftp rather than using paramiko
    cmd = "echo 'put %s /inbound/' | sftp -i %s %s@%s" % (
        path, config.SFTP_KEY_PATH, config.SFTP_USER, config.SFTP_HOST)
    subprocess.check_call(cmd, shell=True)
```

`sample_data/repos/oms-batch-recon/crontab.txt`:

```
# reconciliation runs after the GL extract lands
15 3 * * *  /usr/bin/python /opt/recon/recon_job.py
```

- [ ] **Step 7: Write `scripts/make_pdfs.py`**

```python
"""Regenerate sample_data/docs/*.pdf from sample_data/docs_src/*.md (fpdf2)."""
from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "sample_data" / "docs_src"
OUT = ROOT / "sample_data" / "docs"


def render(md_path: Path, pdf_path: Path) -> None:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    for line in md_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            pdf.set_font("Helvetica", "B", 16)
            line = line[2:]
        elif line.startswith("## "):
            pdf.set_font("Helvetica", "B", 13)
            line = line[3:]
        else:
            pdf.set_font("Helvetica", size=10)
        pdf.multi_cell(0, 5, line if line.strip() else " ")
    pdf.output(str(pdf_path))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for md in sorted(SRC.glob("*.md")):
        render(md, OUT / (md.stem + ".pdf"))
        print(f"wrote {OUT / (md.stem + '.pdf')}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 8: Generate the PDFs**

Run: `poetry run python scripts/make_pdfs.py`
Expected output (3 lines):

```
wrote ...sample_data/docs/01_oms_application_architecture.pdf
wrote ...sample_data/docs/02_enterprise_cloud_strategy.pdf
wrote ...sample_data/docs/03_cybersecurity_standards.pdf
```

- [ ] **Step 9: Write the failing test**

`tests/test_sample_data.py` — asserts the PDFs are extractable and every planting is present where the spec says it is:

```python
from pathlib import Path

from pypdf import PdfReader

DOCS = Path("sample_data/docs")
REPOS = Path("sample_data/repos")


def _pdf_text(name: str) -> str:
    raw = "\n".join(page.extract_text() for page in PdfReader(DOCS / name).pages)
    return " ".join(raw.split())  # normalize: extraction wraps lines mid-phrase


def test_pdfs_exist_and_extract():
    assert sorted(p.name for p in DOCS.glob("*.pdf")) == [
        "01_oms_application_architecture.pdf",
        "02_enterprise_cloud_strategy.pdf",
        "03_cybersecurity_standards.pdf",
    ]


def test_planted_gaps_in_pdfs():
    arch = _pdf_text("01_oms_application_architecture.pdf")
    strategy = _pdf_text("02_enterprise_cloud_strategy.pdf")
    security = _pdf_text("03_cybersecurity_standards.pdf")
    # planting 1 fuel: vague batch mention, recon absent
    assert "Supporting batch processes" in arch
    assert "reconciliation" not in arch.lower()
    # planting 3: no RTO/RPO numbers anywhere in the corpus
    assert "RTO" not in arch and "RPO" not in arch
    # planting 2: Azure mandate
    assert "Microsoft Azure is the approved strategic" in strategy
    # planting 5: MQ retirement date TBC
    assert "retirement (date TBC)" in strategy
    # planting 4: residency region unspecified
    assert "remain in-region" in security
    # planting 6 fuel: Oracle named in arch doc
    assert "Oracle Database 11g" in arch


def test_planted_conflicts_in_repos():
    s3 = (REPOS / "oms-monolith" / "s3_uploader.py").read_text(encoding="utf-8")
    assert "boto3" in s3  # planting 2: AWS in code vs Azure mandate
    cfg = (REPOS / "oms-batch-recon" / "config.py").read_text(encoding="utf-8")
    assert 'ORACLE_PASSWORD = "Rec0n#2011!"' in cfg  # planting 7: hardcoded cred
    readme = (REPOS / "oms-batch-recon" / "README.md").read_text(encoding="utf-8")
    assert "financial reconciliation" in readme  # planting 1: undocumented scope
    fs = (REPOS / "oms-monolith" / "file_store.py").read_text(encoding="utf-8")
    assert "/mnt/nfs/orders" in fs
```

Run: `poetry run pytest tests/test_sample_data.py -v`
Expected: PASS if steps 1-8 were done exactly; any FAIL means a planting got lost — fix the data file, regenerate PDFs, re-run.

(Ordering note: data was written before this test because the test asserts content, not behavior; the test is the guardrail that plantings survive future edits.)

- [ ] **Step 10: Commit**

```bash
git add sample_data scripts tests/test_sample_data.py
git commit -m "feat: synthetic EA corpus, legacy repos, questionnaire template with planted gaps"
```

---
### Task 5: Messages, phase specs, source loaders, prompt builders

**Files:**
- Create: `src/hotl_demo/phases.py` (data + pure functions half; the executor class is Task 6)
- Test: `tests/test_phases.py`

**Interfaces:**
- Consumes: `ArtifactStore.memory_text/open_questions` (Task 2); `sample_data/` layout (Task 4).
- Produces (exact names later tasks import from `hotl_demo.phases`):

```python
@dataclass class PhaseDone:        phase: str; unit: str | None = None      # initial completion
@dataclass class AnalysisDone:     unit: str                                 # analyzer initial completion
@dataclass class RevisionDone:     phase: str; unit: str | None = None      # revision completion
@dataclass class RevisionTrigger:  phase: str; unit: str | None; answers: list[dict]
@dataclass class ReportTrigger:    pass

@dataclass class PhaseSpec:
    name: str                      # one of artifacts.PHASES
    unit: str | None               # repo name for analyzers, else None
    executor_id: str               # "discovery", "analyze:oms-monolith", ...
    report_filename: str           # "phase_01_discovery.md", ...
    instructions: str              # phase-specific system instructions
    load_sources: Callable[[], str]

def load_pdf_text(path: Path) -> str
def load_repo_text(repo_dir: Path) -> str          # "=== <relpath> ===" blocks, sorted
def repo_listing(repo_dir: Path) -> str            # sorted relative paths, one per line
def build_phase_specs(base_dir: Path) -> list[PhaseSpec]   # ordered: discovery, 2 analyzers, ec, questionnaire
def build_initial_prompt(spec: PhaseSpec, sources: str, memory_text: str, open_questions: list[dict]) -> str
def build_revision_prompt(spec: PhaseSpec, sources: str, memory_text: str, answers: list[dict], previous_report: str) -> str
```

- [ ] **Step 1: Write the failing tests**

`tests/test_phases.py`:

```python
from pathlib import Path

from hotl_demo.phases import (
    PhaseSpec,
    build_initial_prompt,
    build_phase_specs,
    build_revision_prompt,
    load_pdf_text,
    load_repo_text,
    repo_listing,
)

BASE = Path("sample_data")


def _spec(name="discovery", unit=None):
    return PhaseSpec(
        name=name, unit=unit, executor_id=name,
        report_filename=f"phase_x_{name}.md",
        instructions="INSTRUCTIONS-SENTINEL",
        load_sources=lambda: "SOURCES-SENTINEL",
    )


def test_load_pdf_text_extracts_planted_content():
    text = " ".join(load_pdf_text(BASE / "docs" / "02_enterprise_cloud_strategy.pdf").split())
    assert "Microsoft Azure is the approved strategic" in text


def test_repo_loaders():
    listing = repo_listing(BASE / "repos" / "oms-monolith")
    assert "s3_uploader.py" in listing and "README.md" in listing
    text = load_repo_text(BASE / "repos" / "oms-batch-recon")
    assert "=== config.py ===" in text
    assert 'ORACLE_PASSWORD = "Rec0n#2011!"' in text


def test_build_phase_specs_order_units_reports():
    specs = build_phase_specs(BASE)
    assert [(s.name, s.unit) for s in specs] == [
        ("discovery", None),
        ("deep_analysis", "oms-monolith"),
        ("deep_analysis", "oms-batch-recon"),
        ("enterprise_context", None),
        ("questionnaire", None),
    ]
    assert [s.executor_id for s in specs] == [
        "discovery", "analyze:oms-monolith", "analyze:oms-batch-recon",
        "enterprise_context", "questionnaire",
    ]
    assert specs[1].report_filename == "phase_02_deep_analysis_oms-monolith.md"
    assert specs[4].report_filename == "phase_04_questionnaire.md"


def test_spec_sources_contain_the_right_material():
    # normalize whitespace: PDF extraction wraps lines mid-phrase
    def norm(text: str) -> str:
        return " ".join(text.split())

    by_id = {s.executor_id: s for s in build_phase_specs(BASE)}
    discovery = norm(by_id["discovery"].load_sources())
    assert "Microsoft Azure is the approved strategic" in discovery   # strategy pdf
    assert "financial reconciliation" in discovery                    # recon README
    assert "=== config.py ===" not in discovery                       # listings only, not full repo text
    monolith = norm(by_id["analyze:oms-monolith"].load_sources())
    assert "boto3" in monolith                                        # full repo text
    assert "Supporting batch processes" in monolith                   # PDF 1 included
    assert "Rec0n#2011!" not in monolith                              # other repo excluded
    ec = norm(by_id["enterprise_context"].load_sources())
    assert "remain in-region" in ec                                   # security pdf
    assert "Supporting batch processes" not in ec                     # PDF 1 excluded
    questionnaire = norm(by_id["questionnaire"].load_sources())
    assert "## 6. Recovery objectives (RTO / RPO)" in questionnaire   # template text


def test_initial_prompt_assembly():
    open_qs = [{"id": "q-1", "phase": "discovery", "unit": None,
                "question": "Scope?", "status": "open"}]
    prompt = build_initial_prompt(_spec(), "SOURCES-SENTINEL", '{"m": 1}', open_qs)
    assert "INSTRUCTIONS-SENTINEL" in prompt
    assert "SOURCES-SENTINEL" in prompt
    assert '{"m": 1}' in prompt
    assert "q-1" in prompt and "Scope?" in prompt        # open ledger for dup suppression
    assert "read_scratchpad" in prompt                   # scratchpad reminder
    assert "raise_question" in prompt and "update_memory" in prompt


def test_initial_prompt_with_empty_ledger_says_so():
    prompt = build_initial_prompt(_spec(), "S", "{}", [])
    assert "No questions raised so far" in prompt


def test_revision_prompt_assembly():
    answers = [{"id": "q-2", "question": "Which cloud?", "human_answer": "AWS, actually",
                "default_assumption": "Azure"}]
    prompt = build_revision_prompt(_spec(), "SOURCES-SENTINEL", "{}", answers, "OLD-REPORT")
    assert "AWS, actually" in prompt
    assert "authoritative" in prompt.lower()
    assert "OLD-REPORT" in prompt
    assert "q-2" in prompt
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/test_phases.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'hotl_demo.phases'`.

- [ ] **Step 3: Write the data/pure-function half of `src/hotl_demo/phases.py`**

```python
"""Phase definitions: workflow messages, specs, source loaders, prompt builders."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from pypdf import PdfReader

from .artifacts import REPOS


# -- workflow messages (types encode mode: no mode flags anywhere) -------

@dataclass
class PhaseDone:
    phase: str
    unit: str | None = None


@dataclass
class AnalysisDone:
    unit: str


@dataclass
class RevisionDone:
    phase: str
    unit: str | None = None


@dataclass
class RevisionTrigger:
    phase: str
    unit: str | None
    answers: list[dict] = field(default_factory=list)


@dataclass
class ReportTrigger:
    pass


# -- specs ----------------------------------------------------------------

@dataclass
class PhaseSpec:
    name: str
    unit: str | None
    executor_id: str
    report_filename: str
    instructions: str
    load_sources: Callable[[], str]


# -- source loaders ---------------------------------------------------------

def load_pdf_text(path: Path) -> str:
    return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)


def repo_listing(repo_dir: Path) -> str:
    files = sorted(p.relative_to(repo_dir).as_posix() for p in repo_dir.rglob("*") if p.is_file())
    return "\n".join(files)


def load_repo_text(repo_dir: Path) -> str:
    blocks = []
    for rel in repo_listing(repo_dir).splitlines():
        blocks.append(f"=== {rel} ===\n{(repo_dir / rel).read_text(encoding='utf-8')}")
    return "\n\n".join(blocks)


# -- phase instructions -----------------------------------------------------

_COMMON_DUTIES = """
Your duties on every run:
1. FIRST call the read_scratchpad tool and follow any operator guidance in it.
2. Record 3-8 key findings with the update_memory tool (short snake_case key,
   concise factual value). Findings must be grounded in the source material.
3. When evidence conflicts or a decision-critical fact is missing, call the
   raise_question tool with the question, the evidence context, and the
   default assumption you will proceed with - then proceed using that default.
   Check the OPEN QUESTIONS list you were given first: never re-raise a
   question that is already open; reference its id instead.
4. Finish by writing your phase report as your final answer: well-structured
   markdown, headings, concise, evidence-cited. The final answer must be the
   report itself - no preamble about what you are going to do.
"""

_INSTRUCTIONS: dict[str, str] = {
    "discovery": (
        "You are the discovery analyst opening a cloud migration readiness "
        "assessment for Meridian Retail's Order Management System (OMS). "
        "Establish the TRUE purpose and shape of the legacy estate: business "
        "function, users, criticality, and the actual scope of what must "
        "migrate. Compare what the documents claim against what the "
        "repositories actually contain; flag scope that code reveals but "
        "documents omit. Do not deep-dive into code internals - that is a "
        "later phase." + _COMMON_DUTIES
    ),
    "deep_analysis": (
        "You are a senior engineer performing a repo-level deep dive on ONE "
        "repository ({unit}) of the OMS estate for cloud migration readiness. "
        "Analyze runtime and language versions, frameworks, data access, "
        "external integrations, file system coupling, schedulers, secrets "
        "handling, and cloud blockers. Be specific: name files and lines of "
        "evidence." + _COMMON_DUTIES
    ),
    "enterprise_context": (
        "You are the enterprise architect overlaying corporate guidance onto "
        "the assessment: cloud strategy and approved patterns, cybersecurity "
        "and data-protection standards. Map each earlier finding (in shared "
        "memory) to the relevant corporate mandate, and call out every "
        "conflict between strategy and observed reality, every policy "
        "violation, and every mandate whose parameters are unspecified."
        + _COMMON_DUTIES
    ),
    "questionnaire": (
        "You are completing the standard Cloud Migration Readiness "
        "Questionnaire. Fill in EVERY slot of the template using the shared "
        "memory and phase evidence. Cite evidence for each answer. Where an "
        "answer rests on a default assumption from an open ledger question, "
        "reference the question id. If a slot cannot be answered and no open "
        "question covers it, raise one. Your final answer is the completed "
        "questionnaire in the template's structure." + _COMMON_DUTIES
    ),
}


# -- spec factory -------------------------------------------------------------

def build_phase_specs(base_dir: Path) -> list[PhaseSpec]:
    docs = base_dir / "docs"
    repos = base_dir / "repos"
    pdf1 = docs / "01_oms_application_architecture.pdf"
    pdf2 = docs / "02_enterprise_cloud_strategy.pdf"
    pdf3 = docs / "03_cybersecurity_standards.pdf"

    def discovery_sources() -> str:
        parts = [f"--- DOCUMENT: {p.name} ---\n{load_pdf_text(p)}" for p in (pdf1, pdf2, pdf3)]
        for repo in REPOS:
            parts.append(f"--- REPO FILE LISTING: {repo} ---\n{repo_listing(repos / repo)}")
            readme = repos / repo / "README.md"
            parts.append(f"--- {repo}/README.md ---\n{readme.read_text(encoding='utf-8')}")
        return "\n\n".join(parts)

    def analyzer_sources(repo: str) -> Callable[[], str]:
        def load() -> str:
            return (
                f"--- DOCUMENT: {pdf1.name} ---\n{load_pdf_text(pdf1)}\n\n"
                f"--- REPOSITORY: {repo} (full contents) ---\n{load_repo_text(repos / repo)}"
            )
        return load

    def ec_sources() -> str:
        return "\n\n".join(
            f"--- DOCUMENT: {p.name} ---\n{load_pdf_text(p)}" for p in (pdf2, pdf3)
        )

    def questionnaire_sources() -> str:
        template = (base_dir / "questionnaire_template.md").read_text(encoding="utf-8")
        return f"--- QUESTIONNAIRE TEMPLATE ---\n{template}"

    specs = [
        PhaseSpec("discovery", None, "discovery", "phase_01_discovery.md",
                  _INSTRUCTIONS["discovery"], discovery_sources),
    ]
    for repo in REPOS:
        specs.append(PhaseSpec(
            "deep_analysis", repo, f"analyze:{repo}",
            f"phase_02_deep_analysis_{repo}.md",
            _INSTRUCTIONS["deep_analysis"].format(unit=repo),
            analyzer_sources(repo),
        ))
    specs.append(PhaseSpec("enterprise_context", None, "enterprise_context",
                           "phase_03_enterprise_context.md",
                           _INSTRUCTIONS["enterprise_context"], ec_sources))
    specs.append(PhaseSpec("questionnaire", None, "questionnaire",
                           "phase_04_questionnaire.md",
                           _INSTRUCTIONS["questionnaire"], questionnaire_sources))
    return specs


# -- prompt builders -----------------------------------------------------------

def _format_open_questions(open_questions: list[dict]) -> str:
    if not open_questions:
        return "No questions raised so far."
    return "\n".join(
        f"- {q['id']} ({q['phase']}{'/' + q['unit'] if q.get('unit') else ''}): {q['question']}"
        for q in open_questions
    )


def build_initial_prompt(spec: PhaseSpec, sources: str, memory_text: str,
                         open_questions: list[dict]) -> str:
    return f"""{spec.instructions}

Remember: call read_scratchpad first; record findings with update_memory;
raise adjudication needs with raise_question.

## OPEN QUESTIONS already in the ledger (do not re-raise)
{_format_open_questions(open_questions)}

## SHARED MEMORY (accumulated by earlier phases)
```json
{memory_text}
```

## SOURCE MATERIAL
{sources}

Produce your phase report now.
"""


def build_revision_prompt(spec: PhaseSpec, sources: str, memory_text: str,
                          answers: list[dict], previous_report: str) -> str:
    answer_lines = "\n".join(
        f"- {a['id']}: Q: {a['question']}\n  Human answer (AUTHORITATIVE): {a['human_answer']}\n"
        f"  (replaces default assumption: {a['default_assumption']})"
        for a in answers
    )
    return f"""{spec.instructions}

A human reviewer has adjudicated questions this phase raised. Human answers
are authoritative and override any conflicting document or code evidence.
Rewrite your phase report and refresh your update_memory findings to reflect
them. Do not raise these questions again.

## HUMAN ANSWERS
{answer_lines}

## YOUR PREVIOUS REPORT
{previous_report}

## SHARED MEMORY
```json
{memory_text}
```

## SOURCE MATERIAL
{sources}

Produce the revised phase report now.
"""
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/test_phases.py -v`
Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hotl_demo/phases.py tests/test_phases.py
git commit -m "feat: phase specs, source loaders, prompt builders, workflow messages"
```

---

### Task 6: PhaseExecutor (agent-running workflow node)

**Files:**
- Modify: `src/hotl_demo/phases.py` (append the executor class + agent preamble)
- Test: `tests/test_phase_executor.py`
- Create: `tests/conftest.py` (FakeCtx used by several executor tests)

**Interfaces:**
- Consumes: Task 5's messages/specs/prompts; `ArtifactStore` (Task 2); `make_tools` (Task 3); `Agent`/`OllamaChatClient` (framework).
- Produces:

```python
class PhaseExecutor(Executor):
    def __init__(self, spec: PhaseSpec, store: ArtifactStore,
                 scratchpad_path: Path = SCRATCHPAD_PATH,
                 agent: Any | None = None) -> None    # agent injectable for tests
    # @handler on_start(go: str, ctx)                 -> initial run (discovery only receives str)
    # @handler on_upstream(done: PhaseDone, ctx)      -> initial run
    # @handler on_revision(trig: RevisionTrigger, ctx)-> revision run
```

Emission rules later tasks depend on: initial completion sends `AnalysisDone(unit)` when `spec.unit` is set, else `PhaseDone(spec.name)`; revision completion always sends `RevisionDone(spec.name, spec.unit)`.

- [ ] **Step 1: Write `tests/conftest.py` with FakeCtx and FakeAgent**

Executor `@handler` methods are plain async methods — call them directly with a duck-typed ctx; no workflow needed.

```python
class FakeCtx:
    """Duck-typed WorkflowContext capturing outbound messages/requests/outputs."""

    def __init__(self):
        self.sent = []
        self.requests = []
        self.outputs = []

    async def send_message(self, message):
        self.sent.append(message)

    async def request_info(self, request_data, response_type=str):
        self.requests.append(request_data)

    async def yield_output(self, output):
        self.outputs.append(output)


class FakeAgentResult:
    def __init__(self, text):
        self.text = text


class FakeAgent:
    """Scripted stand-in for agent_framework.Agent: returns queued texts, records prompts."""

    def __init__(self, texts, side_effect=None):
        self.texts = list(texts)
        self.prompts = []
        self.side_effect = side_effect  # optional callable(prompt) run per call

    async def run(self, prompt):
        self.prompts.append(prompt)
        if self.side_effect:
            self.side_effect(prompt)
        return FakeAgentResult(self.texts.pop(0) if self.texts else "")
```

- [ ] **Step 2: Write the failing tests**

`tests/test_phase_executor.py`:

```python
import pytest

from conftest import FakeAgent, FakeCtx

from hotl_demo.artifacts import REPOS, ArtifactStore
from hotl_demo.phases import (
    AnalysisDone,
    PhaseDone,
    PhaseExecutor,
    PhaseSpec,
    RevisionDone,
    RevisionTrigger,
)


@pytest.fixture()
def store(tmp_path):
    return ArtifactStore(tmp_path / "run", repos=REPOS)


def _spec(name="discovery", unit=None, executor_id=None):
    return PhaseSpec(
        name=name, unit=unit, executor_id=executor_id or name,
        report_filename=f"phase_x_{name}{'_' + unit if unit else ''}.md",
        instructions="do the thing", load_sources=lambda: "SOURCES",
    )


def _executor(store, tmp_path, spec, agent):
    return PhaseExecutor(spec, store, scratchpad_path=tmp_path / "pad.md", agent=agent)


@pytest.mark.asyncio
async def test_initial_run_writes_report_and_sends_phase_done(store, tmp_path):
    spec = _spec()
    agent = FakeAgent(
        ["# Report"], side_effect=lambda p: store.update_memory("discovery", None, "k", "v")
    )
    ctx = FakeCtx()
    await _executor(store, tmp_path, spec, agent).on_start("start", ctx)
    assert store.read_report(spec.report_filename) == "# Report"
    assert ctx.sent == [PhaseDone("discovery")]
    assert len(agent.prompts) == 1  # memory was updated -> no nudge


@pytest.mark.asyncio
async def test_analyzer_sends_analysis_done(store, tmp_path):
    spec = _spec("deep_analysis", "oms-monolith", "analyze:oms-monolith")
    agent = FakeAgent(
        ["# R"], side_effect=lambda p: store.update_memory("deep_analysis", "oms-monolith", "k", "v")
    )
    ctx = FakeCtx()
    await _executor(store, tmp_path, spec, agent).on_upstream(PhaseDone("discovery"), ctx)
    assert ctx.sent == [AnalysisDone("oms-monolith")]


@pytest.mark.asyncio
async def test_nudge_fires_once_when_no_memory_written(store, tmp_path):
    spec = _spec()
    agent = FakeAgent(["# Report", "ignored nudge reply"])  # never writes memory
    ctx = FakeCtx()
    await _executor(store, tmp_path, spec, agent).on_start("start", ctx)
    assert len(agent.prompts) == 2                      # initial + exactly one nudge
    assert "update_memory" in agent.prompts[1]          # nudge asks for memory calls
    report = store.read_report(spec.report_filename)
    assert report.startswith("# Report")
    assert "no memory entries" in report                # noted, not crashed
    assert ctx.sent == [PhaseDone("discovery")]         # pipeline continues


@pytest.mark.asyncio
async def test_revision_run_rewrites_report_and_sends_revision_done(store, tmp_path):
    spec = _spec()
    store.write_report(spec.report_filename, "OLD")
    agent = FakeAgent(["NEW"])
    ctx = FakeCtx()
    trig = RevisionTrigger("discovery", None, answers=[{
        "id": "q-1", "question": "Scope?", "human_answer": "recon in scope",
        "default_assumption": "in scope",
    }])
    await _executor(store, tmp_path, spec, agent).on_revision(trig, ctx)
    assert store.read_report(spec.report_filename) == "NEW"
    assert ctx.sent == [RevisionDone("discovery", None)]
    assert "OLD" in agent.prompts[0]                 # previous report included
    assert "recon in scope" in agent.prompts[0]      # human answer included


@pytest.mark.asyncio
async def test_empty_agent_text_falls_back(store, tmp_path):
    spec = _spec()
    agent = FakeAgent([""], side_effect=lambda p: store.update_memory("discovery", None, "k", "v"))
    ctx = FakeCtx()
    await _executor(store, tmp_path, spec, agent).on_start("start", ctx)
    assert "no text" in store.read_report(spec.report_filename)
```

Also add `pytest-asyncio` as a dev dependency for the async tests:

Run: `poetry add --group dev pytest-asyncio`
Then add to `pyproject.toml` under `[tool.pytest.ini_options]`:

```toml
asyncio_mode = "auto"
```

(With `asyncio_mode = "auto"` the `@pytest.mark.asyncio` markers are redundant but harmless — keep them for readability.)

- [ ] **Step 3: Run tests to verify they fail**

Run: `poetry run pytest tests/test_phase_executor.py -v`
Expected: FAIL — `ImportError: cannot import name 'PhaseExecutor'`.

- [ ] **Step 4: Append the executor to `src/hotl_demo/phases.py`**

Add imports at the top of the file:

```python
import os
from typing import Any

from agent_framework import Agent, Executor, WorkflowContext, handler
from agent_framework.ollama import OllamaChatClient

from .artifacts import ArtifactStore
from .tools import SCRATCHPAD_PATH, make_tools
```

Append at the bottom:

```python
_NUDGE = """You produced your phase report but recorded no findings in shared
memory. Call the update_memory tool now for each of the 3-8 key findings in
the report below, then reply "done".

{report}
"""

_MEMORY_GAP_NOTE = "\n\n> NOTE: agent recorded no memory entries for this phase."


class PhaseExecutor(Executor):
    """One workflow node per phase (and per repo for deep_analysis)."""

    def __init__(self, spec: PhaseSpec, store: ArtifactStore,
                 scratchpad_path=SCRATCHPAD_PATH, agent: Any | None = None) -> None:
        super().__init__(id=spec.executor_id)
        self._spec = spec
        self._store = store
        self._agent = agent or Agent(
            client=OllamaChatClient(),  # model comes from OLLAMA_MODEL env var
            name=spec.executor_id.replace(":", "_"),
            instructions="You are one phase of a multi-agent assessment pipeline.",
            tools=make_tools(store, spec.name, spec.unit, scratchpad_path),
        )

    @handler
    async def on_start(self, go: str, ctx: WorkflowContext[PhaseDone | AnalysisDone]) -> None:
        await self._run_initial(ctx)

    @handler
    async def on_upstream(self, done: PhaseDone,
                          ctx: WorkflowContext[PhaseDone | AnalysisDone]) -> None:
        await self._run_initial(ctx)

    @handler
    async def on_revision(self, trig: RevisionTrigger,
                          ctx: WorkflowContext[RevisionDone]) -> None:
        prompt = build_revision_prompt(
            self._spec, self._spec.load_sources(), self._store.memory_text(),
            trig.answers, self._store.read_report(self._spec.report_filename),
        )
        text = await self._invoke(prompt)
        self._store.write_report(self._spec.report_filename, text)
        print(f"  revised: {self._spec.executor_id}")
        await ctx.send_message(RevisionDone(self._spec.name, self._spec.unit))

    async def _run_initial(self, ctx) -> None:
        before = self._store.memory_key_count(self._spec.name, self._spec.unit)
        prompt = build_initial_prompt(
            self._spec, self._spec.load_sources(), self._store.memory_text(),
            self._store.open_questions(),
        )
        text = await self._invoke(prompt)
        if self._store.memory_key_count(self._spec.name, self._spec.unit) == before:
            # ponytail: one bounded nudge, then proceed and note the gap
            await self._invoke(_NUDGE.format(report=text))
            if self._store.memory_key_count(self._spec.name, self._spec.unit) == before:
                text += _MEMORY_GAP_NOTE
        self._store.write_report(self._spec.report_filename, text)
        raised = [q for q in self._store.read_ledger()
                  if q["phase"] == self._spec.name and q["unit"] == self._spec.unit]
        print(f"  {self._spec.executor_id}: report written ({len(raised)} questions raised)")
        if self._spec.unit is not None:
            await ctx.send_message(AnalysisDone(self._spec.unit))
        else:
            await ctx.send_message(PhaseDone(self._spec.name))

    async def _invoke(self, prompt: str) -> str:
        result = await self._agent.run(prompt)
        return result.text or "(no text returned by the model)"
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `poetry run pytest tests/test_phase_executor.py -v`
Expected: all 5 tests PASS.

Run: `poetry run pytest`
Expected: everything so far PASSES (artifacts, tools, sample data, phases, phase executor).

- [ ] **Step 6: Commit**

```bash
git add src/hotl_demo/phases.py tests/test_phase_executor.py tests/conftest.py pyproject.toml poetry.lock
git commit -m "feat: PhaseExecutor with revision runs and bounded memory nudge"
```

---
### Task 7: ReviewExecutor (the human gate)

**Files:**
- Create: `src/hotl_demo/review.py`
- Test: `tests/test_review.py`

**Interfaces:**
- Consumes: `ArtifactStore` (Task 2); messages from Task 5; `FakeCtx` (Task 6's conftest).
- Produces:

```python
@dataclass
class LedgerQuestionRequest:      # payload of ctx.request_info; the CLI renders it
    question_id: str; phase: str; unit: str | None
    question: str; context: str; default_assumption: str

def affected_targets(ledger: list[dict], revision_order: list[tuple[str, str | None]]) -> list[tuple[str, str | None]]
def answers_for(ledger: list[dict], phase: str, unit: str | None) -> list[dict]

class ReviewExecutor(Executor):   # id="review"
    def __init__(self, store: ArtifactStore, revision_order: list[tuple[str, str | None]]) -> None
    # @handler on_questionnaire_done(done: PhaseDone, ctx)  -> open gate (once) / ReportTrigger
    # @response_handler on_answer(original: LedgerQuestionRequest, answer: str, ctx)
    # @handler on_revision_done(done: RevisionDone, ctx)    -> next RevisionTrigger / ReportTrigger
```

Behavioral contract (spec §8): entering the gate sets `review_completed` BEFORE prompting; empty/whitespace answer = declined (default stands, no re-run); answered questions re-run their `(phase, unit)` sequentially in `revision_order`; when the queue drains, send `ReportTrigger`. If the gate is entered with the flag already set (defensive; shouldn't occur in this graph), it forwards `ReportTrigger` without prompting.

- [ ] **Step 1: Write the failing tests**

`tests/test_review.py`:

```python
import pytest

from conftest import FakeCtx

from hotl_demo.artifacts import REPOS, ArtifactStore
from hotl_demo.phases import PhaseDone, ReportTrigger, RevisionDone, RevisionTrigger
from hotl_demo.review import (
    LedgerQuestionRequest,
    ReviewExecutor,
    affected_targets,
    answers_for,
)

ORDER = [
    ("discovery", None),
    ("deep_analysis", "oms-monolith"),
    ("deep_analysis", "oms-batch-recon"),
    ("enterprise_context", None),
    ("questionnaire", None),
]


@pytest.fixture()
def store(tmp_path):
    s = ArtifactStore(tmp_path / "run", repos=REPOS)
    s.raise_question("discovery", None, "Scope?", "recon undocumented", "in scope")
    s.raise_question("deep_analysis", "oms-batch-recon", "Secrets?", "hardcoded pw", "vault first")
    s.raise_question("enterprise_context", None, "Region?", "unspecified", "EU")
    return s


def test_affected_targets_ordered_and_deduped():
    ledger = [
        {"phase": "enterprise_context", "unit": None, "status": "answered"},
        {"phase": "discovery", "unit": None, "status": "answered"},
        {"phase": "discovery", "unit": None, "status": "answered"},   # second answered q, same phase
        {"phase": "deep_analysis", "unit": "oms-batch-recon", "status": "declined"},
    ]
    assert affected_targets(ledger, ORDER) == [("discovery", None), ("enterprise_context", None)]


def test_answers_for_filters_phase_unit_and_status():
    ledger = [
        {"id": "q-1", "phase": "deep_analysis", "unit": "oms-monolith", "status": "answered"},
        {"id": "q-2", "phase": "deep_analysis", "unit": "oms-batch-recon", "status": "answered"},
        {"id": "q-3", "phase": "deep_analysis", "unit": "oms-monolith", "status": "declined"},
    ]
    assert [a["id"] for a in answers_for(ledger, "deep_analysis", "oms-monolith")] == ["q-1"]


@pytest.mark.asyncio
async def test_gate_emits_one_request_per_open_question_and_sets_flag(store):
    ctx = FakeCtx()
    review = ReviewExecutor(store, ORDER)
    await review.on_questionnaire_done(PhaseDone("questionnaire"), ctx)
    assert store.review_completed() is True
    assert [r.question_id for r in ctx.requests] == ["q-1", "q-2", "q-3"]
    assert all(isinstance(r, LedgerQuestionRequest) for r in ctx.requests)
    assert ctx.sent == []  # nothing dispatched until answers arrive


@pytest.mark.asyncio
async def test_answers_drive_sequential_revisions_then_report(store):
    ctx = FakeCtx()
    review = ReviewExecutor(store, ORDER)
    await review.on_questionnaire_done(PhaseDone("questionnaire"), ctx)
    reqs = list(ctx.requests)
    # q-1 answered, q-2 answered, q-3 declined (whitespace)
    await review.on_answer(reqs[0], "recon is in scope", ctx)
    await review.on_answer(reqs[1], "rotate now, vault during migration", ctx)
    assert ctx.sent == []                              # still waiting for q-3
    await review.on_answer(reqs[2], "   ", ctx)
    assert len(ctx.sent) == 1                          # first revision dispatched
    t1 = ctx.sent[0]
    assert isinstance(t1, RevisionTrigger) and (t1.phase, t1.unit) == ("discovery", None)
    assert t1.answers[0]["human_answer"] == "recon is in scope"
    # ledger updated
    statuses = {e["id"]: e["status"] for e in store.read_ledger()}
    assert statuses == {"q-1": "answered", "q-2": "answered", "q-3": "declined"}
    # revision completes -> next affected target
    await review.on_revision_done(RevisionDone("discovery", None), ctx)
    t2 = ctx.sent[1]
    assert (t2.phase, t2.unit) == ("deep_analysis", "oms-batch-recon")
    # last revision completes -> report
    await review.on_revision_done(RevisionDone("deep_analysis", "oms-batch-recon"), ctx)
    assert isinstance(ctx.sent[2], ReportTrigger)


@pytest.mark.asyncio
async def test_all_declined_goes_straight_to_report(store):
    ctx = FakeCtx()
    review = ReviewExecutor(store, ORDER)
    await review.on_questionnaire_done(PhaseDone("questionnaire"), ctx)
    for r in list(ctx.requests):
        await review.on_answer(r, "", ctx)
    assert len(ctx.sent) == 1 and isinstance(ctx.sent[0], ReportTrigger)


@pytest.mark.asyncio
async def test_no_open_questions_goes_straight_to_report(tmp_path):
    empty_store = ArtifactStore(tmp_path / "run2", repos=REPOS)
    ctx = FakeCtx()
    review = ReviewExecutor(empty_store, ORDER)
    await review.on_questionnaire_done(PhaseDone("questionnaire"), ctx)
    assert ctx.requests == []
    assert len(ctx.sent) == 1 and isinstance(ctx.sent[0], ReportTrigger)
    assert empty_store.review_completed() is True


@pytest.mark.asyncio
async def test_review_once_guard(store):
    ctx = FakeCtx()
    review = ReviewExecutor(store, ORDER)
    store.set_review_completed()                       # gate already consumed
    await review.on_questionnaire_done(PhaseDone("questionnaire"), ctx)
    assert ctx.requests == []                          # never prompts again
    assert len(ctx.sent) == 1 and isinstance(ctx.sent[0], ReportTrigger)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/test_review.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'hotl_demo.review'`.

- [ ] **Step 3: Write `src/hotl_demo/review.py`**

```python
"""The human review gate: presents the ledger once, routes selective re-runs."""
from __future__ import annotations

from dataclasses import dataclass

from agent_framework import Executor, WorkflowContext, handler, response_handler

from .artifacts import ArtifactStore
from .phases import PhaseDone, ReportTrigger, RevisionDone, RevisionTrigger


@dataclass
class LedgerQuestionRequest:
    question_id: str
    phase: str
    unit: str | None
    question: str
    context: str
    default_assumption: str


def affected_targets(ledger: list[dict],
                     revision_order: list[tuple[str, str | None]]) -> list[tuple[str, str | None]]:
    answered = {(e["phase"], e["unit"]) for e in ledger if e["status"] == "answered"}
    return [t for t in revision_order if t in answered]


def answers_for(ledger: list[dict], phase: str, unit: str | None) -> list[dict]:
    return [e for e in ledger
            if e["status"] == "answered" and e["phase"] == phase and e["unit"] == unit]


class ReviewExecutor(Executor):
    def __init__(self, store: ArtifactStore,
                 revision_order: list[tuple[str, str | None]]) -> None:
        super().__init__(id="review")
        self._store = store
        self._revision_order = revision_order
        self._awaiting = 0
        self._queue: list[RevisionTrigger] = []

    @handler
    async def on_questionnaire_done(
        self, done: PhaseDone,
        ctx: WorkflowContext[ReportTrigger | RevisionTrigger],
    ) -> None:
        if self._store.review_completed():
            # Review runs exactly once per run; defensive guard.
            await ctx.send_message(ReportTrigger())
            return
        self._store.set_review_completed()  # set on ENTRY, before prompting (spec 8)
        open_qs = self._store.open_questions()
        if not open_qs:
            await ctx.send_message(ReportTrigger())
            return
        print(f"\n== REVIEW - {len(open_qs)} open questions ==")
        self._awaiting = len(open_qs)
        for q in open_qs:
            await ctx.request_info(
                request_data=LedgerQuestionRequest(
                    question_id=q["id"], phase=q["phase"], unit=q["unit"],
                    question=q["question"], context=q["context"],
                    default_assumption=q["default_assumption"],
                ),
                response_type=str,
            )

    @response_handler
    async def on_answer(
        self, original: LedgerQuestionRequest, answer: str,
        ctx: WorkflowContext[ReportTrigger | RevisionTrigger],
    ) -> None:
        text = (answer or "").strip()
        self._store.resolve_question(
            original.question_id, "answered" if text else "declined", text or None
        )
        self._awaiting -= 1
        if self._awaiting > 0:
            return
        ledger = self._store.read_ledger()
        targets = affected_targets(ledger, self._revision_order)
        self._queue = [
            RevisionTrigger(phase, unit, answers_for(ledger, phase, unit))
            for phase, unit in targets
        ]
        if self._queue:
            pretty = ", ".join(f"{p}[{u}]" if u else p for p, u in targets)
            print(f"Re-running affected: {pretty}")
        await self._dispatch_next(ctx)

    @handler
    async def on_revision_done(
        self, done: RevisionDone,
        ctx: WorkflowContext[ReportTrigger | RevisionTrigger],
    ) -> None:
        await self._dispatch_next(ctx)

    async def _dispatch_next(self, ctx) -> None:
        if self._queue:
            await ctx.send_message(self._queue.pop(0))
        else:
            await ctx.send_message(ReportTrigger())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/test_review.py -v`
Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hotl_demo/review.py tests/test_review.py
git commit -m "feat: review gate executor with single-shot gate and selective re-runs"
```

---

### Task 8: FinalReportExecutor + adjudication log

**Files:**
- Create: `src/hotl_demo/report.py`
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: `ArtifactStore` (Task 2); `ReportTrigger` (Task 5); `FakeAgent`/`FakeCtx` (Task 6).
- Produces:

```python
def render_adjudication_log(ledger: list[dict]) -> str      # deterministic markdown table
def build_report_prompt(memory_text: str, reports: dict[str, str], ledger: list[dict]) -> str

class FinalReportExecutor(Executor):                         # id="final_report"
    def __init__(self, store: ArtifactStore, agent: Any | None = None) -> None
    # @handler on_report(trig: ReportTrigger, ctx) -> writes final_report.md, ctx.yield_output(str(path))
```

- [ ] **Step 1: Write the failing tests**

`tests/test_report.py`:

```python
import pytest

from conftest import FakeAgent, FakeCtx

from hotl_demo.artifacts import REPOS, ArtifactStore
from hotl_demo.phases import ReportTrigger
from hotl_demo.report import FinalReportExecutor, build_report_prompt, render_adjudication_log

LEDGER = [
    {"id": "q-1", "phase": "discovery", "unit": None, "question": "Scope?",
     "context": "c", "default_assumption": "in scope", "status": "answered",
     "human_answer": "recon in scope", "asked_at": "t"},
    {"id": "q-2", "phase": "enterprise_context", "unit": None, "question": "Region?",
     "context": "c", "default_assumption": "EU", "status": "declined",
     "human_answer": None, "asked_at": "t"},
    {"id": "q-3", "phase": "deep_analysis", "unit": "oms-monolith", "question": "RTO?",
     "context": "c", "default_assumption": "4h", "status": "open",
     "human_answer": None, "asked_at": "t"},
]


def test_render_adjudication_log_covers_all_statuses():
    table = render_adjudication_log(LEDGER)
    assert "| q-1 |" in table and "answered" in table and "recon in scope" in table
    assert "| q-2 |" in table and "declined - default applied: EU" in table
    assert "| q-3 |" in table and "open - default assumption applied: 4h" in table
    assert "deep_analysis[oms-monolith]" in table


def test_render_adjudication_log_empty():
    assert "No questions" in render_adjudication_log([])


def test_build_report_prompt_includes_everything():
    prompt = build_report_prompt("{\"mem\": 1}", {"phase_01_discovery.md": "D-REPORT"}, LEDGER)
    assert "{\"mem\": 1}" in prompt
    assert "phase_01_discovery.md" in prompt and "D-REPORT" in prompt
    assert "q-1" in prompt and "recon in scope" in prompt
    assert "scorecard" in prompt.lower()


@pytest.mark.asyncio
async def test_executor_writes_report_and_yields_path(tmp_path):
    store = ArtifactStore(tmp_path / "run", repos=REPOS)
    store.write_report("phase_01_discovery.md", "D")
    for e in LEDGER:
        store.raise_question(e["phase"], e["unit"], e["question"], e["context"],
                             e["default_assumption"])
    store.resolve_question("q-1", "answered", "recon in scope")
    store.resolve_question("q-2", "declined", None)
    agent = FakeAgent(["# Readiness Report\nverdict..."])
    ctx = FakeCtx()
    await FinalReportExecutor(store, agent=agent).on_report(ReportTrigger(), ctx)
    text = store.read_report("final_report.md")
    assert text.startswith("# Readiness Report")
    assert "## Adjudication log" in text
    assert "recon in scope" in text            # deterministic table present even though
    assert "q-3" in text                       # the fake agent never mentioned the ledger
    assert ctx.outputs == [str(store.run_dir / "final_report.md")]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/test_report.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'hotl_demo.report'`.

- [ ] **Step 3: Write `src/hotl_demo/report.py`**

```python
"""Final report: LLM-composed verdict + deterministic adjudication log."""
from __future__ import annotations

from typing import Any

from agent_framework import Agent, Executor, WorkflowContext, handler
from agent_framework.ollama import OllamaChatClient
from typing_extensions import Never

from .artifacts import ArtifactStore
from .phases import ReportTrigger


def render_adjudication_log(ledger: list[dict]) -> str:
    if not ledger:
        return "No questions were raised during this run.\n"
    lines = [
        "| id | phase | question | resolution |",
        "|---|---|---|---|",
    ]
    for e in ledger:
        where = f"{e['phase']}[{e['unit']}]" if e["unit"] else e["phase"]
        if e["status"] == "answered":
            resolution = f"answered: {e['human_answer']}"
        elif e["status"] == "declined":
            resolution = f"declined - default applied: {e['default_assumption']}"
        else:
            resolution = f"open - default assumption applied: {e['default_assumption']}"
        lines.append(f"| {e['id']} | {where} | {e['question']} | {resolution} |")
    return "\n".join(lines) + "\n"


def build_report_prompt(memory_text: str, reports: dict[str, str], ledger: list[dict]) -> str:
    report_blocks = "\n\n".join(f"--- {name} ---\n{text}" for name, text in reports.items())
    adjudications = "\n".join(
        f"- {e['id']} [{e['status']}] {e['question']}"
        + (f" -> HUMAN: {e['human_answer']}" if e["human_answer"] else
           f" -> default: {e['default_assumption']}")
        for e in ledger
    ) or "none"
    return f"""You are writing the final cloud migration readiness report for
Meridian Retail's Order Management System, synthesizing the phase reports and
shared memory below. Human adjudications are authoritative.

Structure your markdown report exactly as:
# Cloud Migration Readiness Report - Meridian Retail OMS
## Executive summary
## Readiness scorecard
(a table scoring: compute, data, integrations, security, operations - one of
Ready / Ready with conditions / Not ready, each with a one-line reason)
## Migration recommendation
(6R approach, target services, sequencing, prerequisites, key risks)

## HUMAN ADJUDICATIONS
{adjudications}

## SHARED MEMORY
```json
{memory_text}
```

## PHASE REPORTS
{report_blocks}

Write the report now. Do not include an adjudication log section - it is
appended automatically.
"""


class FinalReportExecutor(Executor):
    def __init__(self, store: ArtifactStore, agent: Any | None = None) -> None:
        super().__init__(id="final_report")
        self._store = store
        self._agent = agent or Agent(
            client=OllamaChatClient(),
            name="final_report",
            instructions="You write crisp executive assessment reports.",
        )

    @handler
    async def on_report(self, trig: ReportTrigger, ctx: WorkflowContext[Never, str]) -> None:
        ledger = self._store.read_ledger()
        prompt = build_report_prompt(
            self._store.memory_text(), self._store.read_all_reports(), ledger
        )
        result = await self._agent.run(prompt)
        text = (result.text or "(no text returned by the model)")
        text += "\n\n## Adjudication log\n\n" + render_adjudication_log(ledger)
        path = self._store.write_report("final_report.md", text)
        print("  final_report: written")
        await ctx.yield_output(str(path))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/test_report.py -v`
Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hotl_demo/report.py tests/test_report.py
git commit -m "feat: final report executor with deterministic adjudication log"
```

---
### Task 9: Workflow graph assembly (pipeline.py)

**Files:**
- Create: `src/hotl_demo/pipeline.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: everything above — `PhaseExecutor`, `build_phase_specs`, messages (Task 5/6); `ReviewExecutor` (Task 7); `FinalReportExecutor` (Task 8); `ArtifactStore`, `REPOS` (Task 2).
- Produces:

```python
def is_type(message_type: type) -> Callable[[Any], bool]          # edge condition factory
def revision_for(phase: str, unit: str | None) -> Callable[[Any], bool]
class JoinAnalyses(Executor):                                      # id="join"
    def __init__(self, expected: int) -> None
    # @handler on_analysis(done: AnalysisDone, ctx) -> PhaseDone("deep_analysis") once all repos reported
def build_workflow(store: ArtifactStore, base_dir: Path,
                   scratchpad_path: Path = SCRATCHPAD_PATH)        # -> built workflow object
```

Graph contract (spec §7): initial flow discovery → both analyzers (fan-out via two conditioned edges) → join → enterprise_context → questionnaire → review → final_report; `RevisionDone` from ANY phase routes straight to review (bypassing join); `RevisionTrigger` routes from review to exactly the matching `(phase, unit)` executor. Deliberate deviation from the spec's sample reference, decided during planning: we do NOT use `add_fan_in_edges` — its barrier semantics with mixed message types are unverified, so the join is a 4-line executor we own and unit-test. Parallelism is unchanged (both analyzers become ready in the same superstep).

- [ ] **Step 1: Write the failing tests**

`tests/test_pipeline.py`:

```python
from pathlib import Path

import pytest

from conftest import FakeCtx

from hotl_demo.artifacts import REPOS, ArtifactStore
from hotl_demo.phases import AnalysisDone, PhaseDone, ReportTrigger, RevisionDone, RevisionTrigger
from hotl_demo.pipeline import JoinAnalyses, build_workflow, is_type, revision_for


def test_is_type_condition():
    cond = is_type(PhaseDone)
    assert cond(PhaseDone("discovery")) is True
    assert cond(RevisionDone("discovery")) is False
    assert cond("random string") is False


def test_revision_for_condition_matches_phase_and_unit():
    cond = revision_for("deep_analysis", "oms-monolith")
    assert cond(RevisionTrigger("deep_analysis", "oms-monolith", [])) is True
    assert cond(RevisionTrigger("deep_analysis", "oms-batch-recon", [])) is False
    assert cond(RevisionTrigger("discovery", None, [])) is False
    assert cond(ReportTrigger()) is False


@pytest.mark.asyncio
async def test_join_waits_for_all_analyzers():
    join = JoinAnalyses(expected=2)
    ctx = FakeCtx()
    await join.on_analysis(AnalysisDone("oms-monolith"), ctx)
    assert ctx.sent == []
    await join.on_analysis(AnalysisDone("oms-monolith"), ctx)  # duplicate unit: still waiting
    assert ctx.sent == []
    await join.on_analysis(AnalysisDone("oms-batch-recon"), ctx)
    assert ctx.sent == [PhaseDone("deep_analysis")]


def test_build_workflow_smoke(tmp_path, monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "test-model")
    store = ArtifactStore(tmp_path / "run", repos=REPOS)
    workflow = build_workflow(store, Path("sample_data"), scratchpad_path=tmp_path / "pad.md")
    assert workflow is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/test_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'hotl_demo.pipeline'`.

- [ ] **Step 3: Write `src/hotl_demo/pipeline.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/test_pipeline.py -v`
Expected: all 4 tests PASS. If `test_build_workflow_smoke` fails inside `OllamaChatClient()` construction (e.g. it validates the model eagerly), the fix belongs in the test only: monkeypatch `hotl_demo.phases.OllamaChatClient` and `hotl_demo.report.OllamaChatClient` with a stub class — production code stays as-is.

- [ ] **Step 5: Commit**

```bash
git add src/hotl_demo/pipeline.py tests/test_pipeline.py
git commit -m "feat: workflow graph with fan-out analyzers, join, review routing"
```

---

### Task 10: CLI runner (main.py)

**Files:**
- Modify: `src/hotl_demo/main.py` (replace the Task 1 stub entirely)
- Test: `tests/test_main.py`

**Interfaces:**
- Consumes: `build_workflow` (Task 9); `LedgerQuestionRequest` (Task 7); `ArtifactStore`, `REPOS` (Task 2); `ensure_scratchpad`, `SCRATCHPAD_PATH` (Task 3).
- Produces: `run()` — the `poetry run demo` entry point (already wired in Task 1's pyproject); `model_present(tags: dict, model: str) -> bool` and `preflight(base_url: str, model: str) -> None` for testing.

- [ ] **Step 1: Write the failing tests**

`tests/test_main.py`:

```python
from hotl_demo.main import model_present

TAGS = {"models": [{"name": "gemma4:31b"}, {"name": "qwen3.6:latest"}]}


def test_model_present_exact_tag():
    assert model_present(TAGS, "gemma4:31b") is True


def test_model_present_base_name_matches_any_tag():
    assert model_present(TAGS, "qwen3.6") is True


def test_model_absent():
    assert model_present(TAGS, "gemma4:9b") is False
    assert model_present({}, "gemma4:31b") is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/test_main.py -v`
Expected: FAIL — `ImportError: cannot import name 'model_present'`.

- [ ] **Step 3: Replace `src/hotl_demo/main.py`**

```python
"""CLI runner: preflight, run the workflow, prompt the human at the review gate."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import urllib.request
from datetime import datetime
from pathlib import Path

from .artifacts import REPOS, ArtifactStore
from .review import LedgerQuestionRequest
from .tools import SCRATCHPAD_PATH, ensure_scratchpad

DEFAULT_MODEL = "gemma4:31b"


def model_present(tags: dict, model: str) -> bool:
    names = {m.get("name", "") for m in tags.get("models", [])}
    return model in names or any(n.split(":", 1)[0] == model for n in names)


def preflight(base_url: str, model: str) -> None:
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=5) as resp:
            tags = json.load(resp)
    except OSError as exc:
        raise SystemExit(
            f"Ollama not reachable at {base_url} - is it running? ('ollama serve')\n{exc}"
        )
    if not model_present(tags, model):
        raise SystemExit(f"Model '{model}' not found in Ollama. Run: ollama pull {model}")
    print(f"Preflight: Ollama OK, {model} present.")


def _prompt_human(q: LedgerQuestionRequest) -> str:
    where = f"{q.phase}[{q.unit}]" if q.unit else q.phase
    print(f"\n[{q.question_id}] ({where}) {q.question}")
    print(f"      Evidence: {q.context}")
    print(f"      Default if declined: {q.default_assumption}")
    return input("      Your answer (ENTER to decline): ")


async def _amain() -> None:
    parser = argparse.ArgumentParser(description="HOTL cloud migration readiness demo")
    parser.add_argument("--model", default=os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL),
                        help="Ollama model tag (default: %(default)s)")
    parser.add_argument("--data", type=Path, default=Path("sample_data"),
                        help="sample data directory")
    args = parser.parse_args()
    os.environ["OLLAMA_MODEL"] = args.model  # OllamaChatClient reads this
    base_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    preflight(base_url, args.model)
    ensure_scratchpad(SCRATCHPAD_PATH)

    run_dir = Path("output") / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    store = ArtifactStore(run_dir, REPOS)
    # import here so --help and preflight failures never touch the framework
    from .pipeline import build_workflow
    workflow = build_workflow(store, args.data)

    responses: dict[str, str] | None = None
    while True:
        stream = (workflow.run("start", stream=True) if responses is None
                  else workflow.run(stream=True, responses=responses))
        pending: dict[str, LedgerQuestionRequest] = {}
        async for event in stream:
            if event.type == "request_info" and isinstance(event.data, LedgerQuestionRequest):
                pending[event.request_id] = event.data
            elif event.type == "output":
                print(f"\nFinal report: {event.data}")
        if not pending:
            break
        responses = {rid: _prompt_human(q) for rid, q in pending.items()}
    print(f"Run artifacts: {run_dir}")


def run() -> None:
    try:
        asyncio.run(_amain())
    except KeyboardInterrupt:
        print("\nAborted. Artifacts written so far persist under output/.")


if __name__ == "__main__":
    run()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/test_main.py -v`
Expected: 3 tests PASS.

Run: `poetry run pytest`
Expected: full suite PASSES (the `ollama`-marked E2E from Task 11 doesn't exist yet; everything else green).

Run: `poetry run demo --help`
Expected: argparse usage text showing `--model` and `--data` (no Ollama needed for `--help`).

- [ ] **Step 5: Commit**

```bash
git add src/hotl_demo/main.py tests/test_main.py
git commit -m "feat: CLI runner with preflight and interactive review gate"
```

---
### Task 11: Opt-in live E2E test, README, final verification

**Files:**
- Create: `tests/test_e2e_ollama.py`
- Modify: `README.md` (replace stub if any; full content below)

**Interfaces:**
- Consumes: the entire pipeline (`build_workflow`), `ArtifactStore`, `LedgerQuestionRequest`.
- Produces: the demo's documentation and the live proof harness. Nothing downstream.

- [ ] **Step 1: Write `tests/test_e2e_ollama.py`**

Drives the real workflow with the real model, scripting review answers instead of stdin: answer the first question, decline the rest. Assertions are deliberately structural (files/statuses), not content-based — LLM output varies.

```python
"""Live E2E: full pipeline against local Ollama. Opt-in:

    OLLAMA_E2E=1 poetry run pytest -m ollama -s

Takes many minutes on gemma4:31b (7+ LLM calls).
"""
import asyncio
import os
from pathlib import Path

import pytest

from hotl_demo.artifacts import REPOS, ArtifactStore
from hotl_demo.pipeline import build_workflow
from hotl_demo.review import LedgerQuestionRequest

pytestmark = pytest.mark.ollama

ANSWER = ("Yes - reconciliation is in scope. Also: Azure is authoritative; "
          "treat the AWS S3 code as legacy to be replaced.")


@pytest.mark.skipif(os.environ.get("OLLAMA_E2E") != "1", reason="set OLLAMA_E2E=1 to run")
def test_full_pipeline_with_scripted_review(tmp_path):
    os.environ.setdefault("OLLAMA_MODEL", "gemma4:31b")
    store = ArtifactStore(tmp_path / "run", repos=REPOS)
    pad = tmp_path / "scratchpad.md"
    pad.write_text("Prioritise security findings and be concise.", encoding="utf-8")
    workflow = build_workflow(store, Path("sample_data"), scratchpad_path=pad)

    async def drive() -> list:
        responses: dict[str, str] | None = None
        answered_once = False
        outputs: list = []
        for _ in range(10):  # safety bound on resume cycles
            stream = (workflow.run("start", stream=True) if responses is None
                      else workflow.run(stream=True, responses=responses))
            pending: dict[str, LedgerQuestionRequest] = {}
            async for event in stream:
                if event.type == "request_info" and isinstance(event.data, LedgerQuestionRequest):
                    pending[event.request_id] = event.data
                elif event.type == "output":
                    outputs.append(event.data)
            if not pending:
                return outputs
            responses = {}
            for rid in pending:
                responses[rid] = "" if answered_once else ANSWER
                answered_once = True
        raise AssertionError("workflow did not complete within resume bound")

    outputs = asyncio.run(drive())

    assert outputs, "workflow yielded no final output"
    final = store.read_report("final_report.md")
    assert final and "## Adjudication log" in final
    assert store.review_completed() is True
    ledger = store.read_ledger()
    assert ledger, "pipeline raised no questions - plantings failed"
    assert any(e["status"] == "answered" for e in ledger)
    assert any(e["status"] == "declined" for e in ledger)
    assert len(store.read_all_reports()) >= 4  # discovery, 2x deep_analysis, ec, questionnaire
```

- [ ] **Step 2: Run the LLM-free suite to confirm the E2E stays excluded**

Run: `poetry run pytest -v`
Expected: all unit tests PASS; `test_e2e_ollama.py` DESELECTED (the `addopts = "-m 'not ollama'"` filter).

- [ ] **Step 3: Write `README.md`**

````markdown
# agent-framework-hotl

Human-on-the-loop (HOTL) demo on [Microsoft Agent Framework](https://github.com/microsoft/agent-framework):
a multi-phase agent pipeline that assesses a fictional legacy system's cloud
migration readiness, accumulates a **ledger of questions** needing human
adjudication, pauses **exactly once** at a review gate, selectively re-runs
affected phases with the human's answers, and accepts freeform steering via a
**scratchpad** file read by agents through a tool call.

Design spec: `docs/superpowers/specs/2026-07-14-hotl-pipeline-design.md`

## The pipeline

```
                 +- analyze[oms-monolith] --+
discovery -> fan-out                        +-> join -> enterprise_context -> questionnaire -> REVIEW -> final_report
                 +- analyze[oms-batch-recon]+                                                   |
                        ^        ^                            ^                    ^            |
                        +--------+----------------------------+--------------------+-- re-runs -+
```

- **discovery** - what does this system REALLY do (docs vs code)?
- **deep_analysis** - one agent per repo, in parallel; per-repo reports
- **enterprise_context** - corporate cloud strategy + security standards overlay
- **questionnaire** - fills the standard readiness question template
- **review** - the human gate: answer (authoritative) or decline (default assumption applies)
- **final_report** - readiness scorecard + recommendation + adjudication log

## Prerequisites

- Python 3.10+ and [Poetry](https://python-poetry.org/)
- [Ollama](https://ollama.com/) running locally with the model pulled:

```bash
ollama pull gemma4:31b
```

## Run the demo

```bash
poetry install
poetry run demo                 # or: poetry run demo --model <other-tools-capable-model>
```

The four phases run autonomously (the two repo analyzers in parallel), each
writing a markdown report, updating `memory.json`, and appending questions to
`ledger.jsonl`. Then the review gate presents every open question:

```
[q-1] (discovery) Is reconciliation functionality in migration scope?
      Evidence: oms-batch-recon performs financial reconciliation; absent from the architecture doc.
      Default if declined: in scope.
      Your answer (ENTER to decline): _
```

Type an answer to make it authoritative (the raising phase re-runs with it),
or press ENTER to decline (the stated default stands). The review gate runs
only once per pipeline run - questions raised during re-runs are documented
in the final report as "open - default assumption applied".

Artifacts land in `output/run_<timestamp>/`:
phase reports, `memory.json`, `ledger.jsonl`, `final_report.md`.

## Steering via the scratchpad

`scratchpad.md` (repo root) starts empty. Write guidance into it at any time -
before a run or while one is executing:

```markdown
Focus on data-layer risks. Assume the migration window is Q3.
Be terse; bullet points only.
```

Every phase agent calls the `read_scratchpad` tool before working and follows
what it finds. This is the basic steering channel into an otherwise closed
pipeline.

## The sample data

Everything under `sample_data/` is synthetic: three enterprise PDFs
(regenerate with `poetry run python scripts/make_pdfs.py` after editing
`docs_src/`), two fake legacy repos, and a questionnaire template. The corpus
has **planted gaps and conflicts** (Azure mandate vs `boto3` in code, missing
RTO/RPO, unspecified data-residency region, hardcoded credentials, ...) so the
agents reliably find questions worth asking a human.

## Tests

```bash
poetry run pytest                       # fast, LLM-free
OLLAMA_E2E=1 poetry run pytest -m ollama -s   # full live pipeline (slow)
```
````

(PowerShell equivalent for the live test: `$env:OLLAMA_E2E="1"; poetry run pytest -m ollama -s`.)

- [ ] **Step 4: Full verification**

Run: `poetry run pytest`
Expected: entire unit suite PASSES.

Run: `poetry run demo`
Expected (manual, requires Ollama; budget 10-30 min on gemma4:31b): preflight line; four phase progress lines (analyzers interleaved); `== REVIEW - N open questions ==` with at least 3 questions echoing the plantings; answer one, decline the rest; `Re-running affected: ...` for the answered question's phase; `Final report: output\run_...\final_report.md`. Open the final report: scorecard + recommendation + adjudication log listing your answer verbatim. Confirm `scratchpad.md` steering works by writing "Be extremely terse." into it before the run and observing shorter reports.

- [ ] **Step 5: Commit**

```bash
git add tests/test_e2e_ollama.py README.md
git commit -m "docs+test: README and opt-in live E2E pipeline test"
```

---

## Plan-wide verification notes for the executor

- Framework API assumptions in this plan were read from the official samples at `microsoft/agent-framework@main` (`python/samples/...`): `WorkflowBuilder(start_executor=...).add_edge(a, b, condition=fn)`, `Executor` + `@handler`/`@response_handler`, `ctx.send_message/request_info/yield_output`, `event.type in {"request_info", "output"}`, `workflow.run("start", stream=True)` / `workflow.run(stream=True, responses={...})`, `Agent(client=OllamaChatClient(), tools=[...])`, `@tool(approval_mode="never_require")`, `result.text`. If `poetry install` resolves a NEWER agent-framework than 1.11.x and something drifted, pin exactly (`agent-framework (==1.11.0)`) rather than chasing the API.
- If any framework call fails at runtime, the local sparse clone of the samples used while planning may still exist under the session scratchpad (`.../scratchpad/af/python/samples/`) — otherwise fetch matching samples from GitHub before improvising.
- gemma4:31b tool-calling was verified via `ollama show gemma4:31b` (capabilities: completion, vision, tools, thinking).





