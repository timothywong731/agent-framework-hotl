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
