"""File-backed run artifacts: shared memory (json), question ledger (jsonl), reports (md).

The files ARE the pipeline's long-term memory story; executors share one
:class:`ArtifactStore` instance per run. A single lock serializes access
because the two deep_analysis analyzers run concurrently, and every write
lands atomically (temp file + ``os.replace``) because a human may have the
files open in an editor mid-run.
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
    """Thread-safe reader/writer for one run's artifacts.

    Owns three artifact kinds inside ``run_dir``:

    * ``memory.json`` - structured long-term memory, one section per phase
      (``deep_analysis`` nests one sub-section per repo).
    * ``ledger.jsonl`` - append-only question ledger; status changes rewrite
      the file atomically but never drop lines.
    * ``phase_*.md`` / ``final_report.md`` - markdown reports.

    Example:
        >>> store = ArtifactStore(Path("output/run_x"))
        >>> qid = store.raise_question("discovery", None, "In scope?",
        ...                            "docs omit recon", "in scope")
        >>> store.resolve_question(qid, "answered", "yes")["status"]
        'answered'
    """

    def __init__(self, run_dir: Path, repos: tuple[str, ...] = REPOS) -> None:
        """Create/open a run directory, seeding an empty memory if new.

        Args:
            run_dir: Directory for this run's artifacts; created if missing.
                Reopening an existing run directory preserves its memory.
            repos: Repo names that become the ``deep_analysis`` sub-sections.
        """
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._memory_path = self.run_dir / "memory.json"
        self._ledger_path = self.run_dir / "ledger.jsonl"
        if not self._memory_path.exists():
            # deep_analysis is the only unit-nested section: one dict per repo.
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
        """Return the parsed ``memory.json``.

        Returns:
            The full memory document
            (``{"run_id", "review_completed", "sections"}``).
        """
        # Locked: on Windows an unlocked open can collide with a concurrent
        # locked os.replace (no FILE_SHARE_DELETE) and raise PermissionError.
        with self._lock:
            return self._read_memory_unlocked()

    def _read_memory_unlocked(self) -> dict:
        """Read memory without taking the lock; callers must hold it."""
        return json.loads(self._memory_path.read_text(encoding="utf-8"))

    def memory_text(self) -> str:
        """Return memory as pretty-printed JSON, ready to embed in a prompt.

        Example:
            >>> '"review_completed"' in store.memory_text()  # doctest: +SKIP
            True
        """
        return json.dumps(self.read_memory(), indent=2)

    def update_memory(self, phase: str, unit: str | None, key: str, value: str) -> None:
        """Merge one finding into a phase's memory section.

        Args:
            phase: Section name; must be one of :data:`PHASES`.
            unit: Repo name - required (and validated) when ``phase`` is
                ``"deep_analysis"``, ignored otherwise.
            key: Short snake_case finding key, e.g. ``"runtime"``.
            value: Concise factual value.

        Raises:
            KeyError: Unknown ``phase``, or unknown ``unit`` for
                deep_analysis - a misbound tool should fail loudly.
        """
        with self._lock:
            # Read-modify-write under the lock so concurrent analyzers can
            # never lose each other's keys.
            mem = self._read_memory_unlocked()
            target = mem["sections"][phase]
            if phase == "deep_analysis":
                target = target[unit]  # KeyError for unknown repo is intentional
            target[key] = value
            self._write_memory(mem)

    def memory_key_count(self, phase: str, unit: str | None) -> int:
        """Count the findings recorded in one phase's section.

        Used by the executor's nudge check: an unchanged count after a run
        means the agent never called ``update_memory``.

        Args:
            phase: Section name.
            unit: Repo name for deep_analysis, else ignored.

        Returns:
            Number of keys currently in that section.
        """
        section = self.read_memory()["sections"][phase]
        if phase == "deep_analysis":
            section = section[unit]
        return len(section)

    def review_completed(self) -> bool:
        """Return whether the single review gate has already been entered."""
        return bool(self.read_memory()["review_completed"])

    def set_review_completed(self) -> None:
        """Latch the review-once flag; there is deliberately no unset."""
        with self._lock:
            mem = self._read_memory_unlocked()
            mem["review_completed"] = True
            self._write_memory(mem)

    def _write_memory(self, data: dict) -> None:
        """Serialize and atomically replace ``memory.json``."""
        _atomic_write(self._memory_path, json.dumps(data, indent=2))

    # -- ledger ---------------------------------------------------------

    def raise_question(self, phase: str, unit: str | None, question: str,
                       context: str, default_assumption: str) -> str:
        """Append an ``open`` question to the ledger and return its id.

        Args:
            phase: Raising phase; must be one of :data:`PHASES`.
            unit: Raising repo for deep_analysis questions, else ``None``.
            question: The question needing human adjudication.
            context: The evidence that motivated it.
            default_assumption: What the pipeline proceeds with until (and
                unless) a human answers.

        Returns:
            The assigned id, ``"q-<n>"``, numbered by ledger position.

        Raises:
            KeyError: Unknown ``phase``.

        Example:
            >>> store.raise_question("discovery", None, "In scope?",
            ...                      "docs omit recon", "in scope")  # doctest: +SKIP
            'q-1'
        """
        if phase not in PHASES:
            raise KeyError(phase)
        with self._lock:
            # Id assignment and append happen under one lock acquisition so
            # concurrent analyzers can never mint duplicate ids.
            # ponytail: re-reads the ledger to number the next id - fine at
            # demo scale, index it if ledgers ever grow large.
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
        """Return every ledger entry in raise order (empty list if none)."""
        with self._lock:
            return self._read_ledger_unlocked()

    def open_questions(self) -> list[dict]:
        """Return only the entries still awaiting adjudication.

        Example:
            >>> [q["id"] for q in store.open_questions()]  # doctest: +SKIP
            ['q-2', 'q-3']
        """
        return [e for e in self.read_ledger() if e["status"] == "open"]

    def resolve_question(self, question_id: str, status: str, human_answer: str | None) -> dict:
        """Mark one question ``answered`` or ``declined``.

        The ledger file is rewritten atomically with the updated entry;
        raises are the only appends, so ordering and ids never change.

        Args:
            question_id: Id previously returned by :meth:`raise_question`.
            status: ``"answered"`` or ``"declined"``.
            human_answer: The authoritative human text for ``answered``,
                ``None`` for ``declined``.

        Returns:
            The updated entry.

        Raises:
            KeyError: No entry with that id exists.
        """
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
        """Parse the JSONL ledger without locking; callers must hold the lock."""
        if not self._ledger_path.exists():
            return []
        text = self._ledger_path.read_text(encoding="utf-8")
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    # -- reports --------------------------------------------------------

    def write_report(self, filename: str, text: str) -> Path:
        """Atomically write a markdown report into the run directory.

        Args:
            filename: Report file name, e.g. ``"phase_01_discovery.md"``.
            text: Full markdown content (revisions overwrite).

        Returns:
            The report's path.
        """
        path = self.run_dir / filename
        _atomic_write(path, text)
        return path

    def read_report(self, filename: str) -> str:
        """Return a report's text, or ``""`` when it does not exist yet."""
        path = self.run_dir / filename
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def read_all_reports(self) -> dict[str, str]:
        """Return all PHASE reports keyed by filename, sorted by name.

        ``final_report.md`` is deliberately excluded - this feeds the final
        report's own prompt.

        Example:
            >>> list(store.read_all_reports())  # doctest: +SKIP
            ['phase_01_discovery.md', 'phase_02_deep_analysis_oms-monolith.md']
        """
        return {
            p.name: p.read_text(encoding="utf-8")
            for p in sorted(self.run_dir.glob("phase_*.md"))
        }


def _atomic_write(path: Path, text: str) -> None:
    """Write via temp file + ``os.replace`` so readers never see a torn file.

    Args:
        path: Final destination.
        text: Full content to write.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
