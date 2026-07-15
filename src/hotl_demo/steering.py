"""Live scratchpad steering: notice mid-run operator edits and push them to
the agents still working.

The steering file is normally *pulled* by the ``read_scratchpad`` tool, once,
at the start of a phase. This module adds the *push* half: a per-agent
watermark polled after every tool call, and function middleware that hands any
change to the framework's ``MessageInjectionMiddleware`` for delivery on the
agent's next model call.

Kept out of ``tools.py``, whose contract is "tools the agent calls" - this is a
delivery channel, not a tool.
"""
from __future__ import annotations

from pathlib import Path


class ScratchpadWatch:
    """Per-agent watermark over the steering file.

    Deliberately NOT shared between agents: the two deep_analysis analyzers
    run concurrently and each needs its own notion of "have I told THIS agent
    yet".

    Args:
        path: The steering file to watch. It need not exist.

    Example:
        >>> from pathlib import Path
        >>> watch = ScratchpadWatch(Path("scratchpad.md"))  # doctest: +SKIP
        >>> watch.poll()  # first call baselines  # doctest: +SKIP
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._seen: str | None = None

    def poll(self) -> str | None:
        """Return the scratchpad text if it changed since the last poll.

        Compared by content rather than mtime, so re-saving an unedited file
        is correctly silent.

        Returns:
            The current text when it differs from the last poll; ``None`` when
            unchanged, and ``None`` on the very first call - that call only
            baselines, so an agent is never handed back the content it just
            fetched with ``read_scratchpad``.
        """
        text = self._path.read_text(encoding="utf-8") if self._path.exists() else ""
        if text == self._seen:
            return None
        # First poll establishes the baseline and reports nothing; every later
        # change reports, including a clear to "" (which the middleware skips
        # but which must still advance the watermark).
        first, self._seen = self._seen is None, text
        return None if first else text
