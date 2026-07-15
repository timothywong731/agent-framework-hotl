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

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from agent_framework import FunctionInvocationContext, function_middleware


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
            unchanged, when the file is momentarily unreadable, and on the very
            first call - that call only baselines, so an agent is never handed
            back the content it just fetched with ``read_scratchpad``.
        """
        try:
            # The human picks the editor and therefore the encoding: a UTF-16
            # save (Notepad "Unicode", PowerShell 5.1 ">") must not raise here.
            # This runs AFTER call_next(), so an exception would report an
            # already-succeeded tool call as failed - and the model might then
            # repeat a side effect that already landed. Same errors="replace"
            # idiom as read_file in tools.py.
            text = (
                self._path.read_text(encoding="utf-8", errors="replace")
                if self._path.exists()
                else ""
            )
        except OSError:
            # Locked mid-save, or deleted between exists() and the read. Report
            # nothing and leave the watermark untouched, so the next tool call
            # picks the edit up rather than losing it.
            return None
        if text == self._seen:
            return None
        # First poll establishes the baseline and reports nothing; every later
        # change reports, including a clear to "" (which the middleware skips
        # but which must still advance the watermark).
        first, self._seen = self._seen is None, text
        return None if first else text


# Concatenated, NEVER str.format-ed: the scratchpad is human-written and
# routinely contains braces (code snippets, JSON). Same gotcha as the memory
# nudge in phases.py.
_NOTICE_HEAD = (
    "\n[OPERATOR STEERING UPDATE - the human edited the scratchpad while you "
    "were working. Its current contents are:]\n"
)
_NOTICE_TAIL = (
    "\n[Adapt now if this affects your current phase; otherwise ignore it and "
    "continue what you were doing.]"
)


def make_steering_middleware(
    watch: ScratchpadWatch,
    injector: Any,
    label: str,
) -> Callable[[FunctionInvocationContext, Callable[[], Awaitable[None]]], Awaitable[None]]:
    """Build function middleware that pushes scratchpad edits to a live agent.

    Detection rides on the tool calls the agent is already making: an LLM turn
    cannot be interrupted, so the next tool call is the earliest boundary at
    which anything can be delivered. This is also why no file watcher is used -
    delivery waits for that boundary regardless, so a watcher thread would buy
    nothing over polling here.

    Args:
        watch: This agent's own watermark (never share one between agents).
        injector: The ``MessageInjectionMiddleware`` instance registered on the
            same agent; its ``enqueue_messages`` drains into the next model call.
        label: Executor id, for the operator-facing progress line.

    Returns:
        An async function middleware, ready for ``Agent(middleware=[...])``.

    Example:
        >>> mw = make_steering_middleware(watch, injector, "discovery")  # doctest: +SKIP
    """
    # @function_middleware is REQUIRED, not decorative: this module uses
    # `from __future__ import annotations`, so the context annotation below is
    # the *string* "FunctionInvocationContext". The framework's callable
    # classifier reads `annotation.__name__`, which a str lacks - without the
    # decorator's marker it raises "Cannot determine middleware type". Same
    # class of trap as @response_handler in review.py.
    @function_middleware
    async def steering_middleware(
        context: FunctionInvocationContext,
        call_next: Callable[[], Awaitable[None]],
    ) -> None:
        await call_next()
        new = watch.poll()
        # Falsy covers both "no change" (None) and "cleared" (""): withdrawing
        # guidance is not new guidance to act on. Never raise - middleware must
        # not break the tool call it wraps.
        if new and context.session is not None:
            injector.enqueue_messages(
                context.session, [_NOTICE_HEAD + new + _NOTICE_TAIL]
            )
            print(f"  [steering] scratchpad update queued for {label}")

    return steering_middleware
