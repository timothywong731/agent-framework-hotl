"""Bounded context: the hybrid compaction pipeline for phase agents.

Nothing in agent-framework bounds context by default; without this module the
whole session history goes to Ollama every call and the server silently
truncates the OLDEST tokens (system prompt first) at ``num_ctx``. See
``docs/superpowers/specs/2026-07-16-context-compaction-design.md``.

Pipeline (composed, early-stop, checked on EVERY model call, including each
tool-loop iteration): selective eviction of old tool-call groups -> LLM
summarization of what remains -> the framework's deterministic oldest-first
fallback. The token budget is therefore a hard guarantee.
"""
from __future__ import annotations

import os

DEFAULT_NUM_CTX = 4096
_OUTPUT_RESERVE = 1024   # tokens left for the model's own output
_BUDGET_FRACTION = 0.8   # margin: the 4-chars/token estimator is a heuristic
_KEEP_TOOL_GROUPS = 2    # newest tool-call groups kept verbatim


def resolve_num_ctx() -> int:
    """Read the model context window from ``OLLAMA_NUM_CTX``.

    Returns:
        Window size in tokens; 4096 (Ollama's server default) when unset.

    Raises:
        ValueError: Non-integer value - fail fast at startup, never mid-run.

    Example:
        >>> import os; os.environ.pop("OLLAMA_NUM_CTX", None) and None
        >>> resolve_num_ctx()
        4096
    """
    return int(os.environ.get("OLLAMA_NUM_CTX", DEFAULT_NUM_CTX))


def token_budget(num_ctx: int) -> int:
    """Included-token budget for a window: reserve output, keep a margin.

    Example:
        >>> token_budget(4096)
        2457
    """
    return int(_BUDGET_FRACTION * (num_ctx - _OUTPUT_RESERVE))
