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
from typing import Any

from agent_framework import (
    CharacterEstimatorTokenizer,
    SelectiveToolCallCompactionStrategy,
    SummarizationStrategy,
    TokenBudgetComposedStrategy,
    annotate_message_groups,
    included_messages,
    included_token_count,
)

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


class _LoggedStrategy:
    """CompactionStrategy-protocol wrapper: delegate; one line when it acted.

    Console-only by design (see spec 3.5) - no artifact files.
    """

    def __init__(self, inner: Any, label: str, tokenizer: Any) -> None:
        self._inner = inner
        self._label = label
        self._tokenizer = tokenizer

    async def __call__(self, messages: list) -> bool:
        # Annotate up front so the "before" numbers are real - the composed
        # strategy would otherwise annotate after we counted.
        annotate_message_groups(messages, tokenizer=self._tokenizer)
        msgs_before = len(included_messages(messages))
        toks_before = included_token_count(messages)
        changed = await self._inner(messages)
        if changed:
            print(f"  {self._label}: compacted context "
                  f"{msgs_before} -> {len(included_messages(messages))} messages "
                  f"(~{toks_before} -> {included_token_count(messages)} tokens)")
        return changed


def build_compaction_strategy(label: str, num_ctx: int, summarizer: Any = None):
    """Build the hybrid pipeline for one agent.

    Args:
        label: Executor id used in the console line.
        num_ctx: Model window in tokens; drives the budget.
        summarizer: Test seam - anything with ``async get_response(messages,
            stream=False)`` returning ``.text``. Defaults to a dedicated
            ``OllamaChatClient`` carrying ``num_ctx`` so the summarization
            call is not itself truncated.

    Returns:
        An async ``(messages) -> bool`` satisfying the framework's
        ``CompactionStrategy`` protocol.
    """
    if summarizer is None:
        from agent_framework.ollama import OllamaChatClient
        summarizer = OllamaChatClient(default_options={"num_ctx": num_ctx})
    tokenizer = CharacterEstimatorTokenizer()
    composed = TokenBudgetComposedStrategy(
        token_budget=token_budget(num_ctx),
        tokenizer=tokenizer,
        strategies=[
            SelectiveToolCallCompactionStrategy(
                keep_last_tool_call_groups=_KEEP_TOOL_GROUPS),
            # target_count=2/threshold=0: the default trigger (>~6 non-system
            # MESSAGES) is count-based and never fires for a history of a few
            # huge read_file results - exactly this repo's growth profile.
            SummarizationStrategy(client=summarizer, target_count=2, threshold=0),
        ],
        early_stop=True,
    )
    return _LoggedStrategy(composed, label, tokenizer)
