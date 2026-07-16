"""Compaction wiring tests: budget math, strategy composition, logging.

All LLM-free: strategies are driven directly on synthetic message lists,
the summarizer is a recorded fake (SummarizationStrategy only needs
``await client.get_response(...)`` returning an object with ``.text``).
"""
import pytest

from hotl_demo.compaction import DEFAULT_NUM_CTX, resolve_num_ctx, token_budget


def test_resolve_num_ctx_default(monkeypatch):
    monkeypatch.delenv("OLLAMA_NUM_CTX", raising=False)
    assert resolve_num_ctx() == DEFAULT_NUM_CTX == 4096


def test_resolve_num_ctx_env_override(monkeypatch):
    monkeypatch.setenv("OLLAMA_NUM_CTX", "16384")
    assert resolve_num_ctx() == 16384


def test_resolve_num_ctx_garbage_fails_fast(monkeypatch):
    monkeypatch.setenv("OLLAMA_NUM_CTX", "lots")
    with pytest.raises(ValueError):
        resolve_num_ctx()


def test_token_budget_reserves_output_and_margin():
    # 0.8 * (4096 - 1024) = 2457.6 -> 2457
    assert token_budget(4096) == 2457
