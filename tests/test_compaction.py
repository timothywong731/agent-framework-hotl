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


# -- the composed pipeline, driven directly on synthetic histories --------

import asyncio

from agent_framework import Content, Message, included_messages, included_token_count

from hotl_demo.compaction import build_compaction_strategy


class FakeSummaryClient:
    """Duck-typed summarizer: SummarizationStrategy only calls
    ``await client.get_response(messages, stream=False)`` and reads ``.text``."""

    def __init__(self, text="SUMMARY OF OLDER CONTEXT", error=None):
        self.calls = []
        self._text = text
        self._error = error

    async def get_response(self, messages, stream=False):
        self.calls.append(messages)
        if self._error is not None:
            raise self._error

        class _Resp:
            text = self._text
        return _Resp()


def _tool_group(i, payload):
    """One atomic tool-call group: assistant function_call + tool result."""
    return [
        Message(role="assistant", contents=[Content(
            type="function_call", call_id=f"c{i}", name="read_file",
            arguments='{"path": "f"}')]),
        Message(role="tool", contents=[Content(
            type="function_result", call_id=f"c{i}", result=payload)]),
    ]


def _history(groups, payload_chars):
    msgs = [Message(role="system", contents=["phase instructions"]),
            Message(role="user", contents=["initial prompt"])]
    for i in range(groups):
        msgs.extend(_tool_group(i, "x" * payload_chars))
    return msgs


def test_over_budget_evicts_old_tool_groups_keeps_newest_two():
    fake = FakeSummaryClient()
    strategy = build_compaction_strategy("t", 4096, summarizer=fake)
    # 8 groups x ~1000 tokens >> 2457 budget; eviction alone suffices.
    messages = _history(groups=8, payload_chars=4000)
    changed = asyncio.run(strategy(messages))
    assert changed
    assert included_token_count(messages) <= 2457
    kept = included_messages(messages)
    kept_call_ids = {c.call_id for m in kept for c in m.contents
                     if c.type == "function_call"}
    assert kept_call_ids == {"c6", "c7"}          # newest 2 groups verbatim
    assert fake.calls == []                        # stage 2 never needed
    assert kept[0].text == "phase instructions"    # system group preserved


def test_summarizer_fires_only_when_eviction_insufficient():
    fake = FakeSummaryClient()
    strategy = build_compaction_strategy("t", 4096, summarizer=fake)
    # 2 giant groups: eviction keeps both (keep_last=2) and stays over budget.
    messages = _history(groups=2, payload_chars=12000)
    asyncio.run(strategy(messages))
    assert len(fake.calls) >= 1                    # stage 2 engaged
    assert included_token_count(messages) <= 2457  # budget still enforced


def test_summarizer_failure_never_raises_budget_still_enforced():
    fake = FakeSummaryClient(error=RuntimeError("ollama down"))
    strategy = build_compaction_strategy("t", 4096, summarizer=fake)
    messages = _history(groups=2, payload_chars=12000)
    changed = asyncio.run(strategy(messages))      # must not raise
    assert changed
    assert included_token_count(messages) <= 2457  # fallback did the work


def test_under_budget_is_untouched_and_silent(capsys):
    fake = FakeSummaryClient()
    strategy = build_compaction_strategy("t", 4096, summarizer=fake)
    messages = _history(groups=1, payload_chars=200)
    changed = asyncio.run(strategy(messages))
    assert not changed
    assert len(included_messages(messages)) == len(messages)
    assert capsys.readouterr().out == ""


def test_log_line_on_change(capsys):
    strategy = build_compaction_strategy(
        "deep_analysis:oms-monolith", 4096, summarizer=FakeSummaryClient())
    messages = _history(groups=8, payload_chars=4000)
    asyncio.run(strategy(messages))
    out = capsys.readouterr().out
    assert out.count("\n") == 1
    assert out.startswith("  deep_analysis:oms-monolith: compacted context ")
    assert "messages" in out and "tokens" in out


# -- spike-as-test: compaction fires per model call in the tool loop ------

from agent_framework import Agent, BaseChatClient, ChatResponse, FunctionInvocationLayer


class ScriptedToolClient(FunctionInvocationLayer, BaseChatClient):
    """Two-beat script: call the 'ping' tool, then produce final text.

    Composes the same layers OllamaChatClient does (minus middleware and
    telemetry), so the tool loop and the compaction hook are the real ones.
    """

    def __init__(self):
        super().__init__()
        self.model_calls = 0

    async def _inner_get_response(self, *, messages, stream, options, **kwargs):
        self.model_calls += 1
        if self.model_calls == 1:
            return ChatResponse(messages=[Message(role="assistant", contents=[
                Content(type="function_call", call_id="c1", name="ping",
                        arguments="{}")])])
        return ChatResponse(messages=[Message(role="assistant", contents=["done"])])


class CountingStrategy:
    """No-op CompactionStrategy that counts invocations."""

    def __init__(self):
        self.calls = 0

    async def __call__(self, messages):
        self.calls += 1
        return False


def test_compaction_runs_once_per_model_call_in_tool_loop():
    def ping() -> str:
        """Reply with pong."""
        return "pong"

    client = ScriptedToolClient()
    strategy = CountingStrategy()
    agent = Agent(client=client, name="spike", instructions="use tools",
                  tools=[ping], compaction_strategy=strategy)
    result = asyncio.run(agent.run("go"))
    assert result.text == "done"
    assert client.model_calls == 2
    # THE design-carrying assertion: one compaction pass per model call,
    # i.e. compaction also fires between tool iterations.
    assert strategy.calls == 2


# -- wiring: the real PhaseExecutor agent carries the strategy + window ---

from pathlib import Path

from hotl_demo.artifacts import REPOS, ArtifactStore
from hotl_demo.phases import PhaseExecutor, build_phase_specs


def test_phase_executor_wires_compaction_and_num_ctx(monkeypatch, tmp_path):
    monkeypatch.setenv("OLLAMA_MODEL", "test-model")   # ctor reads env; no server contact
    monkeypatch.setenv("OLLAMA_NUM_CTX", "8192")
    spec = build_phase_specs(Path("sample_data"))[0]
    store = ArtifactStore(tmp_path / "run", REPOS)
    ex = PhaseExecutor(spec, store, scratchpad_path=tmp_path / "scratchpad.md")
    assert ex._agent.compaction_strategy is not None
    assert ex._agent.default_options.get("num_ctx") == 8192
