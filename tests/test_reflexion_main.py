"""CLI pure parts: preflight helpers and the agent factories' tool wiring."""
import pytest

from reflexion_demo.main import (
    DEFAULT_TOPIC,
    make_reviewer_factory,
    make_worker_factory,
    model_present,
    normalize_host,
    resolve_num_ctx,
)


def test_normalize_host_adds_scheme():
    assert normalize_host("127.0.0.1:11434") == "http://127.0.0.1:11434"
    assert normalize_host("https://ollama.local/") == "https://ollama.local"


def test_model_present_mirrors_ollama_latest_resolution():
    tags = {"models": [{"name": "gemma4:31b"}, {"name": "phi4:latest"}]}
    assert model_present(tags, "gemma4:31b")
    assert model_present(tags, "phi4")
    assert not model_present(tags, "gemma4")


def test_resolve_num_ctx_env_default(monkeypatch):
    monkeypatch.delenv("OLLAMA_NUM_CTX", raising=False)
    assert resolve_num_ctx() == 4096
    monkeypatch.setenv("OLLAMA_NUM_CTX", "32768")
    assert resolve_num_ctx() == 32768


def test_default_topic_is_the_s3_assessment():
    assert "NFS file store" in DEFAULT_TOPIC and "S3" in DEFAULT_TOPIC


def _tool_names(agent):
    return sorted(t.name for t in agent.default_options["tools"])


def test_worker_factory_tool_sets(tmp_path, monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "test-model")
    factory = make_worker_factory(tmp_path, tmp_path / "report.md", max_tool_calls=5)

    agent, budget, flag = factory(False)
    assert _tool_names(agent) == ["list_files", "read_file", "write_report"]
    assert budget.max_calls == 5 and budget.spent == 0 and flag.written is False

    final_agent, final_budget, _ = factory(True)
    assert _tool_names(final_agent) == ["write_report"]     # stripped at construction
    assert final_budget.spent == 0


def test_reviewer_factory_tool_set(tmp_path, monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "test-model")
    factory = make_reviewer_factory(tmp_path, tmp_path / "report.md", max_tool_calls=5)
    agent, budget = factory()
    assert _tool_names(agent) == ["list_files", "read_file", "read_report"]
    assert budget.max_calls == 5
