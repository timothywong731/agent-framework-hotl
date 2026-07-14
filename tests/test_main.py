import pytest

from hotl_demo.main import _prompt_human, model_present, normalize_host
from hotl_demo.review import LedgerQuestionRequest

TAGS = {"models": [{"name": "gemma4:31b"}, {"name": "qwen3.6:latest"}]}


def test_model_present_exact_tag():
    assert model_present(TAGS, "gemma4:31b") is True


def test_model_present_bare_name_requires_latest_tag():
    # Ollama resolves bare names to ':latest' only.
    assert model_present(TAGS, "qwen3.6") is True      # qwen3.6:latest is pulled
    assert model_present(TAGS, "gemma4") is False      # only gemma4:31b is pulled


def test_model_absent():
    assert model_present(TAGS, "gemma4:9b") is False
    assert model_present({}, "gemma4:31b") is False


def test_normalize_host_adds_scheme_and_strips_slash():
    assert normalize_host("127.0.0.1:11434") == "http://127.0.0.1:11434"
    assert normalize_host("http://localhost:11434/") == "http://localhost:11434"
    assert normalize_host("https://ollama.example.com") == "https://ollama.example.com"


def test_prompt_human_declines_on_eof(monkeypatch, capsys):
    def raise_eof(_):
        raise EOFError

    monkeypatch.setattr("builtins.input", raise_eof)
    q = LedgerQuestionRequest("q-1", "discovery", None, "Scope?", "ctx", "in scope")
    assert _prompt_human(q) == ""                      # decline, not a crash
    assert "declining" in capsys.readouterr().out
