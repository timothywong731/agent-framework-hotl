"""CLI runner helpers: preflight matching, host normalization, review.jsonl."""
import json

import pytest

from hotl_demo.main import (
    _prompt_human,
    already_resumed,
    map_answers,
    model_present,
    normalize_host,
    parse_review_answers,
    render_review_lines,
)
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
    q = LedgerQuestionRequest("q-1", "discovery", None, "Scope?", "ctx", "in scope", "medium", "impact")
    assert _prompt_human(q) == ""                      # decline, not a crash
    assert "declining" in capsys.readouterr().out


def _q(qid):
    return {"id": qid, "phase": "discovery", "unit": None, "question": "Q?",
            "context": "c", "default_assumption": "d", "status": "open",
            "human_answer": None, "asked_at": "t"}


def test_render_review_lines_seeds_id_and_answer_only():
    # The questions stay in ledger.jsonl (agent-curated, read-only); the
    # answer sheet carries ONLY the human's input, joined on id.
    lines = [json.loads(l) for l in render_review_lines([_q("q-1"), _q("q-2")]).splitlines()]
    assert lines == [{"id": "q-1", "answer": ""}, {"id": "q-2", "answer": ""}]


def test_parse_review_answers_round_trip_and_blank_lines():
    text = '{"id": "q-1", "answer": "yes, in scope"}\n\n{"id": "q-2", "answer": ""}\n'
    assert parse_review_answers(text) == {"q-1": "yes, in scope", "q-2": ""}


@pytest.mark.parametrize("bad, hint", [
    ('{"id": "q-1", "answer": "ok"}\nnot json\n', "line 2"),
    ('["q-1", "ok"]\n', "line 1"),                          # not an object
    ('{"answer": "ok"}\n', "line 1"),                       # missing id
    ('{"id": "q-1", "answer": null}\n', "line 1"),          # non-string answer
    ('{"id": "q-1", "answer": "a"}\n{"id": "q-1", "answer": "b"}\n', "line 2"),
])
def test_parse_review_answers_is_loud_on_malformed_input(bad, hint):
    # A parse error must NEVER degrade into "decline" - that would silently
    # discard the human's gathered answers and ship a defaults-only report.
    with pytest.raises(ValueError, match=hint):
        parse_review_answers(bad)


def test_map_answers_missing_id_declines_and_unknown_ids_surface():
    pending = {"r1": LedgerQuestionRequest("q-1", "discovery", None, "Q?", "c", "d", "medium", "impact"),
               "r2": LedgerQuestionRequest("q-2", "discovery", None, "Q?", "c", "d", "medium", "impact")}
    responses, unknown = map_answers(pending, {"q-1": "yes", "q-99": "ghost"})
    assert responses == {"r1": "yes", "r2": ""}   # q-2 unanswered -> decline
    assert unknown == ["q-99"]                    # warned by the caller, ignored


def test_already_resumed_derives_from_ledger_not_the_sheet():
    # A --pause run resolves NO questions before its resume (verdicts are
    # applied only by a resume), so any non-open entry proves one already ran.
    # Reviewers caught the old guard inferring this from the answer sheet: an
    # EMPTIED review.jsonl on a first resume is a legitimate decline-everything,
    # not a crashed resume - it must not be refused.
    assert already_resumed([]) is False
    assert already_resumed([_q("q-1"), _q("q-2")]) is False          # all open
    assert already_resumed([_q("q-1"), {**_q("q-2"), "status": "answered"}]) is True
    assert already_resumed([{**_q("q-1"), "status": "declined"}]) is True
