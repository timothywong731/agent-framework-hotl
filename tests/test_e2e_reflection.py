"""Live E2E smoke: one bounded reflection run against local Ollama.

    OLLAMA_E2E=1 poetry run pytest -m ollama -s tests/test_e2e_reflection.py

max_passes=1 is both the shortest run and the sharpest case: the judge is
never consulted, so the log must still be coherent with zero verdicts.
Asserts the artifacts, not the model's prose.
"""
import json
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.ollama


@pytest.mark.skipif(os.environ.get("OLLAMA_E2E") != "1", reason="set OLLAMA_E2E=1 to run")
async def test_reflection_smoke(tmp_path):
    os.environ.setdefault("OLLAMA_MODEL", "gemma4:31b")  # same idiom as test_e2e_reflexion.py
    from reflection_demo.judging import RunLog
    from reflection_demo.main import (
        DEFAULT_TOPIC,
        LOG_FILENAME,
        REPORT_FILENAME,
        build_agent,
        persist_fallback,
    )
    from reflection_demo.prompting import render_judge_instructions, render_worker_prompt

    corpus = Path("sample_data")
    report_path = tmp_path / REPORT_FILENAME
    log = RunLog(tmp_path / LOG_FILENAME)
    max_passes = 1

    agent, flag = build_agent(
        corpus, report_path, render_judge_instructions(topic=DEFAULT_TOPIC),
        log, max_passes)
    # Session attached exactly as main._amain does it - the loop replaces
    # context.messages between passes, so the session is what makes pass N see
    # pass N-1. Immaterial at max_passes=1, but this smoke test is only worth
    # anything if it runs the production path.
    result = await agent.run(
        render_worker_prompt(topic=DEFAULT_TOPIC, max_passes=max_passes),
        session=agent.create_session())
    persist_fallback(result, report_path, flag)
    outcome, passes = log.finish(max_passes, report_path)

    assert report_path.exists() and report_path.read_text(encoding="utf-8").strip()
    # max_passes=1 means AgentLoopMiddleware's iteration cap fires before the
    # judge is ever consulted (see judging.summarize's docstring) - so this
    # single pass produces zero verdicts, and finish() must still land on a
    # coherent ("unjudged", 1) rather than an empty/broken outcome.
    assert (outcome, passes) == ("unjudged", 1)

    lines = [json.loads(ln) for ln in
             (tmp_path / LOG_FILENAME).read_text(encoding="utf-8").splitlines()]
    assert lines[0] == {"pass": 1, "answered": None, "reasoning": None, "judged": False}
    assert lines[-1]["outcome"] == "unjudged"
