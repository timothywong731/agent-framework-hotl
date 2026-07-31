"""Live E2E smoke: one bounded reflexion run against local Ollama.

    OLLAMA_E2E=1 poetry run pytest -m ollama -s tests/test_e2e_reflexion.py

Tiny budgets keep it short: 1 review cycle means at most draft -> review ->
forced finalize. Asserts the artifacts, not the model's prose.
"""
import json
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.ollama


@pytest.mark.skipif(os.environ.get("OLLAMA_E2E") != "1", reason="set OLLAMA_E2E=1 to run")
async def test_reflexion_smoke(tmp_path):
    os.environ.setdefault("OLLAMA_MODEL", "gemma4:31b")  # same idiom as test_e2e_ollama.py
    from reflexion_demo.graph import (
        REPORT_FILENAME,
        ReviewerExecutor,
        WorkerExecutor,
        build_reflexion_workflow,
    )
    from reflexion_demo.main import DEFAULT_TOPIC, make_reviewer_factory, make_worker_factory

    corpus = Path("sample_data")
    report_path = tmp_path / REPORT_FILENAME
    workflow = build_reflexion_workflow(
        WorkerExecutor(make_worker_factory(corpus, report_path, max_tool_calls=6),
                       tmp_path, max_cycles=1, max_tool_calls=6),
        ReviewerExecutor(make_reviewer_factory(corpus, report_path, max_tool_calls=6)),
    )

    outputs = []
    async for event in workflow.run(DEFAULT_TOPIC, stream=True):
        if event.type == "output":
            outputs.append(event.data)

    assert len(outputs) == 1
    assert report_path.exists() and report_path.read_text(encoding="utf-8").strip()
    lines = [json.loads(ln) for ln in
             (tmp_path / "review_log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert lines[0]["cycle"] == 1
    assert lines[-1]["outcome"] in ("approved", "forced")
