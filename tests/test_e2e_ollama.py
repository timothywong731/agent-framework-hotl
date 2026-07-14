"""Live E2E: full pipeline against local Ollama. Opt-in:

    OLLAMA_E2E=1 poetry run pytest -m ollama -s

Takes many minutes on gemma4:31b (7+ LLM calls).
"""
import asyncio
import os
from pathlib import Path

import pytest

from hotl_demo.artifacts import REPOS, ArtifactStore
from hotl_demo.pipeline import build_workflow
from hotl_demo.review import LedgerQuestionRequest

pytestmark = pytest.mark.ollama

ANSWER = ("Yes - reconciliation is in scope. Also: Azure is authoritative; "
          "treat the AWS S3 code as legacy to be replaced.")


@pytest.mark.skipif(os.environ.get("OLLAMA_E2E") != "1", reason="set OLLAMA_E2E=1 to run")
def test_full_pipeline_with_scripted_review(tmp_path):
    os.environ.setdefault("OLLAMA_MODEL", "gemma4:31b")
    store = ArtifactStore(tmp_path / "run", repos=REPOS)
    pad = tmp_path / "scratchpad.md"
    pad.write_text("Prioritise security findings and be concise.", encoding="utf-8")
    workflow = build_workflow(store, Path("sample_data"), scratchpad_path=pad)

    async def drive() -> list:
        responses: dict[str, str] | None = None
        answered_once = False
        outputs: list = []
        for _ in range(10):  # safety bound on resume cycles
            stream = (workflow.run("start", stream=True) if responses is None
                      else workflow.run(stream=True, responses=responses))
            pending: dict[str, LedgerQuestionRequest] = {}
            async for event in stream:
                if event.type == "request_info" and isinstance(event.data, LedgerQuestionRequest):
                    pending[event.request_id] = event.data
                elif event.type == "output":
                    outputs.append(event.data)
            if not pending:
                return outputs
            responses = {}
            for rid in pending:
                responses[rid] = "" if answered_once else ANSWER
                answered_once = True
        raise AssertionError("workflow did not complete within resume bound")

    outputs = asyncio.run(drive())

    assert outputs, "workflow yielded no final output"
    final = store.read_report("final_report.md")
    assert final and "## Adjudication log" in final
    assert store.review_completed() is True
    ledger = store.read_ledger()
    assert ledger, "pipeline raised no questions - plantings failed"
    assert any(e["status"] == "answered" for e in ledger)
    assert any(e["status"] == "declined" for e in ledger)
    assert len(store.read_all_reports()) >= 4  # discovery, 2x deep_analysis, ec, questionnaire
