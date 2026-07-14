"""CLI runner: preflight, run the workflow, prompt the human at the review gate."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import urllib.request
from datetime import datetime
from pathlib import Path

from .artifacts import REPOS, ArtifactStore
from .review import LedgerQuestionRequest
from .tools import SCRATCHPAD_PATH, ensure_scratchpad

DEFAULT_MODEL = "gemma4:31b"


def model_present(tags: dict, model: str) -> bool:
    names = {m.get("name", "") for m in tags.get("models", [])}
    return model in names or any(n.split(":", 1)[0] == model for n in names)


def preflight(base_url: str, model: str) -> None:
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=5) as resp:
            tags = json.load(resp)
    except OSError as exc:
        raise SystemExit(
            f"Ollama not reachable at {base_url} - is it running? ('ollama serve')\n{exc}"
        )
    if not model_present(tags, model):
        raise SystemExit(f"Model '{model}' not found in Ollama. Run: ollama pull {model}")
    print(f"Preflight: Ollama OK, {model} present.")


def _prompt_human(q: LedgerQuestionRequest) -> str:
    where = f"{q.phase}[{q.unit}]" if q.unit else q.phase
    print(f"\n[{q.question_id}] ({where}) {q.question}")
    print(f"      Evidence: {q.context}")
    print(f"      Default if declined: {q.default_assumption}")
    return input("      Your answer (ENTER to decline): ")


async def _amain() -> None:
    parser = argparse.ArgumentParser(description="HOTL cloud migration readiness demo")
    parser.add_argument("--model", default=os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL),
                        help="Ollama model tag (default: %(default)s)")
    parser.add_argument("--data", type=Path, default=Path("sample_data"),
                        help="sample data directory")
    args = parser.parse_args()
    os.environ["OLLAMA_MODEL"] = args.model  # OllamaChatClient reads this
    base_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    preflight(base_url, args.model)
    ensure_scratchpad(SCRATCHPAD_PATH)

    run_dir = Path("output") / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    store = ArtifactStore(run_dir, REPOS)
    # import here so --help and preflight failures never touch the framework
    from .pipeline import build_workflow
    workflow = build_workflow(store, args.data)

    responses: dict[str, str] | None = None
    while True:
        stream = (workflow.run("start", stream=True) if responses is None
                  else workflow.run(stream=True, responses=responses))
        pending: dict[str, LedgerQuestionRequest] = {}
        async for event in stream:
            if event.type == "request_info" and isinstance(event.data, LedgerQuestionRequest):
                pending[event.request_id] = event.data
            elif event.type == "output":
                print(f"\nFinal report: {event.data}")
        if not pending:
            break
        responses = {rid: _prompt_human(q) for rid, q in pending.items()}
    print(f"Run artifacts: {run_dir}")


def run() -> None:
    try:
        asyncio.run(_amain())
    except KeyboardInterrupt:
        print("\nAborted. Artifacts written so far persist under output/.")


if __name__ == "__main__":
    run()
