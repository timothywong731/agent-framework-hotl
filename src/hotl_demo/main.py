"""CLI runner: preflight, run the workflow, prompt the human at the review gate.

Entry point for ``poetry run demo``. The run loop mirrors the framework's
request-info pattern: stream events, collect ``request_info`` payloads while
the workflow idles at the review gate, ask the human in the terminal, resume
with ``run(responses={...})``, repeat until an iteration ends with no pending
requests.
"""
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
    """Check whether Ollama's tag list contains the requested model.

    Args:
        tags: Parsed ``GET /api/tags`` response
            (``{"models": [{"name": ...}, ...]}``).
        model: Requested tag, e.g. ``"gemma4:31b"`` or bare ``"gemma4"``.

    Returns:
        True when the exact tag is present.

    Example:
        >>> model_present({"models": [{"name": "gemma4:31b"}]}, "gemma4:31b")
        True
        >>> model_present({"models": [{"name": "gemma4:31b"}]}, "gemma4")
        False
    """
    # Ollama resolves a bare name strictly to '<name>:latest' - mirror that,
    # or preflight passes models the chat client will 404 on.
    names = {m.get("name", "") for m in tags.get("models", [])}
    wanted = model if ":" in model else f"{model}:latest"
    return wanted in names


def normalize_host(host: str) -> str:
    """Give a scheme-less ``OLLAMA_HOST`` value an explicit ``http://`` scheme.

    Args:
        host: Host value, with or without scheme, with or without trailing
            slash.

    Returns:
        A urllib-usable base URL.

    Example:
        >>> normalize_host("127.0.0.1:11434")
        'http://127.0.0.1:11434'
        >>> normalize_host("https://ollama.local/")
        'https://ollama.local'
    """
    # Ollama docs use scheme-less OLLAMA_HOST (e.g. 127.0.0.1:11434) and the
    # ollama client accepts it; urllib needs an explicit scheme.
    host = host.rstrip("/")
    return host if "://" in host else f"http://{host}"


def preflight(base_url: str, model: str) -> None:
    """Fail fast, with an actionable message, before any LLM work starts.

    Args:
        base_url: Ollama base URL (already normalized).
        model: Model tag the run will use.

    Raises:
        SystemExit: Server unreachable, or model not pulled.
    """
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
    """Present one ledger question in the terminal and read the verdict.

    Args:
        q: The request payload emitted by the review gate.

    Returns:
        The raw input line - non-empty text is an authoritative answer,
        empty/whitespace (or a closed stdin) means decline.
    """
    where = f"{q.phase}[{q.unit}]" if q.unit else q.phase
    print(f"\n[{q.question_id}] ({where}) {q.question}")
    print(f"      Evidence: {q.context}")
    print(f"      Default if declined: {q.default_assumption}")
    try:
        return input("      Your answer (ENTER to decline): ")
    except EOFError:  # non-interactive stdin: decline, keep the run alive
        print("(stdin closed - declining)")
        return ""


async def _amain() -> None:
    """Parse args, preflight, then drive the workflow's run/pause/resume loop."""
    parser = argparse.ArgumentParser(description="HOTL cloud migration readiness demo")
    parser.add_argument("--model", default=os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL),
                        help="Ollama model tag (default: %(default)s)")
    parser.add_argument("--data", type=Path, default=Path("sample_data"),
                        help="sample data directory")
    args = parser.parse_args()
    os.environ["OLLAMA_MODEL"] = args.model  # OllamaChatClient reads this
    base_url = normalize_host(os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    preflight(base_url, args.model)
    ensure_scratchpad(SCRATCHPAD_PATH)

    run_dir = Path("output") / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    store = ArtifactStore(run_dir, REPOS)
    # import here so --help and preflight failures never touch the framework
    from .pipeline import build_workflow
    workflow = build_workflow(store, args.data)

    responses: dict[str, str] | None = None
    while True:
        # First iteration kicks the workflow off; later iterations resume the
        # SAME workflow instance with the human's responses (state persists
        # across run() calls - that is the framework's pause/resume contract).
        stream = (workflow.run("start", stream=True) if responses is None
                  else workflow.run(stream=True, responses=responses))
        pending: dict[str, LedgerQuestionRequest] = {}
        async for event in stream:
            if event.type == "request_info" and isinstance(event.data, LedgerQuestionRequest):
                pending[event.request_id] = event.data
            elif event.type == "output":
                print(f"\nFinal report: {event.data}")
        if not pending:
            break  # the run finished without pausing: we are done
        responses = {rid: _prompt_human(q) for rid, q in pending.items()}
    print(f"Run artifacts: {run_dir}")


def run() -> None:
    """Synchronous entry point for the ``demo`` poetry script.

    Example:
        $ poetry run demo --model gemma4:31b
    """
    try:
        asyncio.run(_amain())
    except KeyboardInterrupt:
        print("\nAborted. Artifacts written so far persist under output/.")


if __name__ == "__main__":
    run()
