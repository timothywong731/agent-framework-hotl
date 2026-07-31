"""CLI runner for the reflexion demo: preflight, factories, run loop.

The preflight helpers (``normalize_host``, ``model_present``, ``preflight``)
mirror ``hotl_demo/main.py`` - duplicated deliberately, because this package
is standalone by design and must not import from hotl_demo.
"""
import argparse
import asyncio
import json
import os
import urllib.request
from datetime import datetime
from pathlib import Path

from .budget import BUDGETED_TOOL_NAMES, ToolBudget, make_budget_middleware
from .graph import (
    LOG_FILENAME,
    REPORT_FILENAME,
    ReviewerExecutor,
    WorkerExecutor,
    build_reflexion_workflow,
)
from .tools import make_corpus_tools, make_report_tools

DEFAULT_MODEL = "gemma4:31b"
DEFAULT_TOPIC = ("Assess migrating OMS file storage from the NFS file store "
                 "to Amazon S3.")

_WORKER_INSTRUCTIONS = (
    "You are a migration analyst. You produce evidence-grounded reports and "
    "deliver them with the write_report tool."
)
_REVIEWER_INSTRUCTIONS = (
    "You are an independent reviewer. You verify reports against their "
    "sources yourself; you never take the author's word for anything."
)


def resolve_num_ctx() -> int:
    """Read the Ollama context window from ``OLLAMA_NUM_CTX`` (default 4096)."""
    return int(os.environ.get("OLLAMA_NUM_CTX", 4096))


def normalize_host(host: str) -> str:
    """Give a scheme-less ``OLLAMA_HOST`` an explicit http:// scheme."""
    host = host.rstrip("/")
    return host if "://" in host else f"http://{host}"


def model_present(tags: dict, model: str) -> bool:
    """True when Ollama's tag list contains the requested model
    (bare names resolve to ``<name>:latest``, as Ollama does)."""
    names = {m.get("name", "") for m in tags.get("models", [])}
    wanted = model if ":" in model else f"{model}:latest"
    return wanted in names


def ensure_corpus(corpus_root: Path) -> None:
    """Fail fast when the corpus directory is missing.

    ``rglob`` on a missing directory silently yields nothing, so without this
    guard a run from the wrong working directory would "succeed" with an
    evidence-free, hallucinated report.
    """
    if not corpus_root.is_dir():
        raise SystemExit(
            f"Corpus directory '{corpus_root}' not found. Run the demo from "
            "the repository root (it reads sample_data/).")


def preflight(base_url: str, model: str) -> None:
    """Fail fast with an actionable message before any LLM work starts."""
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=5) as resp:
            tags = json.load(resp)
    except OSError as exc:
        raise SystemExit(
            f"Ollama not reachable at {base_url} - is it running? ('ollama serve')\n{exc}")
    if not model_present(tags, model):
        raise SystemExit(f"Model '{model}' not found in Ollama. Run: ollama pull {model}")
    print(f"Preflight: Ollama OK, {model} present.")


def _make_agent(name: str, instructions: str, tools: list, middleware: list):
    """One Ollama-backed agent; model comes from OLLAMA_MODEL, window from
    OLLAMA_NUM_CTX (num_ctx must be pinned or the server silently truncates)."""
    from agent_framework import Agent
    from agent_framework.ollama import OllamaChatClient

    return Agent(
        client=OllamaChatClient(),
        name=name,
        instructions=instructions,
        tools=tools,
        middleware=middleware,
        default_options={"num_ctx": resolve_num_ctx()},
    )


def make_worker_factory(corpus_root: Path, report_path: Path, max_tool_calls: int):
    """Factory-of-factories: the executor calls the result once per turn.

    ``finalize=True`` builds the agent with write_report ONLY - the
    review-cycle budget's tool strip, expressed at construction time.
    """
    def factory(finalize: bool = False):
        write_report, _read_report, flag = make_report_tools(report_path)
        budget = ToolBudget(max_calls=max_tool_calls)
        if finalize:
            return _make_agent("worker", _WORKER_INSTRUCTIONS, [write_report], []), budget, flag
        middleware = [make_budget_middleware(budget, BUDGETED_TOOL_NAMES, "worker")]
        tools = make_corpus_tools(corpus_root) + [write_report]
        return _make_agent("worker", _WORKER_INSTRUCTIONS, tools, middleware), budget, flag

    return factory


def make_reviewer_factory(corpus_root: Path, report_path: Path, max_tool_calls: int):
    """Same corpus binding as the worker (information parity), plus read_report."""
    def factory():
        _write_report, read_report, _flag = make_report_tools(report_path)
        budget = ToolBudget(max_calls=max_tool_calls)
        middleware = [make_budget_middleware(budget, BUDGETED_TOOL_NAMES, "reviewer")]
        tools = make_corpus_tools(corpus_root) + [read_report]
        return _make_agent("reviewer", _REVIEWER_INSTRUCTIONS, tools, middleware), budget

    return factory


async def _amain() -> None:
    parser = argparse.ArgumentParser(
        description="Reflexion demo: worker drafts, reviewer verifies, budgets bound both")
    parser.add_argument("--topic", default=DEFAULT_TOPIC,
                        help="migration topic to report on (default: %(default)s)")
    parser.add_argument("--max-cycles", type=int, default=3, metavar="N",
                        help="review-cycle budget; on exhaustion the worker finalizes "
                             "tool-less and the report ships unapproved (default: %(default)s)")
    parser.add_argument("--max-tool-calls", type=int, default=12, metavar="N",
                        help="per-turn read-tool budget; on exhaustion read tools are "
                             "stripped mid-turn (default: %(default)s)")
    parser.add_argument("--model", default=os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL),
                        help="Ollama model tag (default: %(default)s)")
    parser.add_argument("--num-ctx", type=int,
                        default=int(os.environ.get("OLLAMA_NUM_CTX", 4096)),
                        help="Ollama context window in tokens (default: %(default)s)")
    args = parser.parse_args()
    if args.max_cycles < 1:
        parser.error("--max-cycles must be >= 1")
    if args.max_tool_calls < 1:
        parser.error("--max-tool-calls must be >= 1")
    os.environ["OLLAMA_MODEL"] = args.model
    os.environ["OLLAMA_NUM_CTX"] = str(args.num_ctx)
    corpus_root = Path("sample_data")
    ensure_corpus(corpus_root)
    preflight(normalize_host(os.environ.get("OLLAMA_HOST", "http://localhost:11434")),
              args.model)

    run_dir = Path("output") / datetime.now().strftime("reflexion_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    report_path = run_dir / REPORT_FILENAME

    worker = WorkerExecutor(
        make_worker_factory(corpus_root, report_path, args.max_tool_calls),
        run_dir, args.max_cycles, args.max_tool_calls)
    reviewer = ReviewerExecutor(
        make_reviewer_factory(corpus_root, report_path, args.max_tool_calls))
    workflow = build_reflexion_workflow(worker, reviewer)

    print(f"Topic: {args.topic}")
    print(f"Budgets: {args.max_cycles} review cycles, "
          f"{args.max_tool_calls} read-tool calls per turn")
    async for event in workflow.run(args.topic, stream=True):
        if event.type == "output":
            print(f"\n{event.data}")
    print(f"Run artifacts: {run_dir} ({REPORT_FILENAME}, {LOG_FILENAME})")


def run() -> None:
    """Synchronous entry point for the ``reflexion`` poetry script."""
    try:
        asyncio.run(_amain())
    except KeyboardInterrupt:
        print("\nAborted. Artifacts written so far persist under output/.")


if __name__ == "__main__":
    run()
