"""CLI runner for the reflection demo: preflight, wiring, run.

The preflight helpers (``normalize_host``, ``model_present``, ``preflight``)
mirror ``reflexion_demo/main.py`` - duplicated deliberately, because this
package is standalone by design and must not import from a sibling demo.

There is no workflow graph here. ``AgentLoopMiddleware`` drives every pass
inside a single ``agent.run()`` call; that is the whole point of the
contrast with the reflexion demo.
"""
import argparse
import asyncio
import json
import os
import urllib.request
from datetime import datetime
from pathlib import Path

from .judging import RunLog, make_judge_predicate, make_next_message
from .prompting import render_judge_instructions, render_worker_prompt
from .tools import atomic_write, make_corpus_tools, make_report_tools

DEFAULT_MODEL = "gemma4:31b"
# Character-identical to reflexion_demo.main.DEFAULT_TOPIC - the A/B needs
# both demos assessing exactly the same question. Guarded by a test.
DEFAULT_TOPIC = ("Assess migrating OMS file storage from the NFS file store "
                 "to Amazon S3.")
REPORT_FILENAME = "report.md"
LOG_FILENAME = "reflection_log.jsonl"

_WORKER_INSTRUCTIONS = (
    "You are a migration analyst. You produce evidence-grounded reports and "
    "deliver them with the write_report tool."
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
    evidence-free, hallucinated report - and the judge, having no corpus,
    would never notice.
    """
    if not Path(corpus_root).is_dir():
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


def persist_fallback(result, report_path: Path, flag) -> None:
    """Last resort when the model never called ``write_report``.

    The turn's LONGEST assistant reply is the report, not the latest: the
    model often emits the full report as plain chat text and then answers a
    follow-up nudge with filler ("Done."), which must not clobber it. Same
    reasoning as ``reflexion_demo/graph.py:_draft``.
    """
    if flag.written:
        return
    texts = [(m.text or "").strip() for m in result.messages if m.role == "assistant"]
    atomic_write(report_path, max(texts, key=len, default="") or "(no report produced)")
    print("  [worker] write_report never called - persisted the longest reply instead")


def build_agent(corpus_root: Path, report_path: Path, judge_instructions: str,
                log: RunLog, max_passes: int):
    """Build the looping agent and its report-write flag.

    The tool set is byte-for-byte the reflexion worker's; the judge is a bare
    ``OllamaChatClient`` with none of it. That single difference is the demo.

    ``AgentLoopMiddleware`` is ``@experimental`` and warns on construction.
    The warning is deliberately NOT filtered - a demo that hides the
    framework's own stability signal from its reader is lying by omission.

    ``fresh_context`` stays at its default ``False``: each pass accumulates
    the prior conversation, which is precisely "grounded on its own chat
    history" - the property that separates reflection from reflexion's fresh
    session per cycle.

    Returns:
        ``(agent, flag)``.
    """
    from agent_framework import Agent, AgentLoopMiddleware
    from agent_framework.ollama import OllamaChatClient

    write_report, flag = make_report_tools(report_path)
    loop = AgentLoopMiddleware(
        make_judge_predicate(OllamaChatClient(), judge_instructions, log),
        max_iterations=max_passes,
        next_message=make_next_message(),
    )
    agent = Agent(
        client=OllamaChatClient(),
        name="worker",
        instructions=_WORKER_INSTRUCTIONS,
        tools=make_corpus_tools(corpus_root) + [write_report],
        middleware=[loop],
        default_options={"num_ctx": resolve_num_ctx()},
    )
    return agent, flag


async def _amain() -> None:
    parser = argparse.ArgumentParser(
        description="Reflection demo: one agent drafts, a tool-less judge decides when to stop")
    parser.add_argument("--topic", default=DEFAULT_TOPIC,
                        help="migration topic to report on (default: %(default)s)")
    parser.add_argument("--max-passes", type=int, default=3, metavar="N",
                        help="pass budget; the judge is NOT consulted on the capped "
                             "pass, so the report ships unjudged (default: %(default)s)")
    parser.add_argument("--model", default=os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL),
                        help="Ollama model tag (default: %(default)s)")
    parser.add_argument("--num-ctx", type=int,
                        default=int(os.environ.get("OLLAMA_NUM_CTX", 4096)),
                        help="Ollama context window in tokens (default: %(default)s)")
    args = parser.parse_args()
    if args.max_passes < 1:
        parser.error("--max-passes must be >= 1")
    os.environ["OLLAMA_MODEL"] = args.model
    os.environ["OLLAMA_NUM_CTX"] = str(args.num_ctx)
    corpus_root = Path("sample_data")
    ensure_corpus(corpus_root)
    preflight(normalize_host(os.environ.get("OLLAMA_HOST", "http://localhost:11434")),
              args.model)

    run_dir = Path("output") / datetime.now().strftime("reflection_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    report_path = run_dir / REPORT_FILENAME
    log = RunLog(run_dir / LOG_FILENAME)

    agent, flag = build_agent(
        corpus_root, report_path,
        render_judge_instructions(topic=args.topic), log, args.max_passes)

    print(f"Topic: {args.topic}")
    print(f"Budget: {args.max_passes} passes (the judge holds NO tools)")
    result = await agent.run(render_worker_prompt(topic=args.topic, max_passes=args.max_passes))
    persist_fallback(result, report_path, flag)
    outcome, passes = log.finish(args.max_passes, report_path)

    if outcome == "answered":
        print(f"\nReport answered the judge after {passes} pass(es): {report_path}")
    else:
        print(f"\nPass budget exhausted after {passes} pass(es); the judge was "
              f"never consulted on the last one - report ships unjudged: {report_path}")
    print(f"Run artifacts: {run_dir} ({REPORT_FILENAME}, {LOG_FILENAME})")


def run() -> None:
    """Synchronous entry point for the ``reflection`` poetry script."""
    try:
        asyncio.run(_amain())
    except KeyboardInterrupt:
        print("\nAborted. Artifacts written so far persist under output/.")


if __name__ == "__main__":
    run()
