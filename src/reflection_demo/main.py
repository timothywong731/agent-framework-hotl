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

from .budget import BUDGETED_TOOL_NAMES, PassBudget, make_budget_middleware
from .judging import RunLog, make_judge_predicate, make_next_message
from .prompting import render_finalize_message, render_judge_instructions, render_worker_prompt
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
    # key=len, not the last element - the model tends to answer the loop's
    # follow-up nudge with filler ("Done.") after already writing the real
    # report as plain chat text on an earlier pass, so "latest" would clobber it.
    atomic_write(report_path, max(texts, key=len, default="") or "(no report produced)")
    print("  [worker] write_report never called - persisted the longest reply instead")


def build_agent(corpus_root: Path, report_path: Path, topic: str,
                judge_instructions: str, log: RunLog, max_passes: int,
                max_tool_calls: int):
    """Build the looping agent and its report-write flag.

    The worker's tool set is byte-for-byte the reflexion worker's; the judge
    is a bare ``OllamaChatClient`` with none of it. The judge is handed the
    report's text by the predicate (``judging.make_judge_predicate``), so the
    corpus is the one channel it is denied - the variable changed on purpose.

    ``AgentLoopMiddleware`` is ``@experimental`` and warns on construction.
    The warning is deliberately NOT filtered - a demo that hides the
    framework's own stability signal from its reader is lying by omission.

    The returned agent must be run WITH a session (``agent.create_session()``,
    see ``_amain``) for the accumulation this pattern claims. The loop
    *replaces* ``context.messages`` between passes rather than appending to
    them, so a session-less run hands pass 2 only the injected progress log
    plus the nudge - the original topic and every tool result are gone.

    Returns:
        ``(agent, flag)``.
    """
    from agent_framework import Agent, AgentLoopMiddleware
    from agent_framework.ollama import OllamaChatClient

    write_report, flag = make_report_tools(report_path)
    # No warnings.filterwarnings/catch_warnings here - AgentLoopMiddleware is
    # @experimental and warns on construction, and that warning is left to
    # reach the reader on purpose (see the docstring above).
    #
    # Two defaults are load-bearing and left unset on purpose:
    #
    #   fresh_context=False - with a session attached (see _amain) each pass
    #   runs against the accumulated transcript instead of restarting from the
    #   original prompt. That is the "grounded on its own chat history" half of
    #   the reflection vs. reflexion contrast; reflexion mints a fresh session
    #   per cycle. Setting True would additionally snapshot and restore the
    #   session between passes, discarding exactly that history.
    #
    #   inject_progress=True - the real continuity carrier, and invisible
    #   unless named. After every pass the loop appends that pass's text to a
    #   progress log and prepends it to the next pass's input as a "Progress
    #   so far:" user message. With a session attached the loop injects only
    #   the LATEST entry (the session already holds the earlier turns); with no
    #   session it injects the whole log.
    #
    # The session route is the EXPENSIVE one, and is chosen for fidelity, not
    # for cost: it re-sends the stored transcript - function_call and
    # function_result messages included, i.e. every corpus file the worker
    # read - plus the latest progress entry, where the session-less route
    # sends a digest of prior pass texts and nothing else. There is no
    # compaction_strategy on this agent (unlike hotl_demo's phase agents), so
    # a 3-pass corpus-reading run at the default num_ctx=4096 may silently
    # truncate; raise --num-ctx for multi-pass runs.
    #
    # judge_client is a bare OllamaChatClient with no tools/session/middleware
    # of its own - contrast with the worker Agent below, whose tools are
    # byte-for-byte the reflexion worker's. That asymmetry is the demo.
    #
    # One budget for the run; next_message resets it at each pass boundary.
    # Pass 1 has no boundary before it and needs no priming: PassBudget is born
    # spent=0, finalizing=False, which is exactly pass 1's state.
    #
    # Not even when max_passes == 1. Marking that lone pass finalizing closed
    # exploration after its FIRST call while worker.md had just promised
    # max_tool_calls of them - and next_message never fires on a single-pass
    # run, so finalize.md, the only text that explains the strip, could not be
    # delivered either. Delivery is still forced without it: at exhaustion the
    # read tools are stripped and BUDGET_SPENT says why.
    budget = PassBudget(max_calls=max_tool_calls)
    loop = AgentLoopMiddleware(
        make_judge_predicate(OllamaChatClient(), judge_instructions, log,
                             resolve_num_ctx(), report_path),
        max_iterations=max_passes,
        next_message=make_next_message(
            budget, max_passes,
            render_finalize_message(topic=topic, max_passes=max_passes)),
    )
    agent = Agent(
        client=OllamaChatClient(),
        name="worker",
        instructions=_WORKER_INSTRUCTIONS,
        tools=make_corpus_tools(corpus_root) + [write_report],
        middleware=[make_budget_middleware(budget, BUDGETED_TOOL_NAMES, "worker"), loop],
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
    parser.add_argument("--max-tool-calls", type=int, default=12, metavar="N",
                        help="per-pass read-tool budget; the last 3 calls are "
                             "announced and exploration closes when it is spent "
                             "(default: %(default)s)")
    parser.add_argument("--model", default=os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL),
                        help="Ollama model tag (default: %(default)s)")
    parser.add_argument("--num-ctx", type=int,
                        default=int(os.environ.get("OLLAMA_NUM_CTX", 4096)),
                        help="Ollama context window in tokens (default: %(default)s)")
    args = parser.parse_args()
    if args.max_passes < 1:
        parser.error("--max-passes must be >= 1")
    if args.max_tool_calls < 1:
        parser.error("--max-tool-calls must be >= 1")
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
        corpus_root, report_path, args.topic,
        render_judge_instructions(topic=args.topic), log,
        args.max_passes, args.max_tool_calls)

    print(f"Topic: {args.topic}")
    print(f"Budget: {args.max_passes} passes, {args.max_tool_calls} tool calls "
          f"per pass (the judge holds NO tools)")
    # One session for the whole run - the same idiom as reflexion's worker,
    # used for the opposite purpose. AgentLoopMiddleware REPLACES
    # context.messages between passes, so without a session pass 2 would see
    # only the injected progress log and the nudge: no topic, no tool results,
    # no prior report. The session is what makes each pass build on the last
    # (an InMemoryHistoryProvider is auto-attached, loading the stored
    # transcript before each pass and storing that pass's messages after), and
    # it is also what narrows the loop's progress injection to the latest entry
    # instead of re-sending every prior pass in full.
    session = agent.create_session()
    result = await agent.run(
        render_worker_prompt(topic=args.topic, max_passes=args.max_passes,
                            max_tool_calls=args.max_tool_calls),
        session=session)
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
