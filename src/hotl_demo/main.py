"""CLI runner: preflight, run the workflow, adjudicate the review gate.

Three flows (see the checkpointing spec):

* default        - interactive: stream events, prompt on stdin at the gate,
                   resume with ``run(responses={...})`` in the same process.
* ``--pause``    - checkpointing on: at the gate, seed ``review.jsonl`` with
                   one ``{"id", "answer"}`` line per open question and EXIT;
                   the human answers at leisure (question text stays in
                   ``ledger.jsonl``).
* ``--resume``   - restore the gate checkpoint of a --pause run, apply the
                   answers from ``review.jsonl``, drive revisions to the
                   final report. Re-runs zero phases.
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

REVIEW_FILENAME = "review.jsonl"
CHECKPOINT_DIRNAME = "checkpoints"


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


def render_review_lines(open_questions: list[dict]) -> str:
    """Seed the answer sheet: one ``{"id", "answer": ""}`` line per open question.

    Only the human's input lives in review.jsonl - the question text stays in
    ledger.jsonl (agent-curated, read-only); a frontend joins on ``id``.
    Unedited lines decline naturally at resume time.

    Args:
        open_questions: Ledger entries with ``status == "open"``, in ledger
            order (as returned by ``ArtifactStore.open_questions``).

    Returns:
        JSONL text, one seeded record per question.

    Example:
        >>> render_review_lines([{"id": "q-1"}])
        '{"id": "q-1", "answer": ""}\\n'
    """
    return "".join(json.dumps({"id": q["id"], "answer": ""}) + "\n" for q in open_questions)


def parse_review_answers(text: str) -> dict[str, str]:
    """Parse review.jsonl into ``{question_id: answer}``.

    Loud on ANY malformed line: a parse error must never degrade into
    "decline", which would silently discard the human's gathered answers and
    proceed on defaults. Blank lines are allowed (editors add them).

    Args:
        text: Full review.jsonl content.

    Returns:
        Mapping of question id to raw answer text (``""`` = decline).

    Raises:
        ValueError: Malformed JSON, non-object line, missing/duplicate id, or
            a non-string answer - always naming the offending line number.

    Example:
        >>> parse_review_answers('{"id": "q-1", "answer": "yes"}\\n')
        {'q-1': 'yes'}
    """
    answers: dict[str, str] = {}
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{REVIEW_FILENAME} line {lineno}: invalid JSON ({exc})")
        if not isinstance(record, dict) or not isinstance(record.get("id"), str):
            raise ValueError(
                f"{REVIEW_FILENAME} line {lineno}: expected an object with a string 'id'")
        if not isinstance(record.get("answer"), str):
            raise ValueError(
                f'{REVIEW_FILENAME} line {lineno}: "answer" must be a string (use "" to decline)')
        if record["id"] in answers:
            raise ValueError(f"{REVIEW_FILENAME} line {lineno}: duplicate id {record['id']!r}")
        answers[record["id"]] = record["answer"]
    return answers


def map_answers(pending: dict[str, "LedgerQuestionRequest"],
                answers: dict[str, str]) -> tuple[dict[str, str], list[str]]:
    """Join the answer sheet onto the gate's pending requests.

    Missing id = decline (""), consistent with "empty answer = decline".

    Args:
        pending: ``request_id -> LedgerQuestionRequest`` collected from the
            resumed workflow's re-emitted events.
        answers: ``question_id -> answer`` from review.jsonl.

    Returns:
        ``(responses, unknown_ids)`` - responses keyed by request id, ready
        for ``workflow.run(responses=...)``, plus any sheet ids that match no
        pending question (sorted; the caller warns about them).
    """
    responses = {rid: answers.get(q.question_id, "") for rid, q in pending.items()}
    unknown = sorted(set(answers) - {q.question_id for q in pending.values()})
    return responses, unknown


async def _amain() -> None:
    """Parse args, preflight, then dispatch to the right flow."""
    parser = argparse.ArgumentParser(description="HOTL cloud migration readiness demo")
    parser.add_argument("--model", default=os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL),
                        help="Ollama model tag (default: %(default)s)")
    parser.add_argument("--data", type=Path, default=Path("sample_data"),
                        help="sample data directory")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--pause", action="store_true",
                      help="checkpoint and exit at the review gate instead of prompting; "
                           "fill in <run_dir>/review.jsonl, then rerun with --resume. "
                           "A run that raises no questions never pauses.")
    mode.add_argument("--resume", type=Path, metavar="RUN_DIR", default=None,
                      help="resume a --pause run from its review-gate checkpoint, "
                           "applying the answers in RUN_DIR/review.jsonl. "
                           "Pass the same --model the run was paused with.")
    args = parser.parse_args()
    os.environ["OLLAMA_MODEL"] = args.model  # OllamaChatClient reads this
    base_url = normalize_host(os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    preflight(base_url, args.model)
    ensure_scratchpad(SCRATCHPAD_PATH)

    # import here so --help and preflight failures never touch the framework
    from agent_framework import FileCheckpointStorage, WorkflowCheckpointException
    from .pipeline import (
        ALLOWED_CHECKPOINT_TYPES,
        WORKFLOW_NAME,
        build_workflow,
        gate_checkpoint,
    )

    if args.resume is not None:
        run_dir = args.resume
        review_path = run_dir / REVIEW_FILENAME
        if not review_path.exists():
            raise SystemExit(
                f"{review_path} not found - only runs started with --pause can be resumed.")
        answers = parse_review_answers(review_path.read_text(encoding="utf-8"))
        store = ArtifactStore(run_dir, REPOS)  # reopening preserves memory + ledger
        open_ids = {q["id"] for q in store.open_questions()}
        if not (set(answers) & open_ids):
            # Resume is NOT idempotent (a second pass would re-run every
            # revision), so refuse loudly - distinguishing the two causes.
            report = run_dir / "final_report.md"
            if report.exists():
                raise SystemExit(f"Already resumed - final report at {report}")
            raise SystemExit(
                "Answers were already applied but no final report exists - the previous "
                "resume likely crashed mid-revision; start a fresh run "
                "(mid-revision recovery is out of scope).")
        storage = FileCheckpointStorage(run_dir / CHECKPOINT_DIRNAME,
                                        allowed_checkpoint_types=ALLOWED_CHECKPOINT_TYPES)
        workflow = build_workflow(store, args.data, checkpoint_storage=storage)
        gate = gate_checkpoint(await storage.list_checkpoints(workflow_name=WORKFLOW_NAME))
        if gate is None:
            raise SystemExit(
                f"No review-gate checkpoint under {run_dir / CHECKPOINT_DIRNAME}.\n"
                "Either this run was not started with --pause, or a message type is "
                "missing from ALLOWED_CHECKPOINT_TYPES - unreadable checkpoint files "
                "are SILENTLY skipped, so check for decode warnings above.")
        try:
            await _drive(workflow, store, checkpoint_id=gate.checkpoint_id, answers=answers)
        except WorkflowCheckpointException as exc:
            # e.g. graph_signature_hash mismatch: prompts/repos edited since pause
            raise SystemExit(
                f"The pipeline changed since this run was paused ({exc}).\n"
                "A checkpoint only fits the graph that wrote it - start a fresh run.")
        return

    run_dir = Path("output") / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    store = ArtifactStore(run_dir, REPOS)
    storage = None
    if args.pause:
        storage = FileCheckpointStorage(run_dir / CHECKPOINT_DIRNAME,
                                        allowed_checkpoint_types=ALLOWED_CHECKPOINT_TYPES)
    # storage=None on the default path: the interactive flow is byte-for-byte
    # unchanged; checkpointing risk stays opt-in.
    workflow = build_workflow(store, args.data, checkpoint_storage=storage)
    await _drive(workflow, store, message="start", pause=args.pause)


async def _drive(workflow, store: ArtifactStore, *, message: str | None = None,
                 checkpoint_id: str | None = None,
                 answers: dict[str, str] | None = None, pause: bool = False) -> None:
    """Shared run loop for all three flows.

    First iteration: fresh start (``message``) or checkpoint restore
    (``checkpoint_id``). Later iterations resume the SAME workflow instance
    with the collected responses - the framework's pause/resume contract.
    Verdict source: ``answers`` (--resume; missing id = decline), stdin
    (interactive), or nobody - ``pause`` seeds review.jsonl and exits.

    Args:
        workflow: The built workflow.
        store: The run's artifact store (pause files + final print).
        message: Start message for a fresh run.
        checkpoint_id: Gate checkpoint to restore instead of starting fresh.
        answers: ``{question_id: answer}`` from review.jsonl, or ``None``.
        pause: Seed the answer sheet and exit when the gate opens.
    """
    responses: dict[str, str] | None = None
    first = True
    while True:
        if not first:
            stream = workflow.run(stream=True, responses=responses)
        elif checkpoint_id is not None:
            stream = workflow.run(checkpoint_id=checkpoint_id, stream=True)
        else:
            stream = workflow.run(message, stream=True)
        first = False
        pending: dict[str, LedgerQuestionRequest] = {}
        async for event in stream:
            if event.type == "request_info" and isinstance(event.data, LedgerQuestionRequest):
                pending[event.request_id] = event.data
            elif event.type == "output":
                print(f"\nFinal report: {event.data}")
        if not pending:
            break  # the run finished without pausing: we are done
        if pause:
            _write_pause_files(store, len(pending))
            return
        if answers is not None:
            responses, unknown = map_answers(pending, answers)
            for qid in unknown:
                print(f"  warning: {REVIEW_FILENAME} id {qid!r} matches no pending "
                      "question - ignored")
        else:
            responses = {rid: _prompt_human(q) for rid, q in pending.items()}
    print(f"Run artifacts: {store.run_dir}")


def _write_pause_files(store: ArtifactStore, pending_count: int) -> None:
    """Seed the answer sheet and tell the human how to continue.

    Args:
        store: The run's store; the gate is idle, so ``open_questions()`` is
            exactly the presented set, in ledger order.
        pending_count: Number of pending gate requests (for the banner).
    """
    open_qs = store.open_questions()
    review_path = store.run_dir / REVIEW_FILENAME
    review_path.write_text(render_review_lines(open_qs), encoding="utf-8")
    print(f"\n== PAUSED at the review gate - {pending_count} open questions ==")
    for q in open_qs:
        where = f"{q['phase']}[{q['unit']}]" if q["unit"] else q["phase"]
        print(f"\n[{q['id']}] ({where}) {q['question']}")
        print(f"      Evidence: {q['context']}")
        print(f"      Default if declined: {q['default_assumption']}")
    print(f"\nFill in the answers in {review_path}")
    print('(one {"id", "answer"} JSON line per question; empty answer = decline; '
          f"question text lives in {store.run_dir / 'ledger.jsonl'})")
    print(f"Then: poetry run demo --resume {store.run_dir}")


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
