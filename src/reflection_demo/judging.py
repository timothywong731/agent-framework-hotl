"""The tool-less judge: verdict reading, the run log, and the loop predicate.

This module deliberately avoids ``from __future__ import annotations`` - the
framework introspects callables handed to it, and string annotations are a
known trap in this repo.
"""
import json
from dataclasses import dataclass
from pathlib import Path

from agent_framework import JudgeVerdict, Message

# The framework's own fallback markers, reused verbatim so a judge that
# cannot honour response_format still lands on the same contract.
JUDGE_VERDICT_DONE = "VERDICT: DONE"
JUDGE_VERDICT_MORE = "VERDICT: MORE"


@dataclass
class Verdict:
    """One judged pass."""

    pass_no: int
    answered: bool
    reasoning: str


def read_verdict(value, text) -> tuple:
    """Normalize the judge's reply to ``(answered, reasoning)``.

    ``value`` is the parsed :class:`JudgeVerdict` when the client honoured
    ``response_format``; otherwise fall back to the explicit markers, with
    ``MORE`` winning whenever the reply is ambiguous or marker-less.

    This fails OPEN - an unreadable verdict keeps the loop running and costs
    one pass. That is the opposite of the reflexion reviewer, which fails
    CLOSED and rejects. Both are right for their pattern: reflexion must
    never ship unverified work as approved, whereas here the pass cap is
    what guarantees termination.

    Args:
        value: ``ChatResponse.value`` - a ``JudgeVerdict`` or anything else.
            Callers must read that property inside a ``try``: it parses
            lazily and raises on a non-conforming reply, so the caller
            normalizes the raise to ``None`` before getting here.
        text: ``ChatResponse.text`` - the raw reply, used for the fallback.
    """
    if isinstance(value, JudgeVerdict):
        return value.answered, value.reasoning
    raw = (text or "").strip()
    upper = raw.upper()
    # No DONE marker and no MORE marker both take the else branch and land on
    # False - ambiguous or marker-less replies are MORE, per the fail-OPEN
    # policy above.
    answered = False if JUDGE_VERDICT_MORE in upper else JUDGE_VERDICT_DONE in upper
    return answered, raw


def summarize(verdicts, max_passes: int) -> tuple:
    """Terminal outcome of a finished run: ``("answered"|"unjudged", passes)``.

    ``AgentLoopMiddleware`` checks ``max_iterations`` BEFORE evaluating
    ``should_continue`` (``agent_framework/_harness/_loop.py``), so the judge
    is never consulted on the capped pass: a capped run has one more pass
    than it has verdicts, and that last pass ships unjudged. The reflexion
    parallel is the forced finalize shipping unapproved.

    Args:
        verdicts: Every verdict recorded this run, in pass order.
        max_passes: The ``--max-passes`` cap the loop was built with.
    """
    # The "checks the cap before the judge" premise this function relies on
    # is pinned by tests/test_reflection_loop.py::
    # test_max_iterations_short_circuits_before_the_judge, so a framework
    # upgrade that changes the check order fails loudly here, not silently
    # in this arithmetic.
    if verdicts and verdicts[-1].answered:
        return "answered", len(verdicts)
    return "unjudged", max_passes


class RunLog:
    """Append-only ``reflection_log.jsonl``; the sole writer.

    Deliberately the same line shape as the reflexion demo's
    ``review_log.jsonl`` so the two runs can be read side by side.
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self.verdicts = []

    def record(self, verdict: Verdict) -> None:
        """Append one judged pass."""
        self.verdicts.append(verdict)
        self._append({"pass": verdict.pass_no, "answered": verdict.answered,
                      "reasoning": verdict.reasoning, "judged": True})

    def finish(self, max_passes: int, report_path: Path) -> tuple:
        """Write the unjudged pass (if any) and the outcome line.

        Returns:
            ``(outcome, passes)`` as computed by :func:`summarize`.
        """
        outcome, passes = summarize(self.verdicts, max_passes)
        if outcome == "unjudged":
            self._append({"pass": passes, "answered": None,
                          "reasoning": None, "judged": False})
        self._append({"outcome": outcome, "passes": passes,
                      "report": str(report_path)})
        return outcome, passes

    def _append(self, record: dict) -> None:
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


def make_judge_predicate(judge_client, instructions: str, log: RunLog, num_ctx: int):
    """Build the ``should_continue`` predicate for ``AgentLoopMiddleware``.

    Follows the framework's own ``_build_judge_condition`` layout - same
    ``JudgeVerdict`` schema, same marker fallback - but records every
    verdict, which ``AgentLoopMiddleware.with_judge`` cannot: it builds its
    predicate internally, so an approving verdict's reasoning would be
    unobservable and the A/B would lose its most interesting line.

    One deliberate deviation: the framework's version splats
    ``*last_result.messages`` into the judge prompt, which forwards
    ``function_result`` content (raw tool output - i.e. the corpus, verbatim)
    and ``text_reasoning`` traces (which can quote that same output) verbatim
    to the judge. That is a real leak in ``_build_judge_condition`` /
    ``with_judge``, not a hypothetical - a worker that reads the corpus via
    ``read_file`` and never repeats it in its final text answer still hands
    the judge the corpus, because the tool result rides along in the message
    list. This defeats the asymmetry the whole demo is built to show. Here
    we forward only ``last_result.text`` - ``AgentResponse.text`` filters to
    text content, so tool results and reasoning traces never reach the
    judge, no matter what the worker read.

    The judge is a bare chat client: no tools, no session, no middleware, no
    corpus. It sees the original request and what the worker SAID (text
    only), never the report file and never raw tool output. That asymmetry
    is the reflection pattern.

    The judge bypasses ``Agent`` entirely (it is a bare chat client by
    design), so ``Agent``'s ``default_options={"num_ctx": ...}`` mechanism
    never reaches it. Without an explicit per-call ``num_ctx`` here, Ollama
    silently truncates the judge's context - and since ``fresh_context`` is
    ``False`` the transcript it's asked to judge only grows, so a late pass
    is exactly where a silent truncation would bite: the judge would verdict
    a report it only partially saw, with no error raised anywhere.

    Args:
        judge_client: Any chat client exposing ``get_response``.
        instructions: Rendered judge system instructions.
        log: The run log; every verdict is recorded before returning.
        num_ctx: Ollama context window, pinned per call since this client
            has no ``default_options`` of its own.

    Returns:
        An async predicate returning ``(keep_going, feedback)``.
    """
    async def should_continue(*, iteration, last_result, original_messages, **kwargs):
        messages = [
            Message("system", contents=[instructions]),
            Message("user", contents=[
                "Evaluate the agent's work. The user's original request follows:"]),
            *original_messages,
            Message("user", contents=["The agent's latest response was:"]),
            # .text, NOT *last_result.messages - see the docstring above for
            # why: splatting the raw message list (as the framework's own
            # with_judge does) hands the judge tool_result content (the
            # corpus, verbatim) and reasoning traces that can quote it too.
            # .text filters to TextContent only, which is the whole point.
            Message("assistant", contents=[last_result.text or "(no reply)"]),
            Message("user", contents=["Has the original request been fully addressed?"]),
        ]
        # num_ctx passed per call, not via Agent's default_options - the judge
        # is a bare client with no Agent wrapper to carry that option, and
        # without it Ollama truncates silently instead of erroring.
        response = await judge_client.get_response(
            messages, options={"response_format": JudgeVerdict, "num_ctx": num_ctx})
        # ChatResponse.value parses LAZILY: with response_format set it runs
        # JudgeVerdict.model_validate_json(text) on first access and RAISES
        # pydantic's ValidationError on any non-empty reply that is not
        # schema-conforming JSON - i.e. on exactly the input the VERDICT:
        # marker fallback below exists to handle. Unguarded, that exception
        # escapes should_continue -> _evaluate_stop -> agent.run() and kills
        # the run before persist_fallback and log.finish, so there is no
        # report and no outcome line. The framework's own
        # _build_judge_condition has the identical exposure, so a local model
        # that ignores response_format would crash MAF's built-in judge loop
        # too - the guard is here, not upstream. ValidationError subclasses
        # ValueError in pydantic v2, so no extra import is needed.
        try:
            value = response.value
        except ValueError:
            value = None  # unparseable -> fall through to the VERDICT: markers
        answered, reasoning = read_verdict(value, response.text)
        log.record(Verdict(pass_no=iteration, answered=answered, reasoning=reasoning))
        print(f"  [judge] pass {iteration}: {'ANSWERED' if answered else 'MORE WORK'}")
        if not answered and reasoning:
            print(f"  [judge] {reasoning}")
        return (not answered), (reasoning or None)

    return should_continue


def make_next_message():
    """Build the ``next_message`` callable that relays the judge's reasoning.

    ``AgentLoopMiddleware``'s default next-message is a bare "continue"
    nudge that would drop the feedback on the floor; ``with_judge`` supplies
    its own relay, and since this demo builds the loop directly it must
    supply one too. This is the verbal-feedback channel: without it the
    judge could reject forever without ever saying why.
    """
    def next_message(*, feedback=None, **kwargs) -> str:
        if feedback:
            return ("A reviewer judged your previous report incomplete.\n\n"
                    f"Reviewer feedback: {feedback}\n\n"
                    "Revise the report to address it and save the COMPLETE "
                    "revised report with write_report.")
        return ("Keep improving the report and save the complete text with "
                "write_report.")

    return next_message
