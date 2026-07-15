# Live Scratchpad Steering — Design

**Date:** 2026-07-15
**Status:** Approved
**Extends:** `2026-07-14-hotl-pipeline-design.md` (§6 agent tools, §12 testing) — that spec stays authoritative for everything else.
**Verified against:** `agent-framework` 1.11.0 (every API claim below was checked against the installed package, not the docs alone).

## 1. Purpose

Turn `scratchpad.md` from a **read-once** steering file into a **live** one. Today an agent reads the scratchpad at the start of its phase and never looks again; an edit made 30 seconds into a three-minute analyzer run is invisible for the rest of that run. After this change, an edit made while the pipeline is running is pushed to whichever agents are still working, and each agent decides for itself whether the new guidance is relevant to its phase.

This strengthens the demo's core thesis. The ledger/review gate is the **structured** human channel and fires exactly once; the scratchpad is the **freeform** channel with no such limit. Making it live is what makes the human genuinely *on* the loop — supervising a running process — rather than merely *before* it.

## 2. The constraint that shapes everything

**An in-flight LLM call cannot be interrupted.** There is no mechanism, in this framework or any other, to make a model that is mid-inference notice a file change. "Notify the working agent" therefore always means *deliver at the next boundary*.

The boundaries inside a single `agent.run()` are **model calls**, and our agents make several per phase — every tool call forces another round trip. That is the delivery opportunity.

Two consequences follow, and they are the whole design:

1. **A file watcher (`watchdog`, inotify) buys nothing.** Delivery waits for a boundary regardless of how fast detection is, so polling `st_mtime` *at* that boundary is observably identical — with no dependency and no thread bridging into asyncio. Rejected.
2. **We do not need to build the delivery mechanism.** The framework ships it.

## 3. What the framework already provides

`agent_framework.MessageInjectionMiddleware` (chat middleware, no-arg constructor):

> Chat middleware that injects queued session messages into the model call loop. Messages can be enqueued for an `AgentSession` before a run starts **or while a run is in progress, including from tool code that receives a `FunctionInvocationContext`**. Pending messages are stored in `session.state` and drained into the next model call for that session. After a model call completes, the middleware loops internally only when there are newly queued messages and the response does not contain function calls that the function invocation layer must handle.

Two properties matter:

- **Late notes still land before the report.** If an analyzer was about to finish and emit its phase report (a response with no function calls), and a steering note is queued, the middleware forces one more model call with the message injected. The agent reconsiders *before* writing its report rather than silently missing the boat. Hand-rolling this is fiddly; it is free here.
- **Concurrent analyzers are isolated by construction.** Pending messages live in `session.state`, and each `PhaseExecutor` owns its own agent and session. There is no shared queue to get wrong.

Relevant API surface, as installed:

| Symbol | Signature |
|---|---|
| `MessageInjectionMiddleware()` | no-arg constructor |
| `.enqueue_messages(session, messages)` | `messages: AgentRunInputs = str \| Content \| Message \| Sequence[...]` — plain strings are accepted |
| `FunctionInvocationContext` | `(function, arguments, session, metadata, result, kwargs, tools)` |
| `Agent.create_session()` | `(*, session_id=None) -> AgentSession` |
| `Agent.run(...)` | `(messages, *, session: AgentSession \| None = None, ...)` |
| `Agent(...)` | `middleware: Sequence[MiddlewareTypes] \| None` |

`MiddlewareTypes` is a union spanning agent, function **and** chat middleware, in both class and plain-callable form — so the mixed list `[injector, steering_mw]` in §4 is a supported shape, not a workaround.

Function middleware in this version takes `call_next: Callable[[], Awaitable[None]]` — **no arguments**, mutate the shared `context` then `await call_next()`. (Confirmed from `MessageInjectionMiddleware.process`'s own signature and from the `MiddlewareTypes` union itself; some published API-reference pages still show an older `next(context)` form.)

## 4. Design

Detection rides on the tool calls the agents are already making. No background task, no thread, no session registry, no new dependency.

### New module: `src/hotl_demo/steering.py`

Kept out of `tools.py`, whose module contract is "tools the agent calls" — middleware is a delivery channel, not a tool. It imports `SCRATCHPAD_PATH` from `tools.py` rather than moving it (shortest correct diff).

```python
class ScratchpadWatch:
    """Per-agent watermark over the steering file.

    Not shared between agents: the two analyzers run concurrently and each
    needs its own notion of "have I told THIS agent yet".
    """
    def __init__(self, path: Path) -> None:
        self._path, self._seen = path, None

    def poll(self) -> str | None:
        """Return the scratchpad text if it changed since the last poll, else None.

        The first poll baselines and returns None, so an agent is never handed
        back the content it just fetched with read_scratchpad.
        """
        text = self._path.read_text(encoding="utf-8") if self._path.exists() else ""
        if text == self._seen:
            return None
        first, self._seen = self._seen is None, text
        return None if first else text
```

`poll()` is the only decision-bearing logic in the feature, and it is a pure-ish function of (file content, prior state) — directly unit-testable with no LLM, per the repo's testing convention.

### The middleware

```python
# Concatenated, never str.format-ed: the scratchpad is human-written and may
# contain literal braces. Same gotcha as the memory nudge in phases.py.
_NOTICE_HEAD = (
    "\n\n[OPERATOR STEERING UPDATE - the human edited the scratchpad while "
    "you were working. Current scratchpad contents:]\n"
)
_NOTICE_TAIL = (
    "\n[Adapt now if this affects your current phase; otherwise ignore it "
    "and continue.]"
)


def make_steering_middleware(watch: ScratchpadWatch,
                             injector: MessageInjectionMiddleware,
                             label: str):
    @function_middleware                             # REQUIRED - see below
    async def steering_middleware(
        context: FunctionInvocationContext,
        call_next: Callable[[], Awaitable[None]],
    ) -> None:
        await call_next()
        new = watch.poll()
        if new and context.session is not None:
            injector.enqueue_messages(
                context.session, [_NOTICE_HEAD + new + _NOTICE_TAIL]
            )
            print(f"  [steering] scratchpad update queued for {label}")
    return steering_middleware
```

**`@function_middleware` is load-bearing, not decoration.** The framework's `_determine_middleware_type` classifies a plain callable by `inspect.signature(...)` and then `first_param.annotation.__name__`. Under `from __future__ import annotations` — which the rest of this package uses — that annotation is the *string* `"FunctionInvocationContext"`, and `str` has no `__name__`, so classification falls through to `MiddlewareException: Cannot determine middleware type`. This is the same trap CLAUDE.md already documents for `review.py`'s `@response_handler`.

The decorator sets a `_middleware_type` marker that is checked first and is immune to string annotations (verified empirically with the future import active). Keeping both the decorator and the annotation means they agree — satisfying the framework's mismatch check — and lets `steering.py` use the package's normal future import rather than becoming a second "never add this import" landmine.

The notice deliberately delegates the relevance judgement to the model — matching the request that the agent "take that into account **if relevant**".

### Flow

| Step | Actor |
|---|---|
| 1 | Human edits `scratchpad.md` mid-run (any external process, any editor) |
| 2 | The working agent calls any tool (`read_file`, `update_memory`, …) |
| 3 | `steering_middleware` runs after the tool, `watch.poll()` returns the new text |
| 4 | Text is enqueued to that agent's session via `injector.enqueue_messages` |
| 5 | `MessageInjectionMiddleware` drains it into the agent's next model call — forcing one extra call if the agent was about to stop |
| 6 | Agent adapts or ignores, then continues its phase |

## 5. Changes to existing code

| File | Change |
|---|---|
| `steering.py` | **New.** `ScratchpadWatch` + `make_steering_middleware` (~40 lines) |
| `phases.py` | `PhaseExecutor.__init__`: build a `ScratchpadWatch` + one `MessageInjectionMiddleware`, pass `middleware=[injector, steering_mw]` to `Agent(...)`; create an explicit session via `agent.create_session()` |
| `phases.py` | `_invoke`: pass `session=self._session` to `self._agent.run(...)` |

`middleware` is passed at agent construction, so the `agent=` test seam in `PhaseExecutor.__init__` is untouched.

### The explicit session is a bug fix, not a clarification

An earlier revision of this spec claimed the explicit session merely made today's implicit session persistence explicit. **That was wrong**, and verifying it against the installed package changed the design. The facts:

- `Agent.__init__` stores no session, and `Agent.run` only forwards its `session` parameter — there is no instance-level session.
- `_prepare_session_and_messages` does `provider_session = session; if provider_session is None and self.context_providers: provider_session = AgentSession()`.
- `Agent(client=..., name=..., instructions=..., tools=...)` yields `context_providers == []`.

With no session passed and no context providers, `session_id` is `None` and **every `run()` call is an independent, stateless turn**. So both CLAUDE.md and the `PhaseExecutor` docstring are false where they claim *"the agent session persists across `run()` calls… the follow-up turn sees the whole earlier exploration"*. Consequences today:

- **The memory nudge works anyway** — but by accident, not by session: `_NUDGE_PREFIX + text` embeds the report directly in the prompt, so the model needs no history.
- **The report retry is latently broken.** `_REPORT_RETRY` says *"You explored the sources but did not produce the phase report"* to a model that has no memory of exploring anything. It re-explores from scratch at best.

Adding an explicit session therefore **repairs** the retry path and makes the docstring true. This is a deliberate, in-scope behaviour change — the enqueue handle requires a session regardless, and shipping the handle while leaving the documentation false would be worse.

**Session lifetime: one per run cycle, not one per executor.** `_run_initial` and `on_revision` each mint a fresh session via `agent.create_session()`. Within a cycle, the nudge and retry see the exploration (the fix). Across cycles, the revision prompt stays self-contained — it already carries `previous_report` and the human answers by design — and history cannot grow unboundedly across a re-run.

## 6. Non-goals and known limitations

- **No re-trigger of completed phases.** This notifies agents *still running*. A scratchpad edit made during `final_report` changes nothing upstream. Re-adjudicating settled work is the review gate's job, and the review-once rule deliberately bounds it. Explicitly out of scope.
- **Detection is tool-call-bound.** An agent that goes a long stretch without calling a tool will not notice a change until its next tool call. In practice every phase calls `update_memory`/`raise_question`, and analyzers call `list_files`/`read_file` heavily. The alternative — a background poller plus a registry of live sessions — was rejected: delivery still waits for the next model call, so it buys detection latency only during tool-free stretches, at the cost of real lifecycle machinery.
- **Revision runs are covered for free.** Same agent, same middleware, no extra work.
- **No delta/diff.** The full current scratchpad is sent. The file is small and human-written; a diff would add `difflib` bookkeeping for no clarity gain.
- **Cost is zero when idle.** `poll()` only reads the file; a notice is only enqueued when the content actually changed (compared by content, not mtime, so a no-op save is correctly silent).

## 7. Error handling

- **Clearing or deleting the scratchpad mid-run is silent, not a crash.** A missing or emptied file makes `poll()` return `""`, which the middleware's truthiness check (`if new`) skips: withdrawing guidance is not new guidance to act on, and there is nothing useful to tell the model. The watermark still advances, so if the human later writes fresh content it is delivered normally.
- `context.session is None` (should not occur once sessions are explicit) → skip silently rather than raise. Middleware must never break a tool call.
- The scratchpad is read, never written, by this feature; `ensure_scratchpad`'s never-truncate guarantee is unaffected.

## 8. Testing (LLM-free)

- **`ScratchpadWatch.poll()`** — unit tests via `tmp_path`: first poll baselines to `None`; unchanged content → `None`; changed content → new text; no-op re-save of identical content → `None`; missing file → `None` on first poll; content then cleared → `""`, and content added again afterwards → the new text (proves the watermark advanced through the clear).
- **`make_steering_middleware`** — direct test with a stub `FunctionInvocationContext` and a fake injector recording `enqueue_messages` calls: asserts the notice is enqueued exactly once per change, is not enqueued when unchanged, is not enqueued when the scratchpad was cleared, and that `call_next` is always awaited.
- **Note:** the existing `FakeAgent` seam bypasses middleware entirely (middleware lives on the real `Agent`), so this feature cannot ride the executor tests and needs the direct tests above. The opt-in `OLLAMA_E2E` live test is where the wiring is exercised end to end.

## 9. Decisions log

| Decision | Choice | Alternatives considered |
|---|---|---|
| Delivery mechanism | Framework's `MessageInjectionMiddleware` | Hand-rolled: append notice to the tool's `context.result` (works, but loses the "force one more model call" property) |
| Detection | Poll `ScratchpadWatch` inside function middleware | Background asyncio poller + live-session registry (same delivery latency, real lifecycle cost); `watchdog` file watcher (adds a dep and a thread for zero observable gain) |
| Zero-code option | Rejected | Prompt agents to "re-read the scratchpad periodically" — pull not push, unreliable, and wastes calls when nothing changed |
| Watermark scope | Per-agent (`ScratchpadWatch` per `PhaseExecutor`) | Global/shared — wrong: concurrent analyzers would steal each other's notifications |
| Payload | Full current scratchpad text | Delta via `difflib` (bookkeeping for no gain on a small file) |
| Module | New `steering.py` | Add to `tools.py` (breaks its "tools only" contract) |
| Middleware kind detection | `@function_middleware` decorator **plus** the annotation | Annotation alone — verified to raise `MiddlewareException` under `from __future__ import annotations`; dropping the future import instead would make `steering.py` a second `review.py`-style landmine |
| Session | Explicit `agent.create_session()`, passed to every `run()` | Implicit (today's behaviour) — no handle to enqueue into, and verified stateless per call, which silently breaks `_REPORT_RETRY` |
| Session lifetime | One per run cycle (`_run_initial` / `on_revision`) | One per executor (revision would inherit the whole initial exploration, inflating history and undercutting the self-contained revision prompt); one per `run()` (no persistence — the status quo bug) |
| CLAUDE.md + `PhaseExecutor` docstring | Corrected as part of this change | Leave them asserting session persistence that does not exist |
| Relevance filtering | The model decides, per the notice wording | Code-side heuristics matching scratchpad text to phases (guesswork, and contrary to the ask) |
