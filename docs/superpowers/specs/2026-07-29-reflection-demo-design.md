# Reflection Demo — Design

**Date:** 2026-07-29
**Status:** Approved
**Stack:** Python ≥3.10, Poetry, pytest, Microsoft Agent Framework
(`agent-framework` ≥1.11, `agent-framework-ollama`), Ollama `gemma4:31b`
(local; same client conventions as the HOTL and reflexion demos)

## 1. Purpose

Standalone demo of the **reflection mechanism**, built as a deliberate A/B
foil to `reflexion_demo`: the same corpus, the same default topic, the same
worker tool *set* and tool-call budget, the same report artifact — with
**one variable changed on purpose**, the critic's access to the *sources*.

"On purpose" rather than "only": two further differences fall out of the two
demos being separate packages built on different framework primitives, and are
recorded rather than glossed because a reader will find them anyway.

1. **The delivery retry.** Missing here (§8 — a second `agent.run` would
   re-enter the loop and burn a pass). This cuts against the reflection side,
   and *could*: read a reflection report that landed on the longest-reply
   fallback with that in mind.
2. **One conceded exploratory call on the last pass.** Reflexion's forced
   finalize *constructs* its worker without read tools, so it gets zero.
   Reflection's loop owns its agent and `remove_tools()` is reachable only
   from inside a tool call, so a reflection worker that ignores "do not start
   new exploration" gets exactly one call before the strip fires. One call
   against zero, on one pass of a run.

The tool-call *budget* used to be a third, larger asymmetry here (the
reflexion worker capped, this one unbounded) — see §6, now reversed: both
workers run under an identical per-turn/per-pass budget, with identical
countdown coaching and a byte-identical budget paragraph in their prompts.
The worker side of the A/B therefore differs in the two residues above and
nothing else — which is weaker than "nothing but the delivery retry", and
accurate.

- `reflexion_demo`: the critic is an `Agent` node in a cyclic graph holding
  `list_files` / `read_file` / `read_report`. It reads the report itself and
  re-checks it **against the sources**.
- `reflection_demo`: the critic is a bare `OllamaChatClient` called directly
  by `AgentLoopMiddleware`, with no tools, no session, and no corpus. It is
  handed the report's text by the predicate and can only judge it **on its
  own terms**.

Running both on the same topic and diffing `report.md` is the demo.

This spec previously predicted an outcome — that reflection would converge on
structure and prose and *miss* the planted evidence conflicts. Two live runs
did not reproduce it, and the reason is worth more than the prediction was.

A tool-less judge handles two worker failures very differently:

| Worker failure | Can a tool-less judge catch it? |
|---|---|
| **Omission** - thin report, no citations, a gap it admits to | **Yes.** It is visible on the face of the text. |
| **Fabrication** - a confident claim that contradicts a source | **No.** There is nothing to check it against. |

The planted conflicts are *omission* fuel, and both critics catch omissions —
so on that axis the two demos agree, and the A/B separates nothing. The
difference only bites when the worker asserts something **false**, because
that is the one thing a critic without the sources cannot test.

Observed, on `gemma4:31b`:

- 3 passes, `--num-ctx 16384`: the worker surfaced the Azure/S3 conflict
  unprompted on pass 1 and the judge approved. Nothing was missed.
- 2 passes, `--max-tool-calls 4`: the judge **rejected**, naming the specific
  file the worker admitted it had not read. The blind judge did useful work.
- 3 passes/cycles, `--max-tool-calls 3` — tight enough that neither worker
  can read the whole corpus: the reflection worker never reached the
  cloud-strategy document, did not flag the gap, and shipped a clean report
  recommending S3; the judge called it comprehensive and approved on pass 1.
  The reflexion reviewer opened the same document with its own tool calls,
  caught the identical gap, and rejected — the worker revised to Azure and
  was approved. Same model, same corpus, same budget: one critic shipped the
  mandate violation, the other caught it.

In the first two runs the worker was honest — starved of tool calls it
flagged the gap rather than inventing a recommendation, so the blind judge
had something visible to reject. The third run separates the demos because
the gap was silent: not admitted, not fabricated, just never reached — which
is exactly the failure a critic without the corpus cannot detect. The
README's "What the A/B does and does not show" section has the full
transcripts. This is one reproduction on one local model, not a general claim
that reflection underperforms; it establishes the failure mode is reachable,
not merely theoretical.

Standalone means: no imports from `hotl_demo` or `reflexion_demo`. The demo
shares only `sample_data/` and the repo's Ollama conventions. `tools.py` is
a deliberate copy of the reflexion demo's, not a shared import — see §9.

## 2. Scenario

Corpus and default topic are identical to `reflexion_demo` (that identity is
the experiment):

> Assess migrating OMS file storage from the NFS file store to Amazon S3.

The planted conflicts — the enterprise cloud strategy mandating **Azure**
while `s3_uploader.py` targets S3, in-region data-residency and
secrets-management standards, hardcoded NFS paths in `file_store.py` — are
reachable by the worker's read tools in both demos. Only the *critic's*
ability to verify them differs.

## 3. Architecture

New package, console script `poetry run reflection`
(`reflection = "reflection_demo.main:run"`).

```text
src/reflection_demo/
  __init__.py
  main.py       # argparse CLI, preflight, run dir, agent + loop wiring, narration
  judging.py    # the tool-less judge predicate + verdict log
  tools.py      # corpus read tools + write_report (copy of reflexion's)
  prompting.py  # Jinja2 env + render helper
  prompts/
    worker.md   # ONE template - no revision variant
    judge.md    # judge instructions incl. the criteria block
```

There is **no `graph.py` and no `budget.py`**. That absence is a finding,
not an omission: reflexion needs a `WorkflowBuilder` graph because there are
two participants exchanging typed messages; reflection has one participant,
so the framework expresses the whole loop as a single middleware on a single
agent.

### The loop

`AgentLoopMiddleware` (`agent_framework`, experimental as of 1.11.0) drives
everything. One `agent.run()` call is the entire run:

```python
agent = Agent(
    client=OllamaChatClient(),
    name="worker",
    instructions=_WORKER_INSTRUCTIONS,
    tools=make_corpus_tools(corpus_root) + [write_report],
    middleware=[AgentLoopMiddleware(
        should_continue=make_judge_predicate(
            OllamaChatClient(), judge_instructions, log, resolve_num_ctx(),
            report_path),
        max_iterations=args.max_passes,
        next_message=make_next_message(),
    )],
    default_options={"num_ctx": resolve_num_ctx()},
)
session = agent.create_session()
await agent.run(
    render_worker_prompt(topic=args.topic, max_passes=args.max_passes),
    session=session,
)
```

### Continuity between passes: the session and `inject_progress`

`AgentLoopMiddleware` **replaces** `context.messages` before each pass
(`context.messages = next_messages` in `_process_non_streaming`); it does not
append. Two mechanisms therefore carry continuity, and both must be named
because neither is visible at the call site:

- **The session.** `agent.run(..., session=session)` is what makes each pass
  build on the last. With a session attached and no context providers
  registered, `Agent` auto-attaches an `InMemoryHistoryProvider` that loads the
  stored transcript before every pass and stores that pass's input and response
  messages after it, so pass *N* sees the original topic prompt, every earlier
  reply, and every tool result. **Without** a session, pass 2's entire input is
  the injected progress log plus the nudge — topic and tool results gone — which
  would make the "grounded on its own chat history" claim false.
- **`inject_progress`**, left at its default `True`. After each pass the loop
  appends that pass's text to a progress log and prepends it to the next pass's
  input as a single `Progress so far:` user message. With a session attached the
  loop injects only the **latest** entry (`_resolve_next_message` narrows to
  `progress[-1:]` because the session already holds the earlier turns); with no
  session it injects the whole log.

The session route is the **more expensive** of the two, and is chosen for
fidelity rather than cost: it re-sends the stored transcript — `function_call`
and `function_result` messages included, i.e. every corpus file the worker read
— *plus* the latest progress entry, where the session-less route sends only a
digest of prior pass texts. The latent risk is worth stating plainly: there is
no `compaction_strategy` on this agent (unlike `hotl_demo`'s phase agents), so
a 3-pass corpus-reading run at the default `num_ctx=4096` may silently
truncate. `--num-ctx` exists for that, and the README's gotchas say so.

`fresh_context` stays at its default `False`. Setting it `True` would restart
each pass from the original prompt *and* snapshot/restore the session around the
loop, discarding exactly the accumulated history that distinguishes reflection
from reflexion's fresh `AgentSession` per cycle.

The judge's `OllamaChatClient()` is bare — it bypasses `Agent` by design, so
`default_options` (an `Agent`-level mechanism) never reaches it; `num_ctx`
is instead pinned per call inside `make_judge_predicate`, the same
`options={...}` route `hotl_demo/compaction.py` uses for its bare client.
`report_path` is threaded in for the same reason: the judge has no tool to
open the report with, so the predicate reads it and hands over the text (§4).

`AgentLoopMiddleware` is decorated `@experimental` and emits one
`ExperimentalWarning` on construction. It is **not** filtered: a demo that
hides the framework's own stability signal from its reader is lying by
omission. The warning is mentioned in the README instead.

Packaging: `reflection_demo` must be added to both `[project.scripts]` and
the `[tool.poetry] packages` list in `pyproject.toml` — omitting the second
yields an import error at entry-point time, not at install time.

### Why an explicit predicate instead of `with_judge`

`AgentLoopMiddleware.with_judge(judge_client, criteria=[...])` is the
framework's one-line expression of this exact pattern, and it is what the
reflexion design rejected in its §12 ("judge is a bare chat client without
tools — breaks information parity"). It is not used here for one reason:
it builds the `should_continue` predicate internally, so the judge's
verdicts are unobservable. Approval could only be *inferred* from "the loop
stopped before the cap", and the approving verdict's reasoning — the single
most interesting line in an A/B run — would be lost.

`make_judge_predicate` is ~20 lines making the identical call
(`judge_client.get_response(msgs, options={"response_format": JudgeVerdict,
"num_ctx": num_ctx})`) against the same public `JudgeVerdict` schema, and
logs every verdict. The README documents `with_judge` as the one-liner this
expands to.

### Rubric parity

The judge's rubric is **identical to the reviewer's on Actionability and all
but identical on Coverage** — the judge's Coverage line drops "in the sources",
the only concession to a critic that holds none. Accuracy is necessarily
weaker, because the judge cannot verify that claims match their sources. If the two
critics were also given different standards on those two dimensions, the A/B
would confound two variables and prove nothing. This documented asymmetry in
Accuracy is the distinction the demo exists to demonstrate.

## 4. Information asymmetry

| | reflexion reviewer | reflection judge |
|---|---|---|
| What it is | `Agent` node in a cyclic graph | bare `OllamaChatClient`, called directly |
| Corpus tools | `list_files`, `read_file` | none |
| Report access | `read_report` — fetches the file itself | the predicate reads the file and hands over its text |
| Session | fresh `AgentSession` per cycle | none — the predicate rebuilds its prompt from the original request, the worker's latest text, and the report (the *worker* holds the run's one session, §3) |
| Judges | the report, **against the sources** | the report, **on its own terms** |

The framework states the tool-, session- and middleware-less shape itself, in
the docstring of `_build_judge_condition`: *"The judge is called directly (no
agent tools, session, or middleware)."*

**Both critics see the report; only the reflexion reviewer can check it
against the sources.** That is the single-variable difference, and it is
deliberately the *only* one: the judge is handed the report's text as prompt
text by the harness, holding no tool, no session and no middleware to fetch it
with, so corpus access is all that separates the two critics. It is also what
makes the reflection judge able to converge at all — the worker delivers
through `write_report`, so its reply text is an acknowledgement ("the report
has been written and saved successfully", measured at 87 characters against a
live model). A judge given only that rates the *claim* rather than the work:
live 2-pass runs had it reject with *"the agent provided no response and did
not produce the requested migration report"* while `report.md` held 5635
bytes, so `answered` was unreachable and every run shipped `unjudged` — one of
the two documented terminal states never occurred. It is also the arrangement
the literature uses: in Self-Refine the critic always sees the draft, it
simply has no ground truth to check it against.

**A finding about the framework itself:** the naive way to build "the
agent's response messages" - splatting `*last_result.messages` into the
judge prompt, which is exactly what the framework's own
`_build_judge_condition` does (and therefore what `with_judge` does too) -
does not honor this asymmetry. A tool-using pass's message list carries a
`function_result` content item holding the tool's raw output verbatim, plus
`text_reasoning` traces that can quote that same output, alongside the
worker's final text answer. Forwarding the whole list hands the judge the
corpus even when the worker's actual reply never repeats it - verified
empirically against a live Ollama run, with a worker explicitly instructed
not to quote the file: the secret marker reached the judge through the
`function_result` and `text_reasoning` content, though it was absent from
the concatenated text. `make_judge_predicate` therefore does not mirror the
framework's message assembly on this one point: it forwards
`last_result.text` only (`AgentResponse.text` filters to text content), so
tool results and reasoning traces never reach the judge no matter what the
worker read. `test_judge_never_sees_tool_results_or_reasoning_traces`
(`tests/test_reflection_loop.py`) pins this as a regression.

## 5. Termination, and the unjudged last pass

One bound: `--max-passes` (default 3) → `max_iterations`.

`AgentLoopMiddleware._evaluate_stop` short-circuits on the cap **before**
evaluating `should_continue`, so an expensive judge is not called once the cap
has fired. (Named by method, not by line: `_harness/_loop.py` line numbers
drift every release.) The consequence is behavioural and must be documented,
not hidden: **at the cap the judge is never consulted, and the report ships
unjudged.**

This is the exact parallel of reflexion's forced finalize shipping the
report *unapproved*, reached by a different route:

| | reflexion | reflection |
|---|---|---|
| Happy path | reviewer approves | judge answers `answered: true` |
| Budget exhausted | forced finalize turn, read tools stripped at construction, ships **unapproved** | cap short-circuits the judge, ships **unjudged** |

`test_reflection_loop.py` asserts the judge is not called on the capped
pass, so a framework upgrade cannot silently change the terminal semantics.

## 6. Tools

`tools.py` is a copy of `reflexion_demo/tools.py` minus `read_report` (no
participant reads the report through a *tool* here — the judge predicate reads
it in plain Python and hands over the text, §4): `make_corpus_tools` gives
the traversal-guarded `list_files` / `read_file` pair over `sample_data/`;
`make_report_tools` gives `write_report` plus the `ReportFlag` write-detection
cell. Same idioms — oversized reads truncated, failures returned as
`"ERROR: ..."` strings and never raised.

The worker's tool *set* is byte-for-byte the reflexion worker's. That is the
control in the experiment.

**Reversed: a per-pass tool-call budget and mid-turn stripping, matching the
reflexion worker's.** This section originally argued the strip mechanism was
"already demonstrated by `reflexion_demo`; repeating it here would double the
package for no additional insight." That reasoning is now known to be wrong
on both counts — see
`docs/superpowers/specs/2026-07-30-reflection-tool-budget-design.md` for the
full design, and briefly:

- **It removes a confound.** The unbounded reflection worker was a *second*,
  uncontrolled asymmetry against the reflexion worker's 12-calls-per-turn
  budget — the A/B differed in two variables, not one. Adding the budget is
  what makes "exactly one variable changed" (the critic) true of the worker
  side, rather than merely acknowledged and left standing (§1).
- **It fixes an observed defect.** A live `--max-passes 1` run hit
  `write_report never called - persisted the longest reply instead`: the
  worker spent its only pass exploring and never delivered. The countdown
  coaching gives it a reason to stop and write before that happens.

The budget is per **pass**, not per turn — `AgentLoopMiddleware` runs every
pass inside one `agent.run()`, so there is no turn boundary at which to mint
a fresh counter; `PassBudget.start_pass()` is called from `next_message`
instead, the one hook that fires between passes. `remove_tools()` is only
reachable from inside a tool call, so it cannot strip pre-emptively: the
final pass strips read tools after its first call rather than before it.
`budget.py` is still a deliberate copy of `reflexion_demo`'s, not a shared
import, consistent with §9's standalone rule; countdown wording is
byte-identical across both, pinned by `tests/test_budget_wording_parity.py`.

## 7. Prompts

Jinja2 markdown under `src/reflection_demo/prompts/`, no YAML frontmatter
(no phase discovery here), joining the markdown lint gate.

- `worker.md` — one template, one variant. Unlike the reflexion worker there
  is no `revision` or `finalize` mode: the loop middleware injects the
  judge's feedback as the next iteration's input, so the prompt never has to
  be re-rendered. Instructs: explore the corpus with tools, ground every
  claim in a source file, deliver via `write_report`, explore economically.

  It states that only a `write_report` call **reliably** saves the report, and
  names the consequence of skipping it (the longest reply of the run is
  salvaged instead, §8) — a flat "not stored anywhere" would be false, since
  `persist_fallback` does store it. It deliberately does **not** describe the
  critic's channels. An earlier
  wording ("a reviewer will read your reply and may send it back") did, and it
  cut two ways: it pulled the model toward putting the report in chat text
  instead of calling the tool — the live `--max-passes 1` run hit the
  longest-reply fallback — and it is an instruction the reflexion worker does
  not have, so it broke the worker parity §1 claims.
- `judge.md` — the judge's system instructions. Names what the judge is given
  (the request, the worker's latest reply, the saved report) and what it is
  not (the sources), and tells it to judge the saved report rather than the
  agent's description of it. Carries the same
  accuracy / coverage / actionability rubric as the reflexion reviewer (§3),
  plus the `JudgeVerdict` contract and the `VERDICT: DONE` / `VERDICT: MORE`
  marker fallback wording that the framework's default judge prompt uses.

## 8. Artifacts, console, error handling

`output/reflection_<timestamp>/`:

- `report.md` — the deliverable; each pass overwrites atomically.
- `reflection_log.jsonl` — append-only, one line per pass plus an outcome
  line. Deliberately the same shape as `review_log.jsonl` so the two can be
  read side by side:

```json
{"pass": 1, "answered": false, "reasoning": "...", "judged": true}
{"pass": 2, "answered": false, "reasoning": "...", "judged": true}
{"pass": 3, "answered": null, "reasoning": null, "judged": false}
{"outcome": "unjudged", "passes": 3, "report": "output/reflection_.../report.md"}
```

`outcome` is `"answered"` or `"unjudged"`. `judged: false` marks the capped
pass the judge never saw.

Error handling:

- **Structured-output failure.** Ollama may not honour `response_format`.
  The predicate replicates the framework's own fallback: look for
  `VERDICT: DONE` / `VERDICT: MORE` markers, with `MORE` winning when the
  reply is ambiguous or marker-less. Note this fails **open** (keep
  looping), the opposite direction from reflexion's fail-**closed** (reject
  an unparseable verdict). Both are correct for their pattern: reflexion
  must never ship unverified work as approved; reflection's cap is what
  guarantees termination, so an unreadable verdict simply costs a pass.

  Reaching that fallback requires a guard, and the guard is the point:
  `ChatResponse.value` is a **lazily-parsing property**. With
  `response_format` set it runs `JudgeVerdict.model_validate_json` on first
  access and raises `pydantic.ValidationError` for any non-empty reply that is
  not schema-conforming JSON — i.e. for exactly the reply the markers exist to
  read. Unguarded, that exception escapes `should_continue` →
  `_evaluate_stop` → `agent.run()` — nothing on that path catches it, and this
  demo's own `main.run()` catches only `KeyboardInterrupt` — and kills the run
  *before* `persist_fallback` and `log.finish`: no report,
  no outcome line, and the whole marker branch dead code. The predicate
  therefore reads `response.value` inside `try/except ValueError`
  (`ValidationError` subclasses `ValueError` in pydantic v2) and passes `None`
  to `read_verdict` on failure. The framework's own `_build_judge_condition`
  reads `response.value` unguarded, so a local model that ignores
  `response_format` crashes MAF's built-in judge loop too.
- **`write_report` never called.** Longest-assistant-text fallback only —
  the model often emits the full report as chat text and then answers a later
  nudge with filler, so the turn's *longest* text wins, not the latest. This
  is **not** the same as `reflexion_demo/graph.py:_draft`, which first issues a
  second `agent.run` with an explicit `_WRITE_REPORT_NUDGE` and only falls
  back if that also fails. This demo cannot do that: a second `agent.run`
  re-enters `AgentLoopMiddleware` and would consume a pass. Its worker is
  nudged only *between* passes, through the loop's `next_message` — so there is
  no delivery retry on the capped pass, and none at all under
  `--max-passes 1`. The live E2E runs `--max-passes 1` and lands on this path
  (`write_report never called - persisted the longest reply instead`).
- **Missing corpus / unreachable Ollama.** `ensure_corpus` and `preflight`
  copied from `reflexion_demo/main.py`, same fail-fast messages.
- CLI stays stdlib (`argparse` / `print`).

Console narration mirrors the other two demos: pass banners, each verdict
and its reasoning, the terminal outcome, artifact paths.

## 9. On duplicating `tools.py`

`tools.py`, `ensure_corpus`, `preflight`, `normalize_host` and
`model_present` are copies of the reflexion demo's. This is deliberate and
consistent with the rule that demo packages are standalone: each demo can be
read end to end, and lifted out of the repo, without tracing imports into a
sibling. The cost is drift between three copies of ~40 lines of preflight;
that is accepted. Should a fourth demo appear, revisit.

## 10. CLI

```bash
poetry run reflection                        # defaults: same topic as reflexion, 3 passes
poetry run reflection --topic "..."          # any topic over the same corpus
poetry run reflection --max-passes 2         # quickest tour of the unjudged ending
poetry run reflection --model gemma4:31b     # sets OLLAMA_MODEL
poetry run reflection --num-ctx 32768        # overrides OLLAMA_NUM_CTX
```

`--max-passes` rather than `--max-cycles`: a reflexion *cycle* is a draft
plus its review by a second agent; a reflection *pass* is one agent run. The
vocabularies are kept distinct on purpose.

## 11. Testing

LLM-free by default (`-m 'not ollama'`), decision logic in pure functions,
fakes over mocks — repo rules. No `tests/__init__.py`.

- `tests/test_reflection_judging.py` — the pure functions in `judging.py`:
  `read_verdict` (a structured `JudgeVerdict`; the `VERDICT: DONE` /
  `VERDICT: MORE` marker fallback; an ambiguous or marker-less reply continues
  the loop), `summarize`'s terminal arithmetic, and `RunLog`'s line shapes —
  `unjudged` at the cap, `answered` on early exit.
- `tests/test_reflection_loop.py` — the predicate itself: what it sends the
  judge (what the worker said *and* the report's text; no tool results, no
  reasoning traces, no non-text content at all) including the
  missing- and empty-report wording,
  the per-call `response_format` / `num_ctx` options, the guard for a
  lazily-raising `ChatResponse.value`, the `next_message` feedback relay, and
  the canary that **the judge is not called on the capped pass** (guards
  `_evaluate_stop`'s short-circuit against framework upgrades).
- `tests/test_reflection_integration.py` — the one LLM-free test that
  constructs the real thing: `build_agent`'s wiring driven through **two
  passes** of a real `Agent` + real `AgentLoopMiddleware` against a scripted
  `BaseChatClient`. Covers the judge being called, the feedback relay reaching
  pass 2, session accumulation, progress injection, `num_ctx` reaching both
  clients, and a non-conforming verdict. Without it a typo in an `Agent(...)`
  or `AgentLoopMiddleware(...)` keyword would surface only under
  `OLLAMA_E2E=1`.
- `tests/test_reflection_main.py` — CLI helpers, `ensure_corpus`, `preflight`
  host/model normalization, and the `persist_fallback` longest-text rule.
- `tests/test_reflection_prompts.py` — both templates render and carry their
  contracts (rubric words, verdict markers, delivery instruction).
- `tests/test_reflection_tools.py` — traversal guard, text-file filter,
  atomic report write, `ReportFlag` write detection.
- `tests/test_e2e_reflection.py` — one live smoke test
  (`@pytest.mark.ollama`, `OLLAMA_E2E=1`, `--max-passes 1`): asserts
  `report.md` and a coherent `reflection_log.jsonl` exist.
- Markdown lint gate extended to `src/reflection_demo/prompts/`
  (`.pymarkdown.json` scan set and the lint test's path list).

## 12. Out of scope (deliberate)

Checkpoint/resume, scratchpad steering, compaction, `ArtifactStore`/ledger
reuse, DevUI, and any comparison script — `diff` over the two `report.md`
files is the comparison, and a bespoke tool would be one more thing to
maintain for no insight. (The tool-call budget and mid-turn stripping this
section previously scoped out are no longer out of scope — see §6.)

Also out of scope: a rule-based (`should_continue` predicate over the draft
text, no second model) termination variant. It is the cheaper and more
reliable half of reflection in production, but it removes the interesting
failure — a critic that is confidently wrong because it cannot check — which
is the whole reason this demo sits next to the reflexion one.

## 13. References

- `AgentLoopMiddleware` — `agent_framework/_harness/_loop.py` (installed
  1.11.0); `JudgeVerdict`, `AgentLoopMiddleware.with_judge` are public
  exports of `agent_framework`.
- MAF samples read for this design:
  `02-agents/middleware/agent_loop_middleware_judge.py` (the `with_judge`
  one-liner), `02-agents/middleware/agent_loop_middleware_refinement.py`
  (`should_continue` + `record_feedback` + `fresh_context`),
  `05-end-to-end/evaluation/self_reflection/` (a self-reflection loop scored
  by a Foundry groundedness evaluator — and, notably, citing the *Reflexion*
  paper for a *self-reflection* sample; the naming confusion this demo
  exists to clarify is present in Microsoft's own repository).
- `docs/superpowers/specs/2026-07-17-reflexion-demo-design.md` — the demo
  this one is the foil to.
- `docs/superpowers/specs/2026-07-30-reflection-tool-budget-design.md` —
  reverses §6's original "no budget" call; the per-pass budget and countdown
  coaching design.
- Shinn et al., *Reflexion: Language Agents with Verbal Reinforcement
  Learning* (2023), <https://arxiv.org/abs/2303.11366>.
- Huang et al., *Large Language Models Cannot Self-Correct Reasoning Yet*
  (ICLR 2024), <https://arxiv.org/abs/2310.01798> — the empirical result
  this demo is expected to reproduce in miniature.
