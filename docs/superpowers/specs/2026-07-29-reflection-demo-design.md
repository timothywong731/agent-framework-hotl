# Reflection Demo — Design

**Date:** 2026-07-29
**Status:** Approved
**Stack:** Python ≥3.10, Poetry, pytest, Microsoft Agent Framework
(`agent-framework` ≥1.11, `agent-framework-ollama`), Ollama `gemma4:31b`
(local; same client conventions as the HOTL and reflexion demos)

## 1. Purpose

Standalone demo of the **reflection mechanism**, built as a deliberate A/B
foil to `reflexion_demo`: the same corpus, the same default topic, the same
worker tools, the same report artifact — with **exactly one variable
changed**, the critic's access to evidence.

- `reflexion_demo`: the critic is an `Agent` node in a cyclic graph holding
  `list_files` / `read_file` / `read_report`. It re-checks the draft against
  the sources and judges **what was written**.
- `reflection_demo`: the critic is a bare `OllamaChatClient` called directly
  by `AgentLoopMiddleware`, with no tools, no session, and no corpus. It
  judges **what the worker said**.

Running both on the same topic and diffing `report.md` is the demo. The
expected result: reflection converges quickly on structure and prose, and
misses the planted evidence conflicts that the grounded reviewer catches.

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
            OllamaChatClient(), judge_instructions, log, resolve_num_ctx()),
        max_iterations=args.max_passes,
        next_message=make_next_message(),
    )],
    default_options={"num_ctx": resolve_num_ctx()},
)
await agent.run(render_worker_prompt(topic=args.topic))
```

`fresh_context` stays at its default `False`: each pass accumulates the
prior conversation, which is precisely "grounded on its own chat history" —
the property that distinguishes reflection from reflexion's fresh
`AgentSession` per cycle.

The judge's `OllamaChatClient()` is bare — it bypasses `Agent` by design, so
`default_options` (an `Agent`-level mechanism) never reaches it; `num_ctx`
is instead pinned per call inside `make_judge_predicate`, the same
`options={...}` route `hotl_demo/compaction.py` uses for its bare client.

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

The judge's rubric is **identical to the reviewer's on Coverage and
Actionability**; Accuracy is necessarily weaker because the judge has no
corpus access and cannot verify that claims match their sources. If the two
critics were also given different standards on those two dimensions, the A/B
would confound two variables and prove nothing. This documented asymmetry in
Accuracy is the distinction the demo exists to demonstrate.

## 4. Information asymmetry

| | reflexion reviewer | reflection judge |
|---|---|---|
| What it is | `Agent` node in a cyclic graph | bare `OllamaChatClient`, called directly |
| Corpus tools | `list_files`, `read_file` | none |
| Report access | `read_report` — reads the file off disk | none — the transcript only |
| Session | fresh `AgentSession` per cycle | none; the loop replays the running transcript |
| Judges | what was **written** | what the worker **said** |

The framework states the first three rows itself, in the docstring of
`_build_judge_condition`: *"The judge is called directly (no agent tools,
session, or middleware)."*

The last row is the mechanism's sharp end. The reflexion reviewer opens
`report.md`; the reflection judge sees only the agent's response messages.
A worker that *claims* in conversation to have addressed the Azure mandate
can satisfy the judge without the report saying anything of the kind.

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
evaluating `should_continue` (`_harness/_loop.py:583`, so an expensive judge
is not called once the cap has fired). The consequence is behavioural and
must be documented, not hidden: **at the cap the judge is never consulted,
and the report ships unjudged.**

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
participant reads the report file in this demo): `make_corpus_tools` gives
the traversal-guarded `list_files` / `read_file` pair over `sample_data/`;
`make_report_tools` gives `write_report` plus the `ReportFlag` write-detection
cell. Same idioms — oversized reads truncated, failures returned as
`"ERROR: ..."` strings and never raised.

The worker's tool set is byte-for-byte the reflexion worker's. That is the
control in the experiment.

**No tool-call budget and no mid-turn stripping.** The strip mechanism is
already demonstrated by `reflexion_demo`; repeating it here would double the
package for no additional insight and force a copy of `budget.py` to honour
the standalone rule. `worker.md` instructs the model to explore economically
instead; `max_iterations` is the only bound.

## 7. Prompts

Jinja2 markdown under `src/reflection_demo/prompts/`, no YAML frontmatter
(no phase discovery here), joining the markdown lint gate.

- `worker.md` — one template, one variant. Unlike the reflexion worker there
  is no `revision` or `finalize` mode: the loop middleware injects the
  judge's feedback as the next iteration's input, so the prompt never has to
  be re-rendered. Instructs: explore the corpus with tools, ground every
  claim in a source file, deliver via `write_report`, explore economically.
- `judge.md` — the judge's system instructions. Carries the same
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
- **`write_report` never called.** Same nudge-then-longest-text fallback as
  `reflexion_demo/graph.py:_draft` — the model often emits the full report
  as chat text and answers the nudge with filler, so the turn's *longest*
  text wins, not the latest.
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

- `tests/test_reflection_judge.py` — the predicate: a structured
  `JudgeVerdict` parses; the `VERDICT: DONE` / `VERDICT: MORE` marker
  fallback; an ambiguous or marker-less reply continues the loop; every
  verdict reaches the log with the right `pass` number.
- `tests/test_reflection_loop.py` — **the judge is not called on the capped
  pass** (guards the `_loop.py:583` short-circuit against framework
  upgrades); the log's terminal line is `unjudged` at the cap and
  `answered` on early exit.
- `tests/test_reflection_tools.py` — traversal guard, text-file filter,
  atomic report write, `ReportFlag` write detection.
- `tests/test_e2e_reflection.py` — one live smoke test
  (`@pytest.mark.ollama`, `OLLAMA_E2E=1`, `--max-passes 1`): asserts
  `report.md` and a coherent `reflection_log.jsonl` exist.
- Markdown lint gate extended to `src/reflection_demo/prompts/`
  (`.pymarkdown.json` scan set and the lint test's path list).

## 12. Out of scope (deliberate)

Per-turn tool budget and mid-turn stripping (§6), checkpoint/resume,
scratchpad steering, compaction, `ArtifactStore`/ledger reuse, DevUI, and any
comparison script — `diff` over the two `report.md` files is the comparison,
and a bespoke tool would be one more thing to maintain for no insight.

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
- Shinn et al., *Reflexion: Language Agents with Verbal Reinforcement
  Learning* (2023), <https://arxiv.org/abs/2303.11366>.
- Huang et al., *Large Language Models Cannot Self-Correct Reasoning Yet*
  (ICLR 2024), <https://arxiv.org/abs/2310.01798> — the empirical result
  this demo is expected to reproduce in miniature.
