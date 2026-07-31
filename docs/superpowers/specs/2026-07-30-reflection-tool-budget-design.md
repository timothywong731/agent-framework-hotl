# Reflection Tool Budget and Countdown Coaching — Design

**Date:** 2026-07-30
**Status:** Approved
**Stack:** Python ≥3.10, Poetry, pytest, Microsoft Agent Framework
(`agent-framework` ~=1.11), Ollama `gemma4:31b`
**Touches:** `src/reflection_demo/` (gains a budget) and
`src/reflexion_demo/` (nudge wording replaced to match)

## 1. Purpose

Two changes, one spec, because they cannot be separated without breaking the
A/B experiment the two demos exist to run.

1. **`reflection_demo` gains a per-pass tool-call budget** with tool
   stripping and a forced-finalize pass, so the worker always produces a
   report within a bounded amount of exploration.
2. **Both demos gain anticipatory countdown coaching** in place of the
   single punitive nudge `reflexion_demo` fires after the fact.

### Why the budget, having previously declined it

The reflection demo's design spec §6 explicitly dropped the tool budget:
*"The strip mechanism is already demonstrated by `reflexion_demo`; repeating
it here would double the package for no additional insight."* That reasoning
is now known to be wrong on both counts:

- **It removes a confound.** The final whole-branch review flagged the
  missing budget as a *second uncontrolled asymmetry*: the reflexion worker
  runs under 12 calls with mid-turn stripping, the reflection worker is
  unbounded. The A/B therefore differs in two variables, not one. Adding the
  budget is what makes "exactly one variable changed" true.
- **It fixes an observed defect.** The live `--max-passes 1` E2E hit
  `write_report never called - persisted the longest reply instead`: the
  worker spent its only pass exploring and never delivered. Only the
  longest-reply fallback saved the run.

### Why both demos, not just one

If only the reflection worker receives the improved coaching, the two
workers are coached differently, so they gather evidence differently — for a
reason unrelated to the critic. That is a new confound of exactly the kind
this change exists to remove. The countdown wording and the budget
mechanics are therefore **byte-identical across both packages**, enforced by
a test (§9).

## 2. The countdown

One shared set of constants, duplicated verbatim into both
`budget.py` modules. Framing is deliberately *positive and anticipatory*:
the closing message tells the worker it already has what it needs, rather
than that it has been cut off.

```python
COUNTDOWN = {
    3: "3 tool calls left. Decide now which gaps matter most and spend them there.",
    2: "2 tool calls left.",
    1: "1 tool call left - your last. Spend it on the single most important gap, then write.",
}
BUDGET_SPENT = ("Exploration is closed. You have what you need - write the "
                "complete report now with write_report.")
```

Each message is prefixed `[budget] ` when appended, matching the existing
`[SYSTEM] ` convention's purpose (marking machine-authored text inside a
tool result) without reusing a prefix that reads as an error.

Silent for calls 1–9 of a 12-call budget. Three calls of runway is the point:
at one call remaining a model can only pick a single action, so "use it
wisely" would be a notification rather than something it can act on.

## 3. Budget lifetime

`reflexion_demo` gets a per-turn budget for free — each of its cycles is a
separate `agent.run()`, so `main.py` mints a fresh `ToolBudget` per turn.
`reflection_demo` has no such boundary: `AgentLoopMiddleware` drives every
pass inside a **single** `agent.run()`.

The budget is nonetheless **per pass**, not per run. Per-run would hand the
two workers different total allowances (reflexion: 12 × cycles; reflection:
12 total) and undo the symmetry this change exists to establish.

Verified against the installed framework: **tools re-arm on every pass**
inside one `agent.run()` — a probe stripping `peek` during pass 1 saw it
callable again in pass 2. So a strip does not leak across passes, and
resetting the counter is the only thing needed.

`next_message` is the reset hook. It fires exactly once between passes and
already needs the pass number for the finalize decision (§4), so both
pass-boundary concerns live in one place.

Pass 1 has no boundary before it and is **not** primed: `PassBudget`'s
defaults (`spent=0`, `finalizing=False`) are already pass 1's state, so
`build_agent` constructs the budget and calls nothing on it. That holds at
`--max-passes 1` too. Priming that lone pass `finalizing=True` would close
exploration on its **first** budgeted call, seconds after `worker.md` promised
`max_tool_calls` of them — and since `next_message` never fires on a
single-pass run, `finalize.md`, the only text that explains the strip, could
not be delivered either. Delivery is forced without it: at exhaustion the read
tools are stripped and `BUDGET_SPENT` says why.

```python
@dataclass
class PassBudget:
    """Per-pass tool budget for a run. ``start_pass`` at each boundary.

    One instance per run, mutated in place. Single-threaded by construction:
    AgentLoopMiddleware drives passes sequentially inside one agent.run().
    """

    max_calls: int
    spent: int = 0
    finalizing: bool = False

    @property
    def remaining(self) -> int:
        return max(0, self.max_calls - self.spent)

    def start_pass(self, *, finalizing: bool) -> None:
        self.spent = 0
        self.finalizing = finalizing
```

Owned by `budget.py`; two consumers — the function middleware reads it,
`next_message` advances it.

## 4. The finalize pass

`reflexion_demo`'s forced finalize *constructs* a new agent without read
tools. `reflection_demo` cannot: the loop owns the agent, and
`FunctionInvocationContext.remove_tools()` is only reachable from inside a
tool call, so tools cannot be removed pre-emptively.

The finalize instruction therefore arrives as the final pass's **input**:

```text
after pass N-1   next_message sees iteration == max_passes - 1
                 -> renders prompts/finalize.md
                 -> budget.start_pass(finalizing=True)
otherwise        relays the judge's reasoning
                 -> budget.start_pass(finalizing=False)
```

On a finalizing pass the middleware strips the read tools after the
**first** tool call rather than at the budget. A worker that ignores the
finalize instruction therefore gets exactly one exploratory call and then
the closing nudge — the closest achievable mirror of reflexion's
construction-time strip, and it degrades honestly rather than silently.

`iteration` is the count of completed passes, so `iteration == max_passes - 1`
identifies the boundary into the last pass. `next_message` is a closure over
`max_passes`.

## 5. Middleware rules

Precise behaviour after each **budgeted** tool call completes (the call
always executes first — the budget bounds exploration, it never discards a
result already paid for):

1. `budget.spent += 1` — always, before any guard, so a straggler still
   counts. `spent` may therefore exceed `max_calls`; `remaining` clamps at 0.
2. **Strip guard.** When `context.tools` is a list *and* none of the budgeted
   names is present in it, return — this pass has already been stripped and
   the call is an in-flight straggler from the same batch. Prevents a double
   nudge. When `context.tools` is `None` (the call was made outside a
   function-calling loop, so there is no live list to inspect or mutate) the
   guard does not apply: skip the removal but still append the message, so the
   model is told why regardless. Same guard and same reason as
   `reflexion_demo/budget.py`.
3. **Finalizing pass:** strip the read tools, append `BUDGET_SPENT`, return.
   This dominates — it fires on the first call regardless of `remaining`.
4. **Budget spent** (`remaining == 0`): strip the read tools, append
   `BUDGET_SPENT`, return.
5. **Countdown** (`remaining` in `COUNTDOWN`): append `COUNTDOWN[remaining]`.
6. Otherwise: nothing.

`write_report` is exempt from counting and from stripping — it is delivery,
not exploration.

Budgeted names differ between the demos, and correctly so:

| | Worker | Critic |
|---|---|---|
| `reflexion_demo` | `list_files`, `read_file` | `list_files`, `read_file`, `read_report` |
| `reflection_demo` | `list_files`, `read_file` | none — the judge holds no tools |

The critic column is the demo's intended asymmetry, not a confound. The
worker column is now identical, which is the point of this change.

## 6. Changes to `reflexion_demo`

Its worker is already budgeted; only the coaching changes.

- `budget.py`: `BUDGET_NUDGE` replaced by `COUNTDOWN` + `BUDGET_SPENT`;
  countdown appended at 3/2/1 remaining. `ToolBudget` keeps its current shape
  (per-turn instances make `PassBudget`'s `start_pass` unnecessary there).
- `tests/test_reflexion_budget.py` asserts on the exact `BUDGET_NUDGE`
  string and must be updated. Its structural assertions (counting, exemption,
  strip-once-per-run, re-arm across runs, `None` tools, `None` result) all
  stay — only the wording assertions change, plus new countdown cases.
- `prompts/worker.md`: the initial and revision variants carry the **shared
  budget paragraph** of §7 — byte for byte, `{{ max_tool_calls }}` number
  included. The finalize variant carries none: that agent is constructed with
  `write_report` only, so it has no budget to spend.
- `prompting.py` / `graph.py` / `main.py`: `max_tool_calls` is threaded into
  `render_worker_prompt` through `WorkerExecutor`, which already carries
  `max_cycles`. Required keyword, not defaulted — a plausible-looking default
  would render a silently wrong number into the prompt.
- Its design spec §5 and the README's reflexion section: document the
  countdown.
- Its live E2E is worth re-running — the worker's coaching changed.

## 7. Prompts

- **New `reflection_demo/prompts/finalize.md`** — Jinja2, rendered by
  `render_finalize_message(*, topic, max_passes)`. Instructs: this is the
  final pass, there will be no further review, write the complete report now
  from what you already have, deliver with `write_report`.
- **`reflection_demo/prompts/worker.md`** — the line "There is no tool
  budget here, but a bloated transcript crowds out the report" is now false
  and is replaced by the shared budget paragraph below.
- **The budget paragraph is shared, byte for byte.** Both worker templates —
  `reflection_demo/prompts/worker.md`, and the initial *and* revision variants
  of `reflexion_demo/prompts/worker.md` — carry exactly this:

```jinja
You have {{ max_tool_calls }} tool calls. The last few are announced in the
tool results, and when the budget is spent your exploration tools close and
you write from what you have. Spend them on the gaps that matter.
```

  Both workers are told the **number**. §2's rationale — coaching one worker
  better than the other makes evidence-gathering differ for a reason unrelated
  to the critic — binds the prompts as hard as it binds the runtime nudges: a
  worker that knows it has 12 calls can plan a 12-file sweep, and one told only
  "a limited number" cannot. An earlier draft of this spec prescribed the
  number for reflection (§7) and merely "state that the budget counts down"
  for reflexion (§6), which contradicted §2 and produced exactly that
  confound.

  The paragraph names **no unit** — not "per pass", not "per turn". The scope
  word is the one part that legitimately differs between the demos
  (`reflection_demo` budgets a pass, `reflexion_demo` a turn), and a shared
  paragraph cannot carry both without conflating the two vocabularies. Each
  prompt states its own unit in its own closing paragraph instead
  ("at most `{{ max_passes }}` passes" / "cycle `{{ cycle }}` of at most
  `{{ max_cycles }}` review cycles").
- **`reflection_demo/prompting.py`** — its module docstring says *"Unlike the
  reflexion worker there is no revision or finalize variant"*. Half of that
  becomes false: there is now a finalize message (though still no revision
  variant — the loop injects the judge's feedback). Correct it.

Both prompts directories stay under the markdown lint gate.

## 8. CLI

`reflection_demo` gains one flag, named and defaulted to match its sibling:

```bash
poetry run reflection --max-tool-calls 12   # default, per pass
poetry run reflection --max-tool-calls 4    # quickest tour of the countdown
```

Validated `>= 1`, consistent with `--max-passes`. `--max-tool-calls 1` means
the first call immediately strips: no countdown line is reachable, which is
correct rather than a special case.

## 9. Testing

LLM-free by default, `FakeInvocationContext` in the established style of
`tests/test_reflexion_budget.py` (duck-typed, `tools` a shared live list per
run, `remove_tools` mutating in place).

`tests/test_reflection_budget.py`:

- countdown fires at exactly 3, 2 and 1 remaining — once each, with the
  right text
- no countdown line before 3 remaining
- `write_report` neither counts nor is stripped
- `remaining == 0` strips and appends `BUDGET_SPENT`
- a finalizing pass strips after the **first** call, whatever `remaining` is
- the strip guard suppresses a second nudge for in-flight stragglers
- `start_pass` resets `spent` and sets `finalizing`
- `--max-tool-calls 1`: first call strips, no countdown text

`tests/test_reflection_loop.py` (extended):

- `next_message` renders the finalize message on the boundary into the last
  pass and the judge's feedback otherwise
- `next_message` calls `start_pass`, with `finalizing` true only on that
  boundary

`tests/test_reflection_main.py` (extended):

- `build_agent` leaves pass 1's budget non-finalizing, at `--max-passes 1`
  and above — the one decision-bearing branch in this change that no test
  reached, and the one a prompt/middleware disagreement hides behind

`tests/test_reflection_prompts.py` (extended):

- `finalize.md` states that a read tool on the final pass closes exploration
  immediately afterwards — the only prose in either package describing the
  finalize strip, so the only guard against it drifting from the middleware

Both `budget.py` test files additionally cover the **countdown** branch with a
`list[Content]` result, not just the closing branch: both append through the
same `_append_note`, and the countdown fires up to three times a pass, so it
was where the repr-mangling defect did the most damage.

`tests/test_reflexion_budget.py` (updated): wording assertions moved to the
new constants; new countdown cases; all existing structural cases retained.

**Cross-demo drift guard** — `tests/test_budget_wording_parity.py`:

```python
def test_countdown_and_spent_wording_identical_across_demos():
    from reflection_demo import budget as reflection_budget
    from reflexion_demo import budget as reflexion_budget
    assert reflection_budget.COUNTDOWN == reflexion_budget.COUNTDOWN
    assert reflection_budget.BUDGET_SPENT == reflexion_budget.BUDGET_SPENT
```

Constants are not enough on their own: the two prompts drifted apart while
these assertions stayed green. The same file therefore renders both worker
prompts with the same `max_tool_calls` and compares the extracted **budget
paragraph** (§7), for the initial and revision variants, and asserts the
reflexion finalize variant has none.

Drift is the only real failure mode of the deliberate duplication, and this
is the cheapest possible guard against it. A test importing both packages
does not violate the standalone rule — that rule binds package code, and
there is precedent: `tests/test_reflection_main.py` already imports
`reflexion_demo.main.DEFAULT_TOPIC` to pin topic identity.

## 10. Effect on the documented A/B caveats

The README's A/B recipe currently lists the unbounded reflection worker as an
acknowledged asymmetry, with its direction of bias. **That entry is removed**
— the workers become identically bounded and identically coached. The
delivery-retry caveat stays (reflexion nudges once more before falling back;
reflection cannot, because a second `agent.run()` would re-enter the loop and
burn a pass).

A **second** worker-side residue arrives with this change and must be
documented alongside it: on the last pass/cycle reflexion's construction-time
strip gives its worker **zero** exploratory calls, while reflection's
finalizing pass can only strip from inside a tool call and therefore concedes
**one** (§4). Both residues follow from the two demos being separate packages
built on different framework primitives, not from the experiment.

So: the worker's tool set, tool budget and coaching become identical here, and
the two named residues are what "exactly one variable changed" is qualified
by. No document may claim the worker sides differ in *nothing but* the
delivery retry.

## 11. Out of scope (deliberate)

- **A separate `budget.py` shared between demos.** The standalone rule holds:
  each demo reads end to end without tracing imports into a sibling. The
  duplication is accepted and guarded by §9's parity test.
- **Compaction on the worker agent.** The session change made the worker
  transcript accumulate, and there is no `compaction_strategy`, so a 3-pass
  corpus run at the default `num_ctx=4096` may still truncate. The budget
  reduces the pressure but does not remove it. Left as a known limitation,
  already noted in the README gotchas.
- **Changing `--max-passes`, the judge, the verdict schema, or anything in
  `judging.py` beyond `next_message`'s new pass-boundary duties.**
- **Reconciling the unreproduced A/B prediction.** The reflection demo's spec
  §1 predicts that reflection "misses the planted evidence conflicts"; the
  live 3-pass run instead saw the worker surface the Azure/S3 conflict
  unprompted and the judge approve on pass 1. That prediction is a hypothesis
  rather than a result, and correcting the claim is its own piece of work.

## 12. References

- `docs/superpowers/specs/2026-07-29-reflection-demo-design.md` — §6 dropped
  the budget; this spec reverses that with reasons.
- `docs/superpowers/specs/2026-07-17-reflexion-demo-design.md` §5 — the
  two-budget design and tool stripping this mirrors.
- `src/reflexion_demo/budget.py` — the middleware whose structure is copied,
  including the strip guard and its rationale.
- MAF sample `02-agents/middleware/function_based_middleware.py` and the
  `remove_tools` progressive-tool-exposure API.
