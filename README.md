# agent-framework-hotl

Human-on-the-loop (HOTL) demo on
[Microsoft Agent Framework](https://github.com/microsoft/agent-framework):
a multi-phase agent pipeline that assesses a fictional legacy system's cloud
migration readiness, accumulates a **ledger of questions** needing human
adjudication, pauses **exactly once** at a review gate, selectively re-runs
affected phases with the human's answers, and accepts freeform steering via a
**scratchpad** file - read at phase start, and pushed to agents mid-run when
the human edits it while the pipeline is working.

The repo also carries a second, standalone demo of the **reflexion**
pattern - a worker/reviewer loop with information parity and budget-forced
convergence - see [The reflexion demo](#the-reflexion-demo).

Design specs:
`docs/superpowers/specs/2026-07-14-hotl-pipeline-design.md` (pipeline),
`docs/superpowers/specs/2026-07-15-live-scratchpad-steering-design.md`
(live steering), and
`docs/superpowers/specs/2026-07-17-reflexion-demo-design.md` (reflexion)

## The pipeline

```mermaid
flowchart LR
    D[discovery]
    subgraph DA [deep_analysis - parallel, explores repos via tools]
        A1[analyze:oms-monolith]
        A2[analyze:oms-batch-recon]
    end
    J[join]
    EC[enterprise_context]
    Q[questionnaire]
    R{{REVIEW<br>human gate}}
    F[final_report]

    D --> A1 & A2
    A1 & A2 --> J
    J --> EC --> Q --> R --> F
    R -. "RevisionTrigger (per answered phase, sequential)" .-> D & A1 & A2 & EC & Q
    D & A1 & A2 & EC & Q -. RevisionDone .-> R
```

- **discovery** - what does this system REALLY do (docs vs code)?
- **deep_analysis** - one agent per repo, in parallel; explores its repo
  agentically via `list_files`/`read_file` tools; per-repo reports
- **enterprise_context** - corporate cloud strategy + security standards
  overlay
- **questionnaire** - fills the standard readiness question template
- **review** - the human gate: answer (authoritative) or decline (default
  assumption applies)
- **final_report** - readiness scorecard + recommendation + adjudication log

## The HOTL model

"Human-**on**-the-loop" rather than "in-the-loop": the pipeline runs to
completion on its own and **never blocks waiting for a human**. When an agent
hits a gap or a contradiction it records the question, states the assumption
it will proceed on, and carries on. The human supervises rather than
authorises.

There are exactly two channels from human to pipeline, and they are
deliberately different shapes:

| | Channel 1: question ledger | Channel 2: scratchpad |
|---|---|---|
| Shape | Structured questions with defaults | Freeform markdown |
| Who initiates | The **agent** (`raise_question`) | The **human** |
| When | Once, after `questionnaire` | Any time - before or during a run |
| How often | **Exactly once per run** | Unlimited |
| Blocks the run? | Yes - the workflow idles at the gate | No - never blocks |
| Effect of input | Answer re-runs the raising phase | Advisory; the agent judges relevance |
| Reaches finished phases? | Yes, via targeted re-run | No - only agents still working |

The asymmetry is the point. Interrupting a human is expensive, so the
structured channel is rationed to one batch. Steering is cheap, so the
freeform channel is unlimited.

## Channel 1: the question ledger

### Raising a question never blocks

Phase prompts instruct agents: when evidence conflicts or a decision-critical
fact is missing, call `raise_question` with the assumption you will proceed
on, then **proceed on it**. The tool returns the assigned id and says so
explicitly:

```text
Recorded q-1. Proceed using your stated default assumption.
```

That default is what makes the pipeline closed by construction. A run with
zero human input still produces a complete report - just one built on stated
assumptions rather than adjudicated facts.

### The ledger entry

`ledger.jsonl` is append-only, one JSON object per line:

```json
{
  "id": "q-3",
  "phase": "deep_analysis",
  "unit": "oms-batch-recon",
  "question": "Remediate the hardcoded DB credentials before or during migration?",
  "context": "config.py holds a plaintext password; the security standard forbids credentials in code.",
  "impact": "If remediation must precede migration it adds a pre-cutover security workstream; if it can happen during, it folds into the cutover runbook.",
  "importance": "medium",
  "default_assumption": "Move to a vault and rotate before migration",
  "status": "open | answered | declined | deferred",
  "human_answer": null,
  "asked_at": "2026-07-15T20:41:07"
}
```

`phase` and `unit` are **not supplied by the model**. The tools are
closure-bound to the calling `(phase, unit)` in `tools.py`, so an agent cannot
forge attribution or write into another repo's section. `unit` is the repo
name for `deep_analysis` questions and `null` everywhere else - that pair is
exactly what the gate later re-runs.

`importance` (high / medium / low) and `impact` - how the human's answer
would change the migration decision - are agent-declared, tool-validated,
and exist for the slot competition below.

### Accumulation and duplicate suppression

The ledger grows across phases, and every phase's prompt is rendered with the
**current open ledger** plus an instruction not to re-raise anything already
there. Duplicate suppression is therefore prompt-level, not structural - no
curation machinery, no dedupe pass. The trade is that suppression is only as
reliable as the model.

### Lifecycle of a question

```mermaid
flowchart LR
    A["agent hits a gap<br>or a contradiction"] -->|"raise_question"| O(["status: open"])
    O -->|"human types an answer"| AN(["status: answered"])
    O -->|"human presses ENTER"| DE(["status: declined"])
    O -->|"raised during a re-run<br>(never prompted)"| OP["final report:<br>open - default applied"]
    O -->|"loses the slot competition"| DEF["status: deferred<br>default applied, terminal"]
    AN --> RR["raising phase RE-RUNS<br>answer is AUTHORITATIVE"]
    DE --> DF["no re-run<br>default assumption stands"]
```

## The review gate

The pause/resume is the framework's native `request_info` mechanism - the
workflow genuinely idles while the human decides:

```mermaid
sequenceDiagram
    participant H as Human
    participant CLI as CLI runner
    participant W as Workflow
    participant R as review executor

    W->>R: PhaseDone (questionnaire finished)
    R->>R: latch review_completed (runs once per pipeline)
    R->>W: request_info x N open questions
    Note over W: workflow idles
    W-->>CLI: request_info events
    loop each question
        CLI->>H: question + evidence + default assumption
        H-->>CLI: answer text, or ENTER to decline
    end
    CLI->>W: run(responses={...})
    W->>R: response per question
    R->>W: RevisionTrigger per answered phase (sequential, pipeline order)
    W->>R: RevisionDone per re-run phase
    R->>W: ReportTrigger
    W-->>CLI: final_report.md path
```

The rules the gate enforces:

1. **It runs exactly once.** Entering the gate latches `review_completed` in
   `memory.json`. This is the whole point of the demo: an agent pipeline that
   asks for help *once*, in one batch, instead of nagging.
2. **Slots are scarce - questions compete.** The gate presents at most
   `--max-questions` (default 3). When more are open, one LLM ranking call
   orders them by expected swing on the final report - judging each
   question's substance and impact statement, with the declared importance
   as one input; raise order carries no signal. Losers are marked
   `deferred`: their defaults stand and the adjudication log says so. The
   ranker is fenced (validated output, one retry, deterministic fallback),
   and `--max-questions 0` defers everything - the fully autonomous run.
3. **Answer = authoritative.** Typed text is injected into the re-run prompt
   marked `AUTHORITATIVE`, overriding any conflicting document or code
   evidence.
4. **Decline = the default stands.** Empty input (or a closed stdin) costs
   nothing and triggers no re-run. The stated assumption was already applied,
   so the report is already consistent with it.
5. **Only answered phases re-run**, and only the exact `(phase, unit)` that
   raised the question - answering a `oms-batch-recon` question re-runs that
   one analyzer, not its sibling. Re-runs are sequential, in pipeline order.
6. **Questions raised during a re-run are never prompted.** They append to the
   ledger and the final report lists them as *open - default assumption
   applied*. Otherwise "review once" would be a lie.

### A deliberate simplification

Re-runs do **not** cascade downstream: an answered `discovery` question
re-runs `discovery` alone. Downstream phases change only if they raised
answered questions themselves. The verdict still absorbs every answer, because
`final_report` synthesises from post-adjudication `memory.json`. Marked in the
code with a `ponytail:` comment; the upgrade path is to cascade from the
earliest affected phase.

### The adjudication log is not written by the LLM

`final_report.md` ends with a table rendered deterministically from
`ledger.jsonl` by plain code - answered questions with the human's verbatim
answer, declined and still-open ones with the default that was applied. The
narrative above it is the model's; the decision record is not, so it cannot be
hallucinated.

## Channel 2: the scratchpad

`scratchpad.md` (repo root, stable path, starts empty) is the freeform
channel. Write guidance into it at any time - before a run, or while one is
executing:

```markdown
The nightly reconciliation job is business-critical - treat it as a
first-class migration workload, not a peripheral batch process.

Data residency: assume EU (Ireland). Do not raise further questions on it.
```

It reaches agents two ways:

- **Pulled**, once per phase: every phase agent calls the `read_scratchpad`
  tool before starting work.
- **Pushed**, thereafter: if you edit the file *while* a phase is running, the
  change is delivered to every agent still working, arriving at that agent's
  next tool call. Each agent decides for itself whether the guidance is
  relevant to its phase.

Watch for the push in the run output:

```text
  [steering] scratchpad update queued for analyze:oms-monolith
```

### How the push works

`ScratchpadWatch` (one per agent - the analyzers run concurrently, so a shared
watermark would let one steal the other's notification) is polled by function
middleware after every tool call. A content change - not an mtime change, so
re-saving an unedited file is silent - is handed to the framework's
`MessageInjectionMiddleware`, which drains it into that agent's next model
call.

An LLM turn cannot be interrupted, so a tool call is the earliest possible
delivery point. That is why there is no file watcher: delivery would wait for
the same boundary anyway, so a watcher thread would buy nothing.

One useful property comes free: if an agent was about to finish and emit its
report, `MessageInjectionMiddleware` forces one extra model call rather than
letting the notice miss the boat.

### Limits worth knowing

- **It does not re-run finished phases.** An edit during `final_report`
  changes nothing upstream - that is the review gate's job.
- **The agent may reasonably ignore it.** The notice explicitly says to ignore
  irrelevant guidance, and it will. Steering that is on-topic for the phase
  lands; arbitrary instructions may not.
- **Clearing the scratchpad notifies nobody.** Withdrawing guidance is not new
  guidance to act on.

## Bounded context (compaction)

Local models have small windows (Ollama defaults to 4096 tokens) and, when a
conversation exceeds one, Ollama silently drops the OLDEST tokens - the system
prompt and task framing - producing off-task reports with no error anywhere.

Every phase agent therefore runs a compaction pipeline on each model call,
including between tool calls: old tool-call groups are evicted first
(newest 2 kept verbatim), the remainder is LLM-summarized if still over
budget, and a deterministic oldest-first fallback makes the budget a hard
guarantee. The budget is `0.8 * (num_ctx - 1024)`. Watch for:

```text
deep_analysis:oms-monolith: compacted context 14 -> 6 messages (~3124 -> 1893 tokens)
```

Size the window with `--num-ctx` (or `OLLAMA_NUM_CTX`); the same value is sent
to Ollama as `num_ctx`, so the server window and the compaction budget always
agree. Durable findings survive compaction by design: they live in
`memory.json` and the report drafts, not in chat history.

## Artifacts

```mermaid
flowchart TB
    subgraph AGENT [every phase agent]
        T1[read_scratchpad]
        T2[raise_question]
        T3[update_memory]
    end
    subgraph ANALYZER [deep_analysis only]
        T4[list_files / read_file]
    end
    SP[(scratchpad.md<br>human steering)] --> T1
    SP -. "mid-run edit:<br>pushed at next tool call" .-> AGENT
    T2 --> L[(ledger.jsonl<br>append-only questions)]
    T3 --> M[(memory.json<br>shared long-term memory)]
    T4 --> REPOS[(sample_data/repos)]
    L --> R{{REVIEW}}
    M --> FR[final_report.md]
    R --> FR
```

Everything lands in `output/run_<timestamp>/`: one markdown report per phase
(overwritten on revision), `memory.json`, `ledger.jsonl`, and
`final_report.md`. `scratchpad.md` deliberately lives at the repo root
instead: it is an input the human owns, not a run artifact.

Every mutation goes through `ArtifactStore` (one lock, atomic temp-file +
`os.replace`), because the two analyzers write concurrently and a human may
have the files open mid-run.

## The reflexion demo

`poetry run reflexion` is a separate, self-contained demo in the same repo:
no imports from `hotl_demo`, only `sample_data/` and the local Ollama model
are shared. A **worker** agent researches a migration topic and writes a
report; a **reviewer** agent - holding the *same* read tools over the *same*
corpus - verifies that report against the sources itself and returns a
boolean verdict. Rejection feedback steers the next draft. Two budgets
guarantee the loop always lands on a report.

```mermaid
flowchart LR
    T(["--topic"]) --> W
    W["worker<br>explores corpus with tools,<br>delivers via write_report"]
    R["reviewer<br>read_report + re-checks the<br>same corpus independently"]
    W -->|DraftReady| R
    R -->|"ReviewVerdict<br>(approved, feedback)"| V{verdict?}
    V -->|approved| OK(["report.md ships - approved"])
    V -->|"rejected, cycles left:<br>feedback steers the redraft"| W
    V -->|"rejected at --max-cycles"| FF["forced finalize<br>read tools stripped at construction:<br>write_report only"]
    FF --> UN(["report.md ships - unapproved<br>(review_log.jsonl: forced: true)"])
```

The graph is the framework's native cyclic-workflow shape (the same
`WorkflowBuilder` mechanics as the HOTL pipeline, two nodes and two message
types instead of seven): `worker -> reviewer -> worker`, with the worker
yielding the terminal output.

### Information parity

The reviewer never takes the author's word for anything - and it can't be
fobbed off, because it holds the identical corpus binding:

| Tool | Worker | Reviewer | Counts against the budget |
|---|---|---|---|
| `list_files` / `read_file` (corpus) | yes | yes | yes |
| `write_report` | yes | - | no - delivery, not exploration |
| `read_report` | - | yes | yes |

Parity of information, asymmetry of authority: both agents see the same
sources; only the worker can write the report, only the reviewer can approve
it. The reviewer judges what was actually **written** (`read_report`), not
what the worker claims in conversation.

### Two budgets, one ending

Both budgets finish the same way: tools are taken away and the agent is told
to produce from what it already has.

- **Per-turn tool budget** (`--max-tool-calls`, default 12). Function
  middleware counts read-tool calls; the call that exhausts the budget still
  executes, then the read tools are stripped for the rest of the turn (the
  framework's live `remove_tools()` mutation point) and the nudge rides on
  that tool's result: *"you have been reasoning for a long time - produce
  the report now from the information you already have."* `write_report`
  survives the strip. Applies to worker and reviewer alike.
- **Review-cycle budget** (`--max-cycles`, default 3). A rejection at the
  last cycle triggers one final worker turn whose agent is *constructed*
  without read tools - then the report ships regardless, logged as forced.

```mermaid
sequenceDiagram
    participant W as worker turn
    participant B as budget middleware
    participant R as reviewer turn

    Note over W: fresh agent + fresh budget every turn
    W->>W: list_files / read_file ... (counted)
    B-->>W: exhausting call: read tools stripped,<br>"produce the report from what you have"
    W->>W: write_report (exempt)
    W->>R: DraftReady
    Note over R: same corpus tools, same budget rules
    R->>R: read_report, spot-check claims against sources
    R->>R: structured-verdict turn (approved + feedback)
    R->>W: ReviewVerdict
```

The reviewer's turn is deliberately two calls in one session: an exploration
call with tools, then a schema-forced verdict call
(`{"approved": bool, "feedback": str}`). A verdict that fails to parse gets
one retry and then **fails closed** - an unverifiable draft is rejected,
never shipped as approved.

### Run the reflexion demo

```bash
poetry run reflexion                                    # defaults: 3 cycles, 12 tool calls/turn
poetry run reflexion --max-cycles 2 --max-tool-calls 8  # quickest full tour of both budgets
poetry run reflexion --topic "Assess retiring IBM MQ"   # any topic over the same corpus
```

The default topic (assess migrating OMS file storage from NFS to S3) is
chosen so the planted corpus conflicts - the Azure mandate vs `s3_uploader.py`,
data-residency and secrets standards - give the reviewer real grounds to
reject a shallow first draft. Watch the console for the mechanism firing:

```text
  [worker] tool budget exhausted (12 calls) - read tools stripped
  [reviewer] cycle 1: REJECTED
  [reviewer] feedback: The report ignores the enterprise Azure mandate...
== review budget exhausted: forced finalize (read tools stripped) ==
```

Artifacts land in `output/reflexion_<timestamp>/`: `report.md` (each
revision overwrites atomically) and `review_log.jsonl` - one line per cycle
plus an outcome line, written by plain code, never by the model:

```json
{"cycle": 1, "approved": false, "feedback": "...", "forced": false,
 "worker_tool_calls": 12, "reviewer_tool_calls": 9}
```

## Prerequisites

- Python 3.10+ and [Poetry](https://python-poetry.org/)
- [Ollama](https://ollama.com/) running locally with the model pulled:

```bash
ollama pull gemma4:31b
```

## Run the demo

```bash
poetry install
poetry run demo    # or: poetry run demo --model <other-tools-capable-model>
```

The four phases run autonomously (the two repo analyzers in parallel), each
writing a markdown report, updating `memory.json`, and appending questions to
`ledger.jsonl`. Then the review gate presents every open question:

```text
[q-1] (discovery) Is reconciliation functionality in migration scope?
      Evidence: oms-batch-recon performs financial reconciliation; absent
      from the architecture doc.
      Default if declined: in scope.
      Your answer (ENTER to decline): _
```

Type an answer to make it authoritative (the raising phase re-runs with it),
or press ENTER to decline (the stated default stands).

Pass `--max-questions N` to change the slot budget (default 3; `0` never
pauses and runs entirely on defaults). Paused runs created before this
feature cannot be resumed - the checkpointed question schema changed - so
start those assessments fresh.

### Pausing for days: `--pause` / `--resume`

Gathering real answers can take days, and holding a live process open for
that is the wrong shape. With `--pause` the run checkpoints at the gate and
exits:

```bash
poetry run demo --pause
# == PAUSED at the review gate - 5 open questions ==
# Fill in the answers in output/run_<ts>/review.jsonl
# Then: poetry run demo --resume output/run_<ts>
```

`review.jsonl` carries only the human's input - one `{"id", "answer"}` line
per question, seeded empty. The question text stays in `ledger.jsonl`
(agent-curated, read-only); a frontend joins the two on `id`. Leave an answer
empty to decline. Resuming restores the exact gate state and re-runs **zero**
phases - only the answered phases revise, then the report lands. Resume with
the same `--model` the run was paused with; a run that raises no questions
never pauses. A malformed answer line aborts loudly rather than silently
becoming a decline.

To see the scratchpad channel too, edit `scratchpad.md` while the analyzers
are still working and watch for the `[steering]` line.

## Editing the prompts

Phase prompts are not hardcoded: they live in `src/hotl_demo/prompts/` as
markdown files with YAML frontmatter (phase metadata: `name`, `order`,
`per_repo`, `report_filename`) and Jinja2 bodies (the phase instructions;
analyzers receive `{{ unit }}`). Shared wrappers `initial.md` / `revision.md`
/ `final_report.md` assemble the full prompts from `{{ sources }}`,
`{{ memory }}`, `{{ open_questions }}`, etc. Edit the markdown, rerun the
demo - no Python changes needed.

## The sample data

Everything under `sample_data/` is synthetic: three enterprise PDFs
(regenerate with `poetry run python scripts/make_pdfs.py` after editing
`docs_src/`), two fake legacy repos, and a questionnaire template. The corpus
has **planted gaps and conflicts** (Azure mandate vs `boto3` in code, missing
RTO/RPO, unspecified data-residency region, hardcoded credentials, ...) so
the agents reliably find questions worth asking a human.

## Tests and linting

```bash
poetry run pytest                              # fast, LLM-free (includes markdown lint)
OLLAMA_E2E=1 poetry run pytest -m ollama -s    # full live pipeline (slow)
poetry run pymarkdown --config .pymarkdown.json scan README.md CLAUDE.md src/hotl_demo/prompts src/reflexion_demo/prompts
```
