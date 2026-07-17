# Reflexion Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Standalone MAF demo `poetry run reflexion`: a worker agent drafts a migration report with read/write tools, a reviewer with the same read tools independently verifies it, rejection feedback steers redrafts, and two budgets (per-turn tool calls, review cycles) both end in tool-stripping plus a forced "produce from what you have" finish.

**Architecture:** New package `src/reflexion_demo/` (no `hotl_demo` imports). A cyclic `WorkflowBuilder` graph — `worker → reviewer → worker` — with dataclass messages `DraftReady`/`ReviewVerdict`. Agents are constructed fresh per turn by factories, which is how tool-stripping happens: the forced-finalize agent is built with `write_report` only, and a function middleware strips read tools mid-turn via `FunctionInvocationContext.remove_tools()` when the tool-call budget hits.

**Tech Stack:** Python ≥3.10, Microsoft Agent Framework (`agent-framework ~=1.11`, `agent-framework-ollama`), Jinja2, pydantic (transitive dep of agent-framework), pytest (LLM-free default) + one `ollama`-marked live test.

**Spec:** `docs/superpowers/specs/2026-07-17-reflexion-demo-design.md` (approved). Read it before starting.

## Global Constraints

- Python floor is 3.10: no `enum.StrEnum`, no 3.11+ syntax.
- No new dependencies. pydantic is already available transitively via agent-framework; Jinja2 is a declared dependency.
- No imports from `hotl_demo` anywhere in `src/reflexion_demo/` (standalone requirement). Small helpers (preflight, num_ctx) are deliberately duplicated.
- CLI stays stdlib: `argparse` + `print` only.
- Tools return `"ERROR: ..."` strings, never raise — the framework feeds errors back to the model.
- Never create `tests/__init__.py`; tests import as top-level modules.
- Tests are LLM-free by default (`addopts = "-m 'not ollama'"`); executor tests call `@handler` methods directly with fakes from `tests/conftest.py`.
- Do NOT use `from __future__ import annotations` in `graph.py` (handler signature inspection; see the warning at the top of `src/hotl_demo/review.py`) — safest to skip it in every new module.
- Defaults fixed by spec: topic "Assess migrating OMS file storage from the NFS file store to Amazon S3.", `--max-cycles 3`, `--max-tool-calls 12`, model env `OLLAMA_MODEL` (default `gemma4:31b`), `--num-ctx` overrides `OLLAMA_NUM_CTX` (default 4096).
- Corpus root is `sample_data/` filtered to `.md`/`.py`/`.txt` files; run artifacts under `output/reflexion_<YYYYmmdd_HHMMSS>/` (`report.md`, `review_log.jsonl`).
- Budgeted (read) tools: `list_files`, `read_file`, `read_report`. `write_report` is exempt and survives every strip.
- Commit after every task with the trailer:

```text
Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_019TaG2KQ9yi7xkFJcQKZfFC
```

Run all commands from the repo root. `poetry` may be missing from PATH — use `.venv\Scripts\python.exe -m pytest ...` (Windows) as the equivalent of `poetry run pytest ...`.

---

### Task 1: Package scaffold + `budget.py` (ToolBudget + stripping middleware)

**Files:**
- Create: `src/reflexion_demo/__init__.py`
- Create: `src/reflexion_demo/budget.py`
- Modify: `pyproject.toml:15-19` (register the package + console script up front so every later task can import `reflexion_demo`)
- Test: `tests/test_reflexion_budget.py`

**Interfaces:**
- Consumes: `agent_framework.function_middleware`, `FunctionInvocationContext` (annotation only).
- Produces (later tasks rely on these exact names):
  - `ToolBudget` dataclass: fields `max_calls: int`, `spent: int = 0`; property `exhausted: bool` (`spent >= max_calls`).
  - `BUDGETED_TOOL_NAMES: frozenset[str]` = `{"list_files", "read_file", "read_report"}`.
  - `BUDGET_NUDGE: str` (the "reasoning for a long time" sentence).
  - `make_budget_middleware(budget: ToolBudget, budgeted: frozenset[str] | set[str], label: str)` → async function middleware for `Agent(middleware=[...])`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_reflexion_budget.py`:

```python
"""Tool-call budget: counting, exemption, and mid-turn read-tool stripping."""
import pytest

from reflexion_demo.budget import (
    BUDGET_NUDGE,
    BUDGETED_TOOL_NAMES,
    ToolBudget,
    make_budget_middleware,
)


class _Fn:
    def __init__(self, name):
        self.name = name


class FakeInvocationContext:
    """Duck-typed FunctionInvocationContext: the middleware reads
    .function.name, .tools, .result and calls .remove_tools(names)."""

    def __init__(self, tool_name, tools=("list_files", "read_file", "write_report")):
        self.function = _Fn(tool_name)
        self.tools = list(tools)
        self.result = "tool output"
        self.removed = []

    def remove_tools(self, tools):
        self.removed.append(list(tools))


async def _call(mw, ctx):
    async def call_next():
        pass
    await mw(ctx, call_next)


async def test_read_tool_calls_count_and_write_report_is_exempt():
    budget = ToolBudget(max_calls=2)
    mw = make_budget_middleware(budget, BUDGETED_TOOL_NAMES, "worker")
    await _call(mw, FakeInvocationContext("read_file"))
    await _call(mw, FakeInvocationContext("write_report"))
    assert budget.spent == 1
    assert not budget.exhausted


async def test_strip_fires_exactly_once_at_the_budget_with_nudge():
    budget = ToolBudget(max_calls=2)
    mw = make_budget_middleware(budget, BUDGETED_TOOL_NAMES, "worker")
    first = FakeInvocationContext("read_file")
    await _call(mw, first)
    assert first.removed == []          # under budget: untouched
    assert first.result == "tool output"

    second = FakeInvocationContext("list_files")
    await _call(mw, second)
    assert budget.exhausted
    assert second.removed == [sorted(BUDGETED_TOOL_NAMES)]   # strip fired
    assert second.result.startswith("tool output")
    assert BUDGET_NUDGE in second.result

    third = FakeInvocationContext("read_file")   # in-flight batch straggler
    await _call(mw, third)
    assert budget.spent == 3
    assert third.removed == []          # strip + nudge happen only once
    assert BUDGET_NUDGE not in third.result


async def test_strip_survives_a_none_tools_list():
    # context.tools is None when invoked outside a function-calling loop.
    budget = ToolBudget(max_calls=1)
    mw = make_budget_middleware(budget, BUDGETED_TOOL_NAMES, "worker")
    ctx = FakeInvocationContext("read_file")
    ctx.tools = None
    await _call(mw, ctx)                # must not raise
    assert budget.exhausted
    assert BUDGET_NUDGE in ctx.result   # nudge still delivered


async def test_none_result_is_stringified_not_crashed():
    budget = ToolBudget(max_calls=1)
    mw = make_budget_middleware(budget, BUDGETED_TOOL_NAMES, "worker")
    ctx = FakeInvocationContext("read_file")
    ctx.result = None
    await _call(mw, ctx)
    assert BUDGET_NUDGE in ctx.result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_reflexion_budget.py -v`
Expected: FAIL at import time with `ModuleNotFoundError: No module named 'reflexion_demo'`.

- [ ] **Step 3: Write the implementation**

Create `src/reflexion_demo/__init__.py`:

```python
"""Standalone reflexion demo: worker drafts, reviewer verifies, budgets bound both."""
```

Modify `pyproject.toml` — the two touched sections become:

```toml
[project.scripts]
demo = "hotl_demo.main:run"
reflexion = "reflexion_demo.main:run"

[tool.poetry]
packages = [
    { include = "hotl_demo", from = "src" },
    { include = "reflexion_demo", from = "src" },
]
```

Then re-register the editable install (makes `reflexion_demo` importable in
tests for every subsequent task; the `reflexion` script resolves once
`main.py` exists in Task 6):

Run: `poetry install` (no poetry on PATH? `.venv\Scripts\python.exe -m pip install -e . --no-deps`)
Expected: exit 0.

Create `src/reflexion_demo/budget.py`:

```python
"""Per-turn tool-call budget and the mid-turn read-tool strip.

One function middleware owns the counter. Read tools count; ``write_report``
is exempt (delivery, not exploration). On the call that exhausts the budget
the middleware executes the call normally, strips the read tools for the
remainder of the turn via the framework's live-mutation point
(``FunctionInvocationContext.remove_tools``), and appends a nudge to that
call's result so the model knows why its tools vanished.
"""
from dataclasses import dataclass

from agent_framework import FunctionInvocationContext, function_middleware

BUDGETED_TOOL_NAMES = frozenset({"list_files", "read_file", "read_report"})

BUDGET_NUDGE = (
    "[SYSTEM] Tool budget exhausted - you have been reasoning for a long "
    "time. Your exploration tools have been removed. Produce the report now "
    "from the information you already have."
)


@dataclass
class ToolBudget:
    """Mutable per-turn counter; a fresh instance is made for every agent turn."""

    max_calls: int
    spent: int = 0

    @property
    def exhausted(self) -> bool:
        return self.spent >= self.max_calls


def make_budget_middleware(budget: ToolBudget, budgeted, label: str):
    """Build the counting/stripping middleware for one agent turn.

    Args:
        budget: This turn's fresh counter (never share across turns).
        budgeted: Names of tools that count toward - and get stripped at -
            the budget. ``write_report`` must not be in it.
        label: Console tag ("worker"/"reviewer") for the strip line.
    """
    @function_middleware
    async def budget_middleware(context: FunctionInvocationContext, call_next) -> None:
        await call_next()
        if context.function.name not in budgeted:
            return
        budget.spent += 1
        # == not >=: queued calls from the in-flight batch still execute and
        # count, but the strip and the nudge must happen exactly once.
        if budget.spent == budget.max_calls:
            if context.tools is not None:
                # Names not present are ignored by the framework, so passing
                # the whole budgeted set is safe for both agents.
                context.remove_tools(sorted(budgeted))
            context.result = f"{context.result or ''}\n\n{BUDGET_NUDGE}"
            print(f"  [{label}] tool budget exhausted ({budget.max_calls} calls) - read tools stripped")

    return budget_middleware
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_reflexion_budget.py -v`
Expected: 4 PASS. (If `ModuleNotFoundError: reflexion_demo` persists, the editable install didn't refresh — rerun the install command from Step 3, or as a fallback set `$env:PYTHONPATH="src"` for the test run.)

- [ ] **Step 5: Commit**

```bash
git add src/reflexion_demo tests/test_reflexion_budget.py pyproject.toml
git commit -m "feat(reflexion): package scaffold and tool-call budget with mid-turn strip"
```

---

### Task 2: `tools.py` — corpus read tools, report write/read tools

**Files:**
- Create: `src/reflexion_demo/tools.py`
- Test: `tests/test_reflexion_tools.py`

**Interfaces:**
- Consumes: `agent_framework.tool`.
- Produces (exact names later tasks use):
  - `TEXT_SUFFIXES: frozenset[str]` = `{".md", ".py", ".txt"}`.
  - `atomic_write(path: Path, text: str) -> None` — temp file + `os.replace`.
  - `make_corpus_tools(corpus_root: Path) -> list` — `[list_files, read_file]`, traversal-guarded, suffix-filtered, 20k-char read cap.
  - `ReportFlag` class with attribute `written: bool` (starts `False`).
  - `make_report_tools(report_path: Path) -> tuple` — `(write_report, read_report, flag)`; `write_report(markdown: str)` atomic-writes and sets `flag.written`; `read_report()` returns the report or an `ERROR:` string when absent.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_reflexion_tools.py`:

```python
"""Corpus tools (traversal guard, text filter) and report tools (atomic write, flag)."""
from pathlib import Path

from reflexion_demo.tools import make_corpus_tools, make_report_tools


def _corpus(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    (root / "docs_src").mkdir(parents=True)
    (root / "docs_src" / "strategy.md").write_text("Azure is strategic", encoding="utf-8")
    (root / "app.py").write_text("print('hi')", encoding="utf-8")
    (root / "notes.txt").write_text("note", encoding="utf-8")
    (root / "binary.pdf").write_bytes(b"%PDF-1.4")
    return root


def test_list_files_filters_to_text_suffixes(tmp_path):
    list_files, _ = make_corpus_tools(_corpus(tmp_path))
    listing = list_files()
    assert "docs_src/strategy.md" in listing
    assert "app.py" in listing
    assert "notes.txt" in listing
    assert "binary.pdf" not in listing


def test_read_file_reads_and_guards_traversal(tmp_path):
    root = _corpus(tmp_path)
    (tmp_path / "secret.txt").write_text("secret", encoding="utf-8")
    _, read_file = make_corpus_tools(root)
    assert read_file("docs_src/strategy.md") == "Azure is strategic"
    assert read_file("../secret.txt").startswith("ERROR:")
    assert read_file("no/such.md").startswith("ERROR:")


def test_read_file_rejects_non_text_suffix(tmp_path):
    _, read_file = make_corpus_tools(_corpus(tmp_path))
    assert read_file("binary.pdf").startswith("ERROR:")


def test_read_file_truncates_oversized(tmp_path):
    root = _corpus(tmp_path)
    (root / "big.txt").write_text("x" * 25_000, encoding="utf-8")
    _, read_file = make_corpus_tools(root)
    text = read_file("big.txt")
    assert len(text) < 25_000
    assert text.endswith("... (truncated)")


def test_write_report_sets_flag_and_overwrites_atomically(tmp_path):
    report = tmp_path / "report.md"
    write_report, read_report, flag = make_report_tools(report)
    assert flag.written is False
    assert read_report().startswith("ERROR:")          # nothing written yet

    assert "saved" in write_report("# Draft 1").lower()
    assert flag.written is True
    assert report.read_text(encoding="utf-8") == "# Draft 1"
    assert read_report() == "# Draft 1"

    write_report("# Draft 2")                           # revision overwrites
    assert report.read_text(encoding="utf-8") == "# Draft 2"
    assert not list(tmp_path.glob("*.tmp"))             # no temp litter


def test_write_report_rejects_empty(tmp_path):
    write_report, _, flag = make_report_tools(tmp_path / "report.md")
    assert write_report("   ").startswith("ERROR:")
    assert flag.written is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_reflexion_tools.py -v`
Expected: FAIL with `ImportError` (no `reflexion_demo.tools`).

- [ ] **Step 3: Write the implementation**

Create `src/reflexion_demo/tools.py`:

```python
"""Closure-bound tools. Docstrings are the descriptions the LLM sees.

Same idioms as the HOTL demo's tools: traversal-guarded resolution, oversized
reads truncated, failures returned as ``ERROR:`` strings (never raised) so
the framework feeds them back to the model.

Information parity: worker and reviewer get the IDENTICAL corpus binding from
:func:`make_corpus_tools`; the reviewer additionally reads the artifact under
review (``read_report``), the worker additionally writes it (``write_report``).
"""
import os
from pathlib import Path

from agent_framework import tool

TEXT_SUFFIXES = frozenset({".md", ".py", ".txt"})
_READ_CAP = 20_000


def atomic_write(path: Path, text: str) -> None:
    """Write via temp file + ``os.replace`` so readers never see a torn file."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def make_corpus_tools(corpus_root: Path) -> list:
    """Build the shared read-only corpus pair, bound to one root.

    Args:
        corpus_root: Directory both agents may read; resolved once and used
            as the traversal guard boundary.

    Returns:
        ``[list_files, read_file]`` decorated tool functions.
    """
    root = Path(corpus_root).resolve()

    @tool(approval_mode="never_require")
    def list_files() -> str:
        """List every readable source file in the corpus as relative paths,
        one per line. Call this first to see what documentation and code is
        available."""
        files = sorted(
            p.relative_to(root).as_posix()
            for p in root.rglob("*")
            if p.is_file() and p.suffix in TEXT_SUFFIXES
        )
        return "\n".join(files) or "(empty corpus)"

    @tool(approval_mode="never_require")
    def read_file(path: str) -> str:
        """Read one corpus file by its relative path exactly as shown by
        list_files. Returns the full file contents."""
        target = (root / path).resolve()
        if not target.is_relative_to(root):
            return "ERROR: path escapes the corpus. Use a relative path from list_files."
        if target.suffix not in TEXT_SUFFIXES or not target.is_file():
            return f"ERROR: no such readable file: {path}. Call list_files to see valid paths."
        text = target.read_text(encoding="utf-8", errors="replace")
        if len(text) > _READ_CAP:
            text = text[:_READ_CAP] + "\n... (truncated)"
        return text

    return [list_files, read_file]


class ReportFlag:
    """Mutable cell recording whether write_report ran this turn.

    A fresh instance comes out of :func:`make_report_tools` per turn, so no
    reset discipline is needed anywhere.
    """

    def __init__(self) -> None:
        self.written = False


def make_report_tools(report_path: Path) -> tuple:
    """Build the report write/read pair bound to one run's report file.

    Args:
        report_path: ``output/reflexion_<ts>/report.md`` for this run.

    Returns:
        ``(write_report, read_report, flag)`` - the worker gets
        ``write_report``, the reviewer gets ``read_report``, the worker
        executor checks ``flag.written`` after each turn.
    """
    flag = ReportFlag()

    @tool(approval_mode="never_require")
    def write_report(markdown: str) -> str:
        """Save the complete migration report. Pass the FULL report as
        markdown - this overwrites any previous draft, so never send a
        fragment or a diff."""
        if not markdown.strip():
            return "ERROR: report must be non-empty markdown. Send the full report text."
        try:
            atomic_write(report_path, markdown)
        except OSError as exc:
            return f"ERROR: could not save the report ({exc})."
        flag.written = True
        return f"Report saved ({len(markdown)} chars)."

    @tool(approval_mode="never_require")
    def read_report() -> str:
        """Read the report under review exactly as the author saved it."""
        if not report_path.exists():
            return "ERROR: no report has been written yet."
        return report_path.read_text(encoding="utf-8", errors="replace")

    return write_report, read_report, flag
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_reflexion_tools.py -v`
Expected: 6 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/reflexion_demo/tools.py tests/test_reflexion_tools.py
git commit -m "feat(reflexion): corpus read tools and report write/read tools"
```

---

### Task 3: Prompts (`worker.md`, `reviewer.md`) + rendering helpers + lint gate

**Files:**
- Create: `src/reflexion_demo/prompts/worker.md`
- Create: `src/reflexion_demo/prompts/reviewer.md`
- Create: `src/reflexion_demo/prompting.py`
- Modify: `tests/test_markdown_lint.py:14` (add the new prompts dir to `targets`)
- Test: `tests/test_reflexion_prompts.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `render_worker_prompt(*, mode: str, topic: str, cycle: int, max_cycles: int, feedback: str = "", previous_report: str = "") -> str` — `mode` is exactly one of `"initial"`, `"revision"`, `"finalize"` (raises `ValueError` otherwise).
  - `render_reviewer_prompt(*, topic: str, cycle: int) -> str`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_reflexion_prompts.py`:

```python
"""Prompt rendering: the three worker variants and the reviewer brief."""
import pytest

from reflexion_demo.prompting import render_reviewer_prompt, render_worker_prompt


def test_initial_variant_explores_and_delivers_via_write_report():
    text = render_worker_prompt(mode="initial", topic="NFS to S3", cycle=1, max_cycles=3)
    assert "NFS to S3" in text
    assert "write_report" in text
    assert "read_file" in text            # told to explore
    assert "cycle 1 of at most 3" in text
    assert "REJECTED" not in text         # no revision leakage


def test_revision_variant_carries_feedback_and_previous_report():
    text = render_worker_prompt(
        mode="revision", topic="NFS to S3", cycle=2, max_cycles=3,
        feedback="Missing the Azure mandate conflict.",
        previous_report="# Draft 1 with {braces}",
    )
    assert "Missing the Azure mandate conflict." in text
    assert "# Draft 1 with {braces}" in text   # Jinja2 leaves literal braces alone
    assert "REJECTED" in text


def test_finalize_variant_says_tools_are_gone():
    text = render_worker_prompt(
        mode="finalize", topic="NFS to S3", cycle=4, max_cycles=3,
        feedback="Still missing residency analysis.", previous_report="# Draft 3",
    )
    assert "reasoning for a long time" in text
    assert "exploration tools have been removed" in text
    assert "# Draft 3" in text
    assert "read_file" not in text        # must not tell it to explore


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        render_worker_prompt(mode="bogus", topic="t", cycle=1, max_cycles=3)


def test_reviewer_prompt_demands_independent_verification():
    text = render_reviewer_prompt(topic="NFS to S3", cycle=2)
    assert "NFS to S3" in text
    assert "read_report" in text
    assert "read_file" in text            # spot-check against sources
    assert "cycle 2" in text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_reflexion_prompts.py -v`
Expected: FAIL with `ImportError` (no `reflexion_demo.prompting`).

- [ ] **Step 3: Write the templates and the renderer**

Create `src/reflexion_demo/prompts/worker.md` (no YAML frontmatter — there is
no phase discovery here):

```markdown
You are a migration analyst producing an evidence-grounded report for an
independent reviewer.

## Topic

{{ topic }}

{% if mode == "revision" %}
## Reviewer feedback (your previous draft was REJECTED)

{{ feedback }}

## Your previous report

{{ previous_report }}

Revise the report to address every point of the feedback. Re-check the
corpus with your list_files and read_file tools where the feedback demands
new evidence.
{% elif mode == "finalize" %}
## Reviewer feedback on your last draft

{{ feedback }}

## Your previous report

{{ previous_report }}

You have been reasoning for a long time and the review budget is exhausted.
Your exploration tools have been removed. You must now produce the final
report based on the information you already have: improve the previous
report using only the material above.
{% else %}
Explore the corpus with your list_files and read_file tools before writing.
Ground every claim in a source file and cite its relative path. Cover the
material conflicts and gaps the sources reveal for this topic.
{% endif %}

Deliver the COMPLETE report in markdown by calling the write_report tool
with the full text. This is cycle {{ cycle }} of at most {{ max_cycles }}
review cycles.
```

Create `src/reflexion_demo/prompts/reviewer.md`:

```markdown
You are an independent reviewer with the same corpus access as the report's
author. Do not trust the report - verify it.
This is review cycle {{ cycle }}.

## Topic under review

{{ topic }}

Read the report with your read_report tool, then spot-check its claims
against the source files with your list_files and read_file tools. Evaluate:

- Accuracy: claims match the sources they cite.
- Coverage: the material conflicts and gaps in the sources for this topic
  are addressed (for example a cloud-provider mandate that contradicts the
  proposed target, data-residency or secrets-management standards).
- Actionability: findings lead to concrete migration decisions.

Summarize what you verified and every problem you found.
```

Create `src/reflexion_demo/prompting.py`:

```python
"""Prompt rendering: Jinja2 templates in ``prompts/``, one per agent.

The worker template carries all three variants (initial/revision/finalize)
selected by ``mode`` - explicit variant files would repeat the shared
delivery contract three times.
"""
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

PROMPTS_DIR = Path(__file__).parent / "prompts"
_ENV = Environment(loader=FileSystemLoader(str(PROMPTS_DIR)), keep_trailing_newline=True)

_WORKER_MODES = ("initial", "revision", "finalize")


def render_worker_prompt(*, mode: str, topic: str, cycle: int, max_cycles: int,
                         feedback: str = "", previous_report: str = "") -> str:
    """Render the worker's turn prompt.

    Args:
        mode: ``"initial"``, ``"revision"``, or ``"finalize"``.
        topic: The migration topic under assessment.
        cycle: 1-based draft number this turn produces.
        max_cycles: The review-cycle budget (for the model's situational
            awareness).
        feedback: Reviewer feedback (revision/finalize only).
        previous_report: Prior report text (revision/finalize only).

    Raises:
        ValueError: Unknown ``mode`` - programmer error, fail loud.
    """
    if mode not in _WORKER_MODES:
        raise ValueError(f"unknown worker mode: {mode!r}")
    return _ENV.get_template("worker.md").render(
        mode=mode, topic=topic, cycle=cycle, max_cycles=max_cycles,
        feedback=feedback, previous_report=previous_report,
    ).strip()


def render_reviewer_prompt(*, topic: str, cycle: int) -> str:
    """Render the reviewer's evaluation brief for one cycle."""
    return _ENV.get_template("reviewer.md").render(topic=topic, cycle=cycle).strip()
```

Modify `tests/test_markdown_lint.py` — change the `targets` line:

```python
    targets = ["README.md", "src/hotl_demo/prompts", "src/reflexion_demo/prompts"]
```

- [ ] **Step 4: Run tests to verify they pass (including the lint gate)**

Run: `.venv\Scripts\python.exe -m pytest tests/test_reflexion_prompts.py tests/test_markdown_lint.py -v`
Expected: all PASS. If the lint gate fails, fix the reported rule violations in the two new `.md` files (common ones: trailing spaces MD009, missing final newline MD047, inconsistent list indent MD007).

- [ ] **Step 5: Commit**

```bash
git add src/reflexion_demo/prompts src/reflexion_demo/prompting.py tests/test_reflexion_prompts.py tests/test_markdown_lint.py
git commit -m "feat(reflexion): worker/reviewer prompt templates and renderer"
```

---

### Task 4: `graph.py` part 1 — messages, verdict parsing, ReviewerExecutor

**Files:**
- Create: `src/reflexion_demo/graph.py`
- Modify: `tests/conftest.py:86-92` (`FakeAgent.run` gains `**kwargs`)
- Test: `tests/test_reflexion_reviewer.py`

**Interfaces:**
- Consumes: `render_reviewer_prompt` (Task 3).
- Produces (Task 5 builds on these exact names):
  - `@dataclass DraftReady`: `cycle: int`, `topic: str`.
  - `@dataclass ReviewVerdict`: `approved: bool`, `feedback: str`, `cycle: int`, `reviewer_tool_calls: int`.
  - `class ReviewOutput(BaseModel)`: `approved: bool`, `feedback: str`.
  - `parse_verdict(text: str) -> ReviewOutput | None` (fence-tolerant).
  - `ReviewerExecutor(agent_factory)` — `agent_factory()` returns `(agent, budget)`; handler `on_draft(DraftReady, ctx)` sends `ReviewVerdict`.
  - Module constants `REPORT_FILENAME = "report.md"`, `LOG_FILENAME = "review_log.jsonl"`.

- [ ] **Step 1: Extend the shared FakeAgent (additive, no behavior change)**

In `tests/conftest.py`, replace `FakeAgent.__init__` body line `self.sessions = []` context — full replacement of the two methods shown; keep the rest of the class as is:

```python
    def __init__(self, texts, side_effect=None):
        self.texts = list(texts)
        self.prompts = []
        self.sessions = []
        self.run_kwargs = []
        self.created_sessions = []
        self.side_effect = side_effect  # optional callable(prompt) run per call

    def create_session(self, *, session_id=None):
        """Hand out an opaque session sentinel, as the real Agent does."""
        session = f"session-{len(self.created_sessions) + 1}"
        self.created_sessions.append(session)
        return session

    async def run(self, prompt, *, session=None, **kwargs):
        """Record prompt + session + extra kwargs, fire the side effect, pop the next text."""
        self.prompts.append(prompt)
        self.sessions.append(session)
        self.run_kwargs.append(kwargs)
        if self.side_effect:
            self.side_effect(prompt)
        return FakeAgentResult(self.texts.pop(0) if self.texts else "")
```

Run: `.venv\Scripts\python.exe -m pytest tests/ -q`
Expected: entire existing suite still PASSES (the change is additive).

- [ ] **Step 2: Write the failing tests**

Create `tests/test_reflexion_reviewer.py`:

```python
"""Verdict parsing and the ReviewerExecutor's two-call review turn."""
import json

from conftest import FakeAgent, FakeCtx

from reflexion_demo.budget import ToolBudget
from reflexion_demo.graph import DraftReady, ReviewerExecutor, ReviewVerdict, parse_verdict


def _factory(agent, spent=0):
    budget = ToolBudget(max_calls=12, spent=spent)
    return lambda: (agent, budget)


def test_parse_verdict_plain_json():
    v = parse_verdict('{"approved": false, "feedback": "missing residency"}')
    assert v is not None and v.approved is False and v.feedback == "missing residency"


def test_parse_verdict_tolerates_code_fences():
    v = parse_verdict('```json\n{"approved": true, "feedback": "ok"}\n```')
    assert v is not None and v.approved is True


def test_parse_verdict_garbage_returns_none():
    assert parse_verdict("I approve of this report.") is None
    assert parse_verdict("") is None


async def test_reviewer_sends_verdict_from_structured_second_call():
    agent = FakeAgent(["I checked the sources.",
                       '{"approved": false, "feedback": "cite file_store.py"}'])
    ctx = FakeCtx()
    await ReviewerExecutor(_factory(agent, spent=5)).on_draft(
        DraftReady(cycle=1, topic="NFS to S3"), ctx)

    [verdict] = ctx.sent
    assert isinstance(verdict, ReviewVerdict)
    assert verdict.approved is False
    assert verdict.feedback == "cite file_store.py"
    assert verdict.cycle == 1
    assert verdict.reviewer_tool_calls == 5

    assert len(agent.prompts) == 2                      # explore, then verdict
    assert "NFS to S3" in agent.prompts[0]
    assert agent.sessions[0] == agent.sessions[1]       # same session: turn 2 sees turn 1
    assert agent.run_kwargs[0] == {}                    # exploration: no format forcing
    assert "response_format" in agent.run_kwargs[1].get("options", {})


async def test_reviewer_retries_once_then_rejects_on_unparseable():
    agent = FakeAgent(["explored", "not json", "still not json"])
    ctx = FakeCtx()
    await ReviewerExecutor(_factory(agent)).on_draft(DraftReady(cycle=2, topic="t"), ctx)

    [verdict] = ctx.sent
    assert verdict.approved is False                    # fail-closed, never approve
    assert "could not produce a valid verdict" in verdict.feedback
    assert len(agent.prompts) == 3                      # explore + verdict + one retry
    assert "not valid JSON" in agent.prompts[2]         # retry names the problem
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_reflexion_reviewer.py -v`
Expected: FAIL with `ImportError` (no `reflexion_demo.graph`).

- [ ] **Step 4: Write the implementation**

Create `src/reflexion_demo/graph.py` (NO `from __future__ import annotations` — see Global Constraints):

```python
"""Reflexion workflow: messages, verdict schema, and the two executors.

Routing convention (same as the HOTL pipeline): message TYPES encode meaning.
``DraftReady`` only ever flows worker -> reviewer; ``ReviewVerdict`` only ever
flows reviewer -> worker. No mode flags anywhere.

This module deliberately avoids ``from __future__ import annotations``: the
framework inspects handler signatures, and string annotations are a known
trap (see the warning at the top of hotl_demo/review.py).
"""
import re
from dataclasses import dataclass

from pydantic import BaseModel, ValidationError

from .prompting import render_reviewer_prompt

REPORT_FILENAME = "report.md"
LOG_FILENAME = "review_log.jsonl"


@dataclass
class DraftReady:
    """Worker -> reviewer: draft ``cycle`` is on disk, review it."""

    cycle: int
    topic: str


@dataclass
class ReviewVerdict:
    """Reviewer -> worker: boolean verdict plus steering feedback.

    ``reviewer_tool_calls`` rides along so the worker executor - the sole
    review-log writer - can log complete cycle lines.
    """

    approved: bool
    feedback: str
    cycle: int
    reviewer_tool_calls: int


class ReviewOutput(BaseModel):
    """The reviewer's structured verdict (``response_format`` schema)."""

    approved: bool
    feedback: str


_FENCE_OPEN = re.compile(r"^```[a-zA-Z]*\s*")
_FENCE_CLOSE = re.compile(r"```\s*$")

_VERDICT_PROMPT = (
    "Based on your review, return your verdict now as a JSON object with "
    'exactly two fields: "approved" (boolean) and "feedback" (string). '
    "Approve only if accuracy, coverage, and actionability all hold. On "
    "rejection, the feedback must name what is missing or wrong and which "
    "angle to pursue next."
)

_VERDICT_RETRY = (
    "\nYour previous reply was not valid JSON with boolean \"approved\" and "
    "string \"feedback\". Return ONLY that JSON object, nothing else."
)

_UNPARSEABLE_FEEDBACK = (
    "The reviewer could not produce a valid verdict this cycle. Improve the "
    "report's evidence citations and completeness, then resubmit."
)


def parse_verdict(text: str) -> "ReviewOutput | None":
    """Parse the model's verdict text; ``None`` when it does not validate.

    Tolerates markdown code fences - local models add them even when told
    not to.
    """
    cleaned = _FENCE_CLOSE.sub("", _FENCE_OPEN.sub("", (text or "").strip())).strip()
    try:
        return ReviewOutput.model_validate_json(cleaned)
    except ValidationError:
        return None


# Executor/handler imports live below the pure helpers so tests of the pure
# parts stay importable even if the framework changes.
from agent_framework import Executor, WorkflowContext, handler  # noqa: E402


class ReviewerExecutor(Executor):
    """Independent verification: explore, then emit a structured verdict.

    Two calls in ONE session: the exploration turn builds context (read the
    report, spot-check sources), the verdict turn extracts the boolean +
    feedback under ``response_format``. Splitting them keeps schema forcing
    away from the tool-calling turn - local models handle each half better
    than both at once.
    """

    def __init__(self, agent_factory, id: str = "reviewer") -> None:
        """Args:
            agent_factory: Zero-arg callable returning ``(agent, budget)``
                fresh for this turn; the agent carries the corpus read tools,
                ``read_report``, and the budget middleware.
            id: Workflow node id.
        """
        super().__init__(id=id)
        self._agent_factory = agent_factory

    @handler
    async def on_draft(self, draft: DraftReady, ctx: WorkflowContext[ReviewVerdict]) -> None:
        agent, budget = self._agent_factory()
        session = agent.create_session()
        print(f"  [reviewer] cycle {draft.cycle}: verifying against the corpus...")
        await agent.run(
            render_reviewer_prompt(topic=draft.topic, cycle=draft.cycle),
            session=session,
        )
        verdict = None
        prompt = _VERDICT_PROMPT
        for _ in range(2):  # one attempt + one retry
            result = await agent.run(
                prompt, session=session,
                options={"response_format": ReviewOutput},
            )
            verdict = parse_verdict(result.text)
            if verdict is not None:
                break
            prompt = _VERDICT_PROMPT + _VERDICT_RETRY
        if verdict is None:
            # Fail closed: an unverifiable draft must never ship as approved.
            verdict = ReviewOutput(approved=False, feedback=_UNPARSEABLE_FEEDBACK)
        print(f"  [reviewer] cycle {draft.cycle}: "
              f"{'APPROVED' if verdict.approved else 'REJECTED'}")
        if not verdict.approved:
            print(f"  [reviewer] feedback: {verdict.feedback}")
        await ctx.send_message(ReviewVerdict(
            approved=verdict.approved, feedback=verdict.feedback,
            cycle=draft.cycle, reviewer_tool_calls=budget.spent,
        ))
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_reflexion_reviewer.py -v`
Expected: 5 PASS.

- [ ] **Step 6: Commit**

```bash
git add src/reflexion_demo/graph.py tests/test_reflexion_reviewer.py tests/conftest.py
git commit -m "feat(reflexion): messages, verdict parsing, reviewer executor"
```

---

### Task 5: `graph.py` part 2 — WorkerExecutor, workflow builder, full LLM-free loop

**Files:**
- Modify: `src/reflexion_demo/graph.py` (append WorkerExecutor + `build_reflexion_workflow`; extend imports)
- Test: `tests/test_reflexion_worker.py`

**Interfaces:**
- Consumes: Task 1 `ToolBudget`; Task 2 `atomic_write`, `ReportFlag`; Task 3 `render_worker_prompt`; Task 4 messages/`ReviewerExecutor`.
- Produces:
  - `WorkerExecutor(agent_factory, run_dir: Path, max_cycles: int)` — `agent_factory(finalize: bool)` returns `(agent, budget, flag)`; handlers `on_topic(str, ctx)` and `on_verdict(ReviewVerdict, ctx)`.
  - `build_reflexion_workflow(worker: WorkerExecutor, reviewer: ReviewerExecutor)` → built cyclic workflow.
  - `_WRITE_REPORT_NUDGE: str` (module constant; tests reference behavior, not the constant).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_reflexion_worker.py`:

```python
"""WorkerExecutor cycle logic, forced finalize, write_report fallback, review log."""
import json

from conftest import FakeAgent, FakeCtx

from reflexion_demo.budget import ToolBudget
from reflexion_demo.graph import (
    DraftReady,
    ReviewerExecutor,
    ReviewVerdict,
    WorkerExecutor,
    build_reflexion_workflow,
)
from reflexion_demo.tools import ReportFlag, atomic_write


def _worker_factory(run_dir, *, write=True, spent=3):
    """Fake agent factory: each call yields a FakeAgent whose side effect
    mimics the write_report tool (writes the file, sets the flag)."""
    calls = []

    def factory(finalize=False):
        flag = ReportFlag()
        agent_holder = {}

        def side_effect(prompt):
            if write:
                atomic_write(run_dir / "report.md", f"# Draft after: {prompt[:40]}")
                flag.written = True

        agent = FakeAgent(["draft text", "retry text"], side_effect=side_effect)
        agent_holder["agent"] = agent
        calls.append({"finalize": finalize, "agent": agent, "flag": flag})
        return agent, ToolBudget(max_calls=12, spent=spent), flag

    factory.calls = calls
    return factory


def _log_lines(run_dir):
    path = run_dir / "review_log.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


async def test_on_topic_drafts_and_announces_cycle_one(tmp_path):
    factory = _worker_factory(tmp_path)
    worker = WorkerExecutor(factory, tmp_path, max_cycles=3)
    ctx = FakeCtx()
    await worker.on_topic("NFS to S3", ctx)

    assert ctx.sent == [DraftReady(cycle=1, topic="NFS to S3")]
    assert factory.calls[0]["finalize"] is False
    assert "NFS to S3" in factory.calls[0]["agent"].prompts[0]
    assert (tmp_path / "report.md").exists()


async def test_approved_verdict_yields_and_logs(tmp_path):
    factory = _worker_factory(tmp_path)
    worker = WorkerExecutor(factory, tmp_path, max_cycles=3)
    await worker.on_topic("t", FakeCtx())

    ctx = FakeCtx()
    await worker.on_verdict(
        ReviewVerdict(approved=True, feedback="good", cycle=1, reviewer_tool_calls=7), ctx)

    assert ctx.sent == []
    assert len(ctx.outputs) == 1 and "approved" in ctx.outputs[0]
    lines = _log_lines(tmp_path)
    assert lines[0] == {"cycle": 1, "approved": True, "feedback": "good",
                        "forced": False, "worker_tool_calls": 3,
                        "reviewer_tool_calls": 7}
    assert lines[1]["outcome"] == "approved" and lines[1]["cycles"] == 1


async def test_rejection_with_cycles_left_revises_with_feedback(tmp_path):
    factory = _worker_factory(tmp_path)
    worker = WorkerExecutor(factory, tmp_path, max_cycles=3)
    await worker.on_topic("t", FakeCtx())

    ctx = FakeCtx()
    await worker.on_verdict(
        ReviewVerdict(approved=False, feedback="cover Azure mandate",
                      cycle=1, reviewer_tool_calls=2), ctx)

    assert ctx.sent == [DraftReady(cycle=2, topic="t")]
    assert ctx.outputs == []
    revision_agent = factory.calls[1]["agent"]
    assert factory.calls[1]["finalize"] is False
    assert "cover Azure mandate" in revision_agent.prompts[0]
    assert "# Draft after:" in revision_agent.prompts[0]   # previous report inlined


async def test_rejection_at_budget_forces_toolless_finalize(tmp_path):
    factory = _worker_factory(tmp_path)
    worker = WorkerExecutor(factory, tmp_path, max_cycles=3)
    await worker.on_topic("t", FakeCtx())

    ctx = FakeCtx()
    await worker.on_verdict(
        ReviewVerdict(approved=False, feedback="still wrong",
                      cycle=3, reviewer_tool_calls=4), ctx)

    assert ctx.sent == []                                   # no fourth review
    assert len(ctx.outputs) == 1 and "forced" in ctx.outputs[0]
    assert factory.calls[1]["finalize"] is True             # read tools stripped at construction
    assert "reasoning for a long time" in factory.calls[1]["agent"].prompts[0]
    lines = _log_lines(tmp_path)
    assert lines[0]["forced"] is False and lines[0]["approved"] is False
    assert lines[1] == {"cycle": 4, "approved": False, "feedback": "still wrong",
                        "forced": True, "worker_tool_calls": 3,
                        "reviewer_tool_calls": None}
    assert lines[2]["outcome"] == "forced" and lines[2]["cycles"] == 3


async def test_missing_write_report_gets_one_nudge_then_text_fallback(tmp_path):
    factory = _worker_factory(tmp_path, write=False)        # tool never "runs"
    worker = WorkerExecutor(factory, tmp_path, max_cycles=3)
    await worker.on_topic("t", FakeCtx())

    agent = factory.calls[0]["agent"]
    assert len(agent.prompts) == 2                          # draft + one nudge
    assert "write_report" in agent.prompts[1]
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert report == "retry text"                           # final text persisted


async def test_full_loop_reject_then_approve_llm_free(tmp_path):
    worker_factory = _worker_factory(tmp_path)
    reviewer_scripts = [
        ["looked at sources", '{"approved": false, "feedback": "add residency"}'],
        ["looked again", '{"approved": true, "feedback": "solid"}'],
    ]

    def reviewer_factory():
        return FakeAgent(reviewer_scripts.pop(0)), ToolBudget(max_calls=12, spent=1)

    workflow = build_reflexion_workflow(
        WorkerExecutor(worker_factory, tmp_path, max_cycles=3),
        ReviewerExecutor(reviewer_factory),
    )
    outputs = []
    async for event in workflow.run("NFS to S3", stream=True):
        if event.type == "output":
            outputs.append(event.data)

    assert len(outputs) == 1 and "approved" in outputs[0]
    assert len(worker_factory.calls) == 2                   # draft + one revision
    lines = _log_lines(tmp_path)
    assert [ln.get("approved") for ln in lines[:2]] == [False, True]
    assert lines[2]["outcome"] == "approved"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_reflexion_worker.py -v`
Expected: FAIL with `ImportError: cannot import name 'WorkerExecutor'`.

- [ ] **Step 3: Write the implementation**

In `src/reflexion_demo/graph.py`, extend the top-of-file imports (final form):

```python
import json
import re
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ValidationError

from .prompting import render_reviewer_prompt, render_worker_prompt
from .tools import atomic_write
```

and change the framework import line (added `WorkflowBuilder`):

```python
from agent_framework import Executor, WorkflowBuilder, WorkflowContext, handler  # noqa: E402
```

Append after `ReviewerExecutor`:

```python
_WRITE_REPORT_NUDGE = (
    "You finished without saving the report. Call the write_report tool NOW "
    "with the complete report markdown."
)


class WorkerExecutor(Executor):
    """Draft/revise/finalize the report; sole writer of the review log.

    Cycle state lives in executor memory - this demo has no checkpoint or
    resume, so in-memory counters are safe here (unlike the HOTL gate, whose
    progress must be ledger-derived).
    """

    def __init__(self, agent_factory, run_dir: Path, max_cycles: int,
                 id: str = "worker") -> None:
        """Args:
            agent_factory: ``factory(finalize: bool) -> (agent, budget, flag)``,
                fresh per turn. ``finalize=True`` must construct the agent
                with ``write_report`` only - stripping expressed at
                construction time.
            run_dir: This run's artifact directory (report + review log).
            max_cycles: Review-cycle budget; cycle ``max_cycles`` rejecting
                triggers the forced finalize.
            id: Workflow node id.
        """
        super().__init__(id=id)
        self._agent_factory = agent_factory
        self._report_path = run_dir / REPORT_FILENAME
        self._log_path = run_dir / LOG_FILENAME
        self._max_cycles = max_cycles
        self._topic = ""
        self._last_spent = 0

    @handler
    async def on_topic(self, topic: str, ctx: WorkflowContext[DraftReady]) -> None:
        """First draft, then hand to the reviewer."""
        self._topic = topic
        print("== cycle 1: drafting ==")
        await self._draft(render_worker_prompt(
            mode="initial", topic=topic, cycle=1, max_cycles=self._max_cycles))
        await ctx.send_message(DraftReady(cycle=1, topic=topic))

    @handler
    async def on_verdict(self, verdict: ReviewVerdict,
                         ctx: WorkflowContext[DraftReady, str]) -> None:
        """Approve -> ship; reject -> revise; budget exhausted -> forced finalize."""
        self._append_log({
            "cycle": verdict.cycle, "approved": verdict.approved,
            "feedback": verdict.feedback, "forced": False,
            "worker_tool_calls": self._last_spent,
            "reviewer_tool_calls": verdict.reviewer_tool_calls,
        })
        if verdict.approved:
            self._append_log({"outcome": "approved", "cycles": verdict.cycle,
                              "report": str(self._report_path)})
            await ctx.yield_output(
                f"Report approved after {verdict.cycle} cycle(s): {self._report_path}")
            return
        if verdict.cycle >= self._max_cycles:
            print("== review budget exhausted: forced finalize (read tools stripped) ==")
            await self._draft(render_worker_prompt(
                mode="finalize", topic=self._topic, cycle=verdict.cycle + 1,
                max_cycles=self._max_cycles, feedback=verdict.feedback,
                previous_report=self._previous_report()), finalize=True)
            self._append_log({
                "cycle": verdict.cycle + 1, "approved": False,
                "feedback": verdict.feedback, "forced": True,
                "worker_tool_calls": self._last_spent,
                "reviewer_tool_calls": None,
            })
            self._append_log({"outcome": "forced", "cycles": verdict.cycle,
                              "report": str(self._report_path)})
            await ctx.yield_output(
                f"Report shipped unapproved after {verdict.cycle} cycle(s) "
                f"(forced finalize): {self._report_path}")
            return
        next_cycle = verdict.cycle + 1
        print(f"== cycle {next_cycle}: revising ==")
        await self._draft(render_worker_prompt(
            mode="revision", topic=self._topic, cycle=next_cycle,
            max_cycles=self._max_cycles, feedback=verdict.feedback,
            previous_report=self._previous_report()))
        await ctx.send_message(DraftReady(cycle=next_cycle, topic=self._topic))

    async def _draft(self, prompt: str, finalize: bool = False) -> None:
        """One drafting turn: run the agent, ensure the report landed.

        One session covers the turn and its nudge retry, so the retry sees
        the exploration (``Agent.run(session=None)`` is stateless per call -
        same idiom and same reason as hotl_demo/phases.py).
        """
        agent, budget, flag = self._agent_factory(finalize)
        session = agent.create_session()
        result = await agent.run(prompt, session=session)
        if not flag.written:
            result = await agent.run(_WRITE_REPORT_NUDGE, session=session)
            if not flag.written:
                # Last resort: the turn's final text IS the report.
                text = (result.text or "").strip() or "(no report produced)"
                atomic_write(self._report_path, text)
                print("  [worker] write_report never called - persisted final text instead")
        self._last_spent = budget.spent

    def _previous_report(self) -> str:
        """Prior report text for revision/finalize prompts (worker has no
        read access to the report file - the prompt carries it)."""
        if not self._report_path.exists():
            return "(no previous report)"
        return self._report_path.read_text(encoding="utf-8", errors="replace")

    def _append_log(self, record: dict) -> None:
        with self._log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


def build_reflexion_workflow(worker: WorkerExecutor, reviewer: ReviewerExecutor):
    """Assemble the cyclic worker <-> reviewer graph.

    Both edges are unconditional: each direction carries exactly one message
    type and each executor handles exactly that type, so isinstance dispatch
    does the routing.
    """
    return (
        WorkflowBuilder(start_executor=worker)
        .add_edge(worker, reviewer)
        .add_edge(reviewer, worker)
        .build()
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_reflexion_worker.py -v`
Expected: 6 PASS. If `test_full_loop...` fails on event shapes, print `event.type` values — the assertion loop must only rely on `event.type == "output"` and `event.data` (same API `hotl_demo/main.py:361-365` uses).

- [ ] **Step 5: Run the whole suite**

Run: `.venv\Scripts\python.exe -m pytest -q`
Expected: everything passes (existing HOTL tests untouched).

- [ ] **Step 6: Commit**

```bash
git add src/reflexion_demo/graph.py tests/test_reflexion_worker.py
git commit -m "feat(reflexion): worker executor, forced finalize, cyclic workflow"
```

---

### Task 6: `main.py` CLI + packaging

**Files:**
- Create: `src/reflexion_demo/main.py`
- Test: `tests/test_reflexion_main.py`

**Interfaces:**
- Consumes: everything above; `agent_framework.Agent`, `agent_framework.ollama.OllamaChatClient`.
- Produces: console script `reflexion = "reflexion_demo.main:run"`; factories `make_worker_factory(corpus_root, report_path, max_tool_calls)` and `make_reviewer_factory(corpus_root, report_path, max_tool_calls)` (module-level, importable by the E2E test).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_reflexion_main.py`:

```python
"""CLI pure parts: preflight helpers and the agent factories' tool wiring."""
import pytest

from reflexion_demo.main import (
    DEFAULT_TOPIC,
    make_reviewer_factory,
    make_worker_factory,
    model_present,
    normalize_host,
    resolve_num_ctx,
)


def test_normalize_host_adds_scheme():
    assert normalize_host("127.0.0.1:11434") == "http://127.0.0.1:11434"
    assert normalize_host("https://ollama.local/") == "https://ollama.local"


def test_model_present_mirrors_ollama_latest_resolution():
    tags = {"models": [{"name": "gemma4:31b"}, {"name": "phi4:latest"}]}
    assert model_present(tags, "gemma4:31b")
    assert model_present(tags, "phi4")
    assert not model_present(tags, "gemma4")


def test_resolve_num_ctx_env_default(monkeypatch):
    monkeypatch.delenv("OLLAMA_NUM_CTX", raising=False)
    assert resolve_num_ctx() == 4096
    monkeypatch.setenv("OLLAMA_NUM_CTX", "32768")
    assert resolve_num_ctx() == 32768


def test_default_topic_is_the_s3_assessment():
    assert "NFS file store" in DEFAULT_TOPIC and "S3" in DEFAULT_TOPIC


def _tool_names(agent):
    return sorted(t.name for t in agent.default_options["tools"])


def test_worker_factory_tool_sets(tmp_path, monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "test-model")
    factory = make_worker_factory(tmp_path, tmp_path / "report.md", max_tool_calls=5)

    agent, budget, flag = factory(False)
    assert _tool_names(agent) == ["list_files", "read_file", "write_report"]
    assert budget.max_calls == 5 and budget.spent == 0 and flag.written is False

    final_agent, final_budget, _ = factory(True)
    assert _tool_names(final_agent) == ["write_report"]     # stripped at construction
    assert final_budget.spent == 0


def test_reviewer_factory_tool_set(tmp_path, monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "test-model")
    factory = make_reviewer_factory(tmp_path, tmp_path / "report.md", max_tool_calls=5)
    agent, budget = factory()
    assert _tool_names(agent) == ["list_files", "read_file", "read_report"]
    assert budget.max_calls == 5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_reflexion_main.py -v`
Expected: FAIL with `ImportError` (no `reflexion_demo.main`).

- [ ] **Step 3: Write the implementation**

Create `src/reflexion_demo/main.py`:

```python
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
    preflight(normalize_host(os.environ.get("OLLAMA_HOST", "http://localhost:11434")),
              args.model)

    run_dir = Path("output") / datetime.now().strftime("reflexion_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    corpus_root = Path("sample_data")
    report_path = run_dir / REPORT_FILENAME

    worker = WorkerExecutor(
        make_worker_factory(corpus_root, report_path, args.max_tool_calls),
        run_dir, args.max_cycles)
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
```

(`pyproject.toml` was already updated in Task 1; the `reflexion` script now
resolves because `reflexion_demo.main:run` exists. If `.venv\Scripts\reflexion.exe`
is missing, rerun the install command from Task 1 Step 3.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_reflexion_main.py -v`
Expected: 7 PASS. Note: the factory tests construct real `Agent` objects with a real `OllamaChatClient()` but never call the server — construction is offline (it only reads env vars).

- [ ] **Step 5: Smoke the CLI wiring without a server**

Run: `.venv\Scripts\python.exe -m reflexion_demo.main --help`
Expected: usage text listing `--topic`, `--max-cycles`, `--max-tool-calls`, `--model`, `--num-ctx`; exit 0.

- [ ] **Step 6: Commit**

```bash
git add src/reflexion_demo/main.py tests/test_reflexion_main.py
git commit -m "feat(reflexion): CLI entry point and agent factories"
```

---

### Task 7: Live E2E test + full verification

**Files:**
- Create: `tests/test_e2e_reflexion.py`
- Test: the whole suite + markdown lint

**Interfaces:**
- Consumes: `make_worker_factory`/`make_reviewer_factory` (Task 6), executors + builder (Tasks 4-5).

- [ ] **Step 1: Write the live smoke test**

Create `tests/test_e2e_reflexion.py`:

```python
"""Live E2E smoke: one bounded reflexion run against local Ollama.

    OLLAMA_E2E=1 poetry run pytest -m ollama -s tests/test_e2e_reflexion.py

Tiny budgets keep it short: 1 review cycle means at most draft -> review ->
forced finalize. Asserts the artifacts, not the model's prose.
"""
import json
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.ollama


@pytest.mark.skipif(os.environ.get("OLLAMA_E2E") != "1", reason="set OLLAMA_E2E=1 to run")
async def test_reflexion_smoke(tmp_path):
    from reflexion_demo.graph import (
        REPORT_FILENAME,
        ReviewerExecutor,
        WorkerExecutor,
        build_reflexion_workflow,
    )
    from reflexion_demo.main import DEFAULT_TOPIC, make_reviewer_factory, make_worker_factory

    corpus = Path("sample_data")
    report_path = tmp_path / REPORT_FILENAME
    workflow = build_reflexion_workflow(
        WorkerExecutor(make_worker_factory(corpus, report_path, max_tool_calls=6),
                       tmp_path, max_cycles=1),
        ReviewerExecutor(make_reviewer_factory(corpus, report_path, max_tool_calls=6)),
    )

    outputs = []
    async for event in workflow.run(DEFAULT_TOPIC, stream=True):
        if event.type == "output":
            outputs.append(event.data)

    assert len(outputs) == 1
    assert report_path.exists() and report_path.read_text(encoding="utf-8").strip()
    lines = [json.loads(ln) for ln in
             (tmp_path / "review_log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert lines[0]["cycle"] == 1
    assert lines[-1]["outcome"] in ("approved", "forced")
```

- [ ] **Step 2: Run the LLM-free suite + lint**

Run: `.venv\Scripts\python.exe -m pytest -q`
Expected: all pass, E2E deselected by the default `-m 'not ollama'`.

Run: `.venv\Scripts\python.exe -m pymarkdown --config .pymarkdown.json scan README.md CLAUDE.md src/hotl_demo/prompts src/reflexion_demo/prompts`
Expected: exit 0, no output.

- [ ] **Step 3: Run the live E2E (requires local Ollama with the model pulled)**

PowerShell: `$env:OLLAMA_E2E="1"; .venv\Scripts\python.exe -m pytest -m ollama -s tests/test_e2e_reflexion.py`
Expected: PASS in a few minutes on `gemma4:31b`. If the reviewer's structured verdict misbehaves on the local model (never parses), the run still completes via the reject-fallback — the test asserts artifacts, not approval.

- [ ] **Step 4: Try the real demo once**

Run: `.venv\Scripts\reflexion.exe --max-cycles 2 --max-tool-calls 8` (or `poetry run reflexion ...`)
Expected: console shows cycle banners, reviewer verdict lines, possibly a strip line; `output/reflexion_<ts>/report.md` and `review_log.jsonl` exist. This is a manual observation step, not a gate — note anything odd for follow-up.

- [ ] **Step 5: Commit**

```bash
git add tests/test_e2e_reflexion.py
git commit -m "test(reflexion): live E2E smoke behind OLLAMA_E2E"
```

---

## Post-plan notes

- Out of scope per spec §11: checkpoint/resume, scratchpad steering, compaction, ArtifactStore reuse, PDF reading, DevUI, `as_agent()`, README changes.
- Known ceiling (`ponytail:` candidates): no compaction — with default `num_ctx=4096` and 12 tool calls a worker turn can overflow; the mitigation is `--num-ctx 32768` on capable hardware. Add compaction only if a real run overflows.
- The reviewer's two-call turn (explore, then `response_format` verdict) is deliberate: schema forcing on the tool-calling turn confuses local models.




