"""Phase definitions: workflow messages, specs, source loaders, prompt builders."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from agent_framework import Agent, Executor, WorkflowContext, handler
from agent_framework.ollama import OllamaChatClient
from pypdf import PdfReader

from .artifacts import REPOS, ArtifactStore
from .tools import SCRATCHPAD_PATH, make_tools


# -- workflow messages (types encode mode: no mode flags anywhere) -------

@dataclass
class PhaseDone:
    phase: str
    unit: str | None = None


@dataclass
class AnalysisDone:
    unit: str


@dataclass
class RevisionDone:
    phase: str
    unit: str | None = None


@dataclass
class RevisionTrigger:
    phase: str
    unit: str | None
    answers: list[dict] = field(default_factory=list)


@dataclass
class ReportTrigger:
    pass


# -- specs ----------------------------------------------------------------

@dataclass
class PhaseSpec:
    name: str
    unit: str | None
    executor_id: str
    report_filename: str
    instructions: str
    load_sources: Callable[[], str]


# -- source loaders ---------------------------------------------------------

def load_pdf_text(path: Path) -> str:
    return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)


def repo_listing(repo_dir: Path) -> str:
    files = sorted(p.relative_to(repo_dir).as_posix() for p in repo_dir.rglob("*") if p.is_file())
    return "\n".join(files)


def load_repo_text(repo_dir: Path) -> str:
    blocks = []
    for rel in repo_listing(repo_dir).splitlines():
        blocks.append(f"=== {rel} ===\n{(repo_dir / rel).read_text(encoding='utf-8')}")
    return "\n\n".join(blocks)


# -- phase instructions -----------------------------------------------------

_COMMON_DUTIES = """
Your duties on every run:
1. FIRST call the read_scratchpad tool and follow any operator guidance in it.
2. Record 3-8 key findings with the update_memory tool (short snake_case key,
   concise factual value). Findings must be grounded in the source material.
3. When evidence conflicts or a decision-critical fact is missing, call the
   raise_question tool with the question, the evidence context, and the
   default assumption you will proceed with - then proceed using that default.
   Check the OPEN QUESTIONS list you were given first: never re-raise a
   question that is already open; reference its id instead.
4. Finish by writing your phase report as your final answer: well-structured
   markdown, headings, concise, evidence-cited. The final answer must be the
   report itself - no preamble about what you are going to do.
"""

_INSTRUCTIONS: dict[str, str] = {
    "discovery": (
        "You are the discovery analyst opening a cloud migration readiness "
        "assessment for Meridian Retail's Order Management System (OMS). "
        "Establish the TRUE purpose and shape of the legacy estate: business "
        "function, users, criticality, and the actual scope of what must "
        "migrate. Compare what the documents claim against what the "
        "repositories actually contain; flag scope that code reveals but "
        "documents omit. Do not deep-dive into code internals - that is a "
        "later phase." + _COMMON_DUTIES
    ),
    "deep_analysis": (
        "You are a senior engineer performing a repo-level deep dive on ONE "
        "repository ({unit}) of the OMS estate for cloud migration readiness. "
        "Analyze runtime and language versions, frameworks, data access, "
        "external integrations, file system coupling, schedulers, secrets "
        "handling, and cloud blockers. Be specific: name files and lines of "
        "evidence." + _COMMON_DUTIES
    ),
    "enterprise_context": (
        "You are the enterprise architect overlaying corporate guidance onto "
        "the assessment: cloud strategy and approved patterns, cybersecurity "
        "and data-protection standards. Map each earlier finding (in shared "
        "memory) to the relevant corporate mandate, and call out every "
        "conflict between strategy and observed reality, every policy "
        "violation, and every mandate whose parameters are unspecified."
        + _COMMON_DUTIES
    ),
    "questionnaire": (
        "You are completing the standard Cloud Migration Readiness "
        "Questionnaire. Fill in EVERY slot of the template using the shared "
        "memory and phase evidence. Cite evidence for each answer. Where an "
        "answer rests on a default assumption from an open ledger question, "
        "reference the question id. If a slot cannot be answered and no open "
        "question covers it, raise one. Your final answer is the completed "
        "questionnaire in the template's structure." + _COMMON_DUTIES
    ),
}


# -- spec factory -------------------------------------------------------------

def build_phase_specs(base_dir: Path) -> list[PhaseSpec]:
    docs = base_dir / "docs"
    repos = base_dir / "repos"
    pdf1 = docs / "01_oms_application_architecture.pdf"
    pdf2 = docs / "02_enterprise_cloud_strategy.pdf"
    pdf3 = docs / "03_cybersecurity_standards.pdf"

    def discovery_sources() -> str:
        parts = [f"--- DOCUMENT: {p.name} ---\n{load_pdf_text(p)}" for p in (pdf1, pdf2, pdf3)]
        for repo in REPOS:
            parts.append(f"--- REPO FILE LISTING: {repo} ---\n{repo_listing(repos / repo)}")
            readme = repos / repo / "README.md"
            parts.append(f"--- {repo}/README.md ---\n{readme.read_text(encoding='utf-8')}")
        return "\n\n".join(parts)

    def analyzer_sources(repo: str) -> Callable[[], str]:
        def load() -> str:
            return (
                f"--- DOCUMENT: {pdf1.name} ---\n{load_pdf_text(pdf1)}\n\n"
                f"--- REPOSITORY: {repo} (full contents) ---\n{load_repo_text(repos / repo)}"
            )
        return load

    def ec_sources() -> str:
        return "\n\n".join(
            f"--- DOCUMENT: {p.name} ---\n{load_pdf_text(p)}" for p in (pdf2, pdf3)
        )

    def questionnaire_sources() -> str:
        template = (base_dir / "questionnaire_template.md").read_text(encoding="utf-8")
        return f"--- QUESTIONNAIRE TEMPLATE ---\n{template}"

    specs = [
        PhaseSpec("discovery", None, "discovery", "phase_01_discovery.md",
                  _INSTRUCTIONS["discovery"], discovery_sources),
    ]
    for repo in REPOS:
        specs.append(PhaseSpec(
            "deep_analysis", repo, f"analyze:{repo}",
            f"phase_02_deep_analysis_{repo}.md",
            _INSTRUCTIONS["deep_analysis"].format(unit=repo),
            analyzer_sources(repo),
        ))
    specs.append(PhaseSpec("enterprise_context", None, "enterprise_context",
                           "phase_03_enterprise_context.md",
                           _INSTRUCTIONS["enterprise_context"], ec_sources))
    specs.append(PhaseSpec("questionnaire", None, "questionnaire",
                           "phase_04_questionnaire.md",
                           _INSTRUCTIONS["questionnaire"], questionnaire_sources))
    return specs


# -- prompt builders -----------------------------------------------------------

def _format_open_questions(open_questions: list[dict]) -> str:
    if not open_questions:
        return "No questions raised so far."
    return "\n".join(
        f"- {q['id']} ({q['phase']}{'/' + q['unit'] if q.get('unit') else ''}): {q['question']}"
        for q in open_questions
    )


def build_initial_prompt(spec: PhaseSpec, sources: str, memory_text: str,
                         open_questions: list[dict]) -> str:
    return f"""{spec.instructions}

Remember: call read_scratchpad first; record findings with update_memory;
raise adjudication needs with raise_question.

## OPEN QUESTIONS already in the ledger (do not re-raise)
{_format_open_questions(open_questions)}

## SHARED MEMORY (accumulated by earlier phases)
```json
{memory_text}
```

## SOURCE MATERIAL
{sources}

Produce your phase report now.
"""


def build_revision_prompt(spec: PhaseSpec, sources: str, memory_text: str,
                          open_questions: list[dict], answers: list[dict],
                          previous_report: str) -> str:
    answer_lines = "\n".join(
        f"- {a['id']}: Q: {a['question']}\n  Human answer (AUTHORITATIVE): {a['human_answer']}\n"
        f"  (replaces default assumption: {a['default_assumption']})"
        for a in answers
    )
    return f"""{spec.instructions}

A human reviewer has adjudicated questions this phase raised. Human answers
are authoritative and override any conflicting document or code evidence.
Rewrite your phase report and refresh your update_memory findings to reflect
them. Do not raise these questions again.

## OPEN QUESTIONS already in the ledger (do not re-raise)
{_format_open_questions(open_questions)}

## HUMAN ANSWERS
{answer_lines}

## YOUR PREVIOUS REPORT
{previous_report}

## SHARED MEMORY
```json
{memory_text}
```

## SOURCE MATERIAL
{sources}

Produce the revised phase report now.
"""


# -- executor -----------------------------------------------------------------

_NUDGE = """You produced your phase report but recorded no findings in shared
memory. Call the update_memory tool now for each of the 3-8 key findings in
the report below, then reply "done".

{report}
"""

_MEMORY_GAP_NOTE = "\n\n> NOTE: agent recorded no memory entries for this phase."


class PhaseExecutor(Executor):
    """One workflow node per phase (and per repo for deep_analysis)."""

    def __init__(self, spec: PhaseSpec, store: ArtifactStore,
                 scratchpad_path=SCRATCHPAD_PATH, agent: Any | None = None) -> None:
        super().__init__(id=spec.executor_id)
        self._spec = spec
        self._store = store
        self._agent = agent or Agent(
            client=OllamaChatClient(),  # model comes from OLLAMA_MODEL env var
            name=spec.executor_id.replace(":", "_"),
            instructions="You are one phase of a multi-agent assessment pipeline.",
            tools=make_tools(store, spec.name, spec.unit, scratchpad_path),
        )

    @handler
    async def on_start(self, go: str, ctx: WorkflowContext[PhaseDone | AnalysisDone]) -> None:
        await self._run_initial(ctx)

    @handler
    async def on_upstream(self, done: PhaseDone,
                          ctx: WorkflowContext[PhaseDone | AnalysisDone]) -> None:
        await self._run_initial(ctx)

    @handler
    async def on_revision(self, trig: RevisionTrigger,
                          ctx: WorkflowContext[RevisionDone]) -> None:
        prompt = build_revision_prompt(
            self._spec, self._spec.load_sources(), self._store.memory_text(),
            self._store.open_questions(), trig.answers,
            self._store.read_report(self._spec.report_filename),
        )
        text = await self._invoke(prompt)
        self._store.write_report(self._spec.report_filename, text)
        print(f"  revised: {self._spec.executor_id}")
        await ctx.send_message(RevisionDone(self._spec.name, self._spec.unit))

    async def _run_initial(self, ctx) -> None:
        before = self._store.memory_key_count(self._spec.name, self._spec.unit)
        prompt = build_initial_prompt(
            self._spec, self._spec.load_sources(), self._store.memory_text(),
            self._store.open_questions(),
        )
        text = await self._invoke(prompt)
        if self._store.memory_key_count(self._spec.name, self._spec.unit) == before:
            # ponytail: one bounded nudge, then proceed and note the gap
            await self._invoke(_NUDGE.format(report=text))
            if self._store.memory_key_count(self._spec.name, self._spec.unit) == before:
                text += _MEMORY_GAP_NOTE
        self._store.write_report(self._spec.report_filename, text)
        raised = [q for q in self._store.read_ledger()
                  if q["phase"] == self._spec.name and q["unit"] == self._spec.unit]
        print(f"  {self._spec.executor_id}: report written ({len(raised)} questions raised)")
        if self._spec.unit is not None:
            await ctx.send_message(AnalysisDone(self._spec.unit))
        else:
            await ctx.send_message(PhaseDone(self._spec.name))

    async def _invoke(self, prompt: str) -> str:
        result = await self._agent.run(prompt)
        return result.text or "(no text returned by the model)"
