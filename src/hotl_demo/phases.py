"""Phase definitions: workflow messages, specs, source loaders, prompt rendering.

Phase prompts live in prompts/ as markdown files: YAML frontmatter carries the
phase metadata (name, order, per_repo, report_filename), the body is the phase
instructions (a Jinja2 template). Shared Jinja2 wrappers (initial.md,
revision.md, final_report.md) assemble the full prompts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml
from agent_framework import Agent, Executor, WorkflowContext, handler
from agent_framework.ollama import OllamaChatClient
from jinja2 import Environment, FileSystemLoader
from pypdf import PdfReader

from .artifacts import PHASES, REPOS, ArtifactStore
from .tools import SCRATCHPAD_PATH, make_repo_tools, make_tools

PROMPTS_DIR = Path(__file__).parent / "prompts"
PROMPT_ENV = Environment(loader=FileSystemLoader(str(PROMPTS_DIR)), keep_trailing_newline=True)


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
    repo_dir: Path | None = None  # set for deep_analysis: enables repo-exploration tools


# -- source loaders ---------------------------------------------------------

def load_pdf_text(path: Path) -> str:
    return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)


def repo_listing(repo_dir: Path) -> str:
    files = sorted(p.relative_to(repo_dir).as_posix() for p in repo_dir.rglob("*") if p.is_file())
    return "\n".join(files)


# -- prompt files (markdown + YAML frontmatter, Jinja2 bodies) ----------------

def parse_prompt_file(path: Path) -> tuple[dict, str]:
    """Split a prompt file into (frontmatter dict, body template)."""
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        raise ValueError(f"{path.name}: missing YAML frontmatter")
    _, meta_block, body = text.split("---", 2)
    meta = yaml.safe_load(meta_block) or {}
    return meta, body.lstrip("\n")


# -- spec factory -------------------------------------------------------------

def build_phase_specs(base_dir: Path, prompts_dir: Path = PROMPTS_DIR) -> list[PhaseSpec]:
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
                f"--- REPOSITORY: {repo} ---\n"
                "Contents not included. Explore the repository with your "
                "list_files and read_file tools."
            )
        return load

    def ec_sources() -> str:
        return "\n\n".join(
            f"--- DOCUMENT: {p.name} ---\n{load_pdf_text(p)}" for p in (pdf2, pdf3)
        )

    def questionnaire_sources() -> str:
        template = (base_dir / "questionnaire_template.md").read_text(encoding="utf-8")
        return f"--- QUESTIONNAIRE TEMPLATE ---\n{template}"

    source_loaders: dict[str, Callable[[], str]] = {
        "discovery": discovery_sources,
        "enterprise_context": ec_sources,
        "questionnaire": questionnaire_sources,
    }

    phase_files = [
        parse_prompt_file(p) for p in prompts_dir.glob("*.md") if p.stem in PHASES
    ]
    phase_files.sort(key=lambda mb: mb[0]["order"])

    specs: list[PhaseSpec] = []
    for meta, body in phase_files:
        name = meta["name"]
        if meta.get("per_repo"):
            for repo in REPOS:
                specs.append(PhaseSpec(
                    name=name, unit=repo, executor_id=f"analyze:{repo}",
                    report_filename=meta["report_filename"].format(unit=repo),
                    instructions=PROMPT_ENV.from_string(body).render(unit=repo).strip(),
                    load_sources=analyzer_sources(repo),
                    repo_dir=repos / repo,
                ))
        else:
            specs.append(PhaseSpec(
                name=name, unit=None, executor_id=name,
                report_filename=meta["report_filename"],
                instructions=PROMPT_ENV.from_string(body).render().strip(),
                load_sources=source_loaders[name],
            ))
    return specs


# -- prompt rendering ----------------------------------------------------------

def _format_open_questions(open_questions: list[dict]) -> str:
    if not open_questions:
        return "No questions raised so far."
    return "\n".join(
        f"- {q['id']} ({q['phase']}{'/' + q['unit'] if q.get('unit') else ''}): {q['question']}"
        for q in open_questions
    )


def _format_answers(answers: list[dict]) -> str:
    return "\n".join(
        f"- {a['id']}: Q: {a['question']}\n  Human answer (AUTHORITATIVE): {a['human_answer']}\n"
        f"  (replaces default assumption: {a['default_assumption']})"
        for a in answers
    )


def build_initial_prompt(spec: PhaseSpec, sources: str, memory_text: str,
                         open_questions: list[dict]) -> str:
    return PROMPT_ENV.get_template("initial.md").render(
        instructions=spec.instructions,
        open_questions=_format_open_questions(open_questions),
        memory=memory_text,
        sources=sources,
    )


def build_revision_prompt(spec: PhaseSpec, sources: str, memory_text: str,
                          open_questions: list[dict], answers: list[dict],
                          previous_report: str) -> str:
    return PROMPT_ENV.get_template("revision.md").render(
        instructions=spec.instructions,
        open_questions=_format_open_questions(open_questions),
        answers=_format_answers(answers),
        previous_report=previous_report,
        memory=memory_text,
        sources=sources,
    )


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
        tools = make_tools(store, spec.name, spec.unit, scratchpad_path)
        if spec.repo_dir is not None:
            tools += make_repo_tools(spec.repo_dir)
        self._agent = agent or Agent(
            client=OllamaChatClient(),  # model comes from OLLAMA_MODEL env var
            name=spec.executor_id.replace(":", "_"),
            instructions="You are one phase of a multi-agent assessment pipeline.",
            tools=tools,
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
