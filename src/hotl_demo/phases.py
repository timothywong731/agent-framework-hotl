"""Phase definitions: workflow messages, specs, source loaders, prompt rendering.

Phase prompts live in ``prompts/`` as markdown files: YAML frontmatter carries
the phase metadata (``name``, ``order``, ``per_repo``, ``report_filename``),
the body is the phase instructions (a Jinja2 template). Shared Jinja2 wrappers
(``initial.md``, ``revision.md``, ``final_report.md``) assemble the full
prompts. ``build_phase_specs`` discovers phases by reading that directory, so
phases can be edited or reordered without touching Python.

Routing convention used across the workflow: message *types* encode the run
mode. ``PhaseDone``/``AnalysisDone`` only ever mean "initial run finished";
``RevisionDone`` only ever means "post-review re-run finished". Edges filter
on type, so no executor ever needs a mode flag.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml
from agent_framework import Agent, Executor, MessageInjectionMiddleware, WorkflowContext, handler
from agent_framework.ollama import OllamaChatClient
from jinja2 import Environment, FileSystemLoader
from pypdf import PdfReader

from .artifacts import PHASES, REPOS, ArtifactStore
from .steering import ScratchpadWatch, make_steering_middleware
from .tools import SCRATCHPAD_PATH, make_repo_tools, make_tools

PROMPTS_DIR = Path(__file__).parent / "prompts"
PROMPT_ENV = Environment(loader=FileSystemLoader(str(PROMPTS_DIR)), keep_trailing_newline=True)


# -- workflow messages (types encode mode: no mode flags anywhere) -------

@dataclass
class PhaseDone:
    """Initial-run completion of a non-analyzer phase (or of the join node).

    Routed forward along the pipeline chain by ``is_type(PhaseDone)`` edges.

    Attributes:
        phase: Phase name, e.g. ``"discovery"``.
        unit: Always ``None`` for this message (analyzers emit
            :class:`AnalysisDone` instead).

    Example:
        >>> PhaseDone("discovery")
        PhaseDone(phase='discovery', unit=None)
    """

    phase: str
    unit: str | None = None


@dataclass
class AnalysisDone:
    """Initial-run completion of ONE repo analyzer; routed to the join node.

    A distinct type (rather than ``PhaseDone`` with a unit) so the fan-in
    join can never confuse an analyzer completion with anything else.

    Attributes:
        unit: The repo the analyzer covered, e.g. ``"oms-monolith"``.
    """

    unit: str


@dataclass
class RevisionDone:
    """Post-review re-run completion of any phase; routed back to the review gate.

    Attributes:
        phase: Phase name that finished its revision.
        unit: Repo name for deep_analysis revisions, else ``None``.
    """

    phase: str
    unit: str | None = None


@dataclass
class RevisionTrigger:
    """Review gate's instruction to one phase: re-run with human answers.

    Attributes:
        phase: Target phase name.
        unit: Target repo for deep_analysis, else ``None``.
        answers: The answered ledger entries for this ``(phase, unit)``;
            each is a full ledger dict (``id``, ``question``,
            ``human_answer``, ``default_assumption``, ...).

    Example:
        >>> RevisionTrigger("discovery", None, answers=[{"id": "q-1",
        ...     "question": "In scope?", "human_answer": "yes",
        ...     "default_assumption": "in scope"}]).phase
        'discovery'
    """

    phase: str
    unit: str | None
    answers: list[dict] = field(default_factory=list)


@dataclass
class ReportTrigger:
    """Review gate's instruction to the final-report executor: compose now."""


# -- specs ----------------------------------------------------------------

@dataclass
class PhaseSpec:
    """Everything an executor needs to run one phase instance.

    Built by :func:`build_phase_specs` from the prompt files; ``per_repo``
    phases expand into one spec per repo.

    Attributes:
        name: Phase name; one of :data:`hotl_demo.artifacts.PHASES`.
        unit: Repo name for analyzer instances, else ``None``.
        executor_id: Workflow node id (``"discovery"``,
            ``"analyze:oms-monolith"``, ...).
        report_filename: Markdown report written into the run directory.
        instructions: Rendered phase instructions (the prompt-file body with
            ``{{ unit }}`` already substituted).
        load_sources: Zero-arg callable returning the pre-loaded source
            material for this phase's prompt.
        repo_dir: Repo path for deep_analysis instances - presence switches
            on the ``list_files``/``read_file`` exploration tools.
    """

    name: str
    unit: str | None
    executor_id: str
    report_filename: str
    instructions: str
    load_sources: Callable[[], str]
    repo_dir: Path | None = None  # set for deep_analysis: enables repo-exploration tools


# -- source loaders ---------------------------------------------------------

def load_pdf_text(path: Path) -> str:
    """Extract plain text from every page of a PDF.

    Args:
        path: PDF file to read.

    Returns:
        Page texts joined with newlines. Extraction may wrap lines
        mid-phrase; normalize whitespace before substring-matching on it.

    Example:
        >>> text = load_pdf_text(Path("sample_data/docs/02_enterprise_cloud_strategy.pdf"))
        >>> "Azure" in text
        True
    """
    return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)


def repo_listing(repo_dir: Path) -> str:
    """List every file in a repo as sorted, POSIX-style relative paths.

    Args:
        repo_dir: Repository root directory.

    Returns:
        One relative path per line, e.g. ``"README.md\\ndb.py\\n..."``.

    Example:
        >>> "s3_uploader.py" in repo_listing(Path("sample_data/repos/oms-monolith"))
        True
    """
    files = sorted(p.relative_to(repo_dir).as_posix() for p in repo_dir.rglob("*") if p.is_file())
    return "\n".join(files)


# -- prompt files (markdown + YAML frontmatter, Jinja2 bodies) ----------------

def parse_prompt_file(path: Path) -> tuple[dict, str]:
    """Split a prompt file into its YAML frontmatter and Jinja2 body.

    Args:
        path: A ``prompts/<phase>.md`` file that begins with a ``---``
            frontmatter fence.

    Returns:
        ``(meta, body)`` where ``meta`` is the parsed frontmatter dict and
        ``body`` is the template text with the fence and leading blank
        lines stripped.

    Raises:
        ValueError: If the file does not start with a frontmatter fence.

    Example:
        >>> meta, body = parse_prompt_file(PROMPTS_DIR / "discovery.md")
        >>> meta["name"], meta["order"]
        ('discovery', 1)
    """
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        raise ValueError(f"{path.name}: missing YAML frontmatter")
    # maxsplit=2: the body may legitimately contain "---" (markdown rules).
    _, meta_block, body = text.split("---", 2)
    meta = yaml.safe_load(meta_block) or {}
    return meta, body.lstrip("\n")


# -- spec factory -------------------------------------------------------------

def build_phase_specs(base_dir: Path, prompts_dir: Path = PROMPTS_DIR) -> list[PhaseSpec]:
    """Discover phases from the prompt files and wire in their source loaders.

    Each phase's pre-loaded source material differs deliberately (see spec
    section 3): discovery reads everything shallowly, analyzers get the app
    doc only (they explore their repo through tools), enterprise_context gets
    the corporate-guidance docs, and the questionnaire gets its template.

    Args:
        base_dir: Sample-data root (contains ``docs/``, ``repos/``,
            ``questionnaire_template.md``).
        prompts_dir: Directory of prompt files; defaults to the packaged
            ``prompts/``.

    Returns:
        Specs ordered by frontmatter ``order``, with ``per_repo`` phases
        expanded into one spec per entry of
        :data:`hotl_demo.artifacts.REPOS`.

    Example:
        >>> specs = build_phase_specs(Path("sample_data"))
        >>> [s.executor_id for s in specs]
        ['discovery', 'analyze:oms-monolith', 'analyze:oms-batch-recon', \
'enterprise_context', 'questionnaire']
    """
    docs = base_dir / "docs"
    repos = base_dir / "repos"
    pdf1 = docs / "01_oms_application_architecture.pdf"
    pdf2 = docs / "02_enterprise_cloud_strategy.pdf"
    pdf3 = docs / "03_cybersecurity_standards.pdf"

    def discovery_sources() -> str:
        """All three PDFs plus each repo's file listing and README - shallow, wide."""
        parts = [f"--- DOCUMENT: {p.name} ---\n{load_pdf_text(p)}" for p in (pdf1, pdf2, pdf3)]
        for repo in REPOS:
            parts.append(f"--- REPO FILE LISTING: {repo} ---\n{repo_listing(repos / repo)}")
            readme = repos / repo / "README.md"
            parts.append(f"--- {repo}/README.md ---\n{readme.read_text(encoding='utf-8')}")
        return "\n\n".join(parts)

    def analyzer_sources(repo: str) -> Callable[[], str]:
        """App-architecture PDF only; the repo itself is explored via tools."""
        def load() -> str:
            return (
                f"--- DOCUMENT: {pdf1.name} ---\n{load_pdf_text(pdf1)}\n\n"
                f"--- REPOSITORY: {repo} ---\n"
                "Contents not included. Explore the repository with your "
                "list_files and read_file tools."
            )
        return load

    def ec_sources() -> str:
        """Corporate guidance: cloud strategy + security standards PDFs."""
        return "\n\n".join(
            f"--- DOCUMENT: {p.name} ---\n{load_pdf_text(p)}" for p in (pdf2, pdf3)
        )

    def questionnaire_sources() -> str:
        """The readiness question template the phase must fill in."""
        template = (base_dir / "questionnaire_template.md").read_text(encoding="utf-8")
        return f"--- QUESTIONNAIRE TEMPLATE ---\n{template}"

    source_loaders: dict[str, Callable[[], str]] = {
        "discovery": discovery_sources,
        "enterprise_context": ec_sources,
        "questionnaire": questionnaire_sources,
    }

    # Only phase files (stem in PHASES) carry frontmatter; the wrapper
    # templates (initial.md, revision.md, final_report.md) are skipped here.
    phase_files = [
        parse_prompt_file(p) for p in prompts_dir.glob("*.md") if p.stem in PHASES
    ]
    phase_files.sort(key=lambda mb: mb[0]["order"])

    specs: list[PhaseSpec] = []
    for meta, body in phase_files:
        name = meta["name"]
        if meta.get("per_repo"):
            # One executor instance per repo: frontmatter report_filename uses
            # str.format ({unit}); the Jinja2 body uses {{ unit }}.
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
    """Render open ledger entries as a bullet list for duplicate suppression.

    Args:
        open_questions: Ledger entries with ``status == "open"``.

    Returns:
        One ``- q-N (phase[/unit]): question`` bullet per entry, or a
        sentence saying none exist yet.

    Example:
        >>> _format_open_questions([{"id": "q-1", "phase": "discovery",
        ...     "unit": None, "question": "Scope?"}])
        '- q-1 (discovery): Scope?'
    """
    if not open_questions:
        return "No questions raised so far."
    return "\n".join(
        f"- {q['id']} ({q['phase']}{'/' + q['unit'] if q.get('unit') else ''}): {q['question']}"
        for q in open_questions
    )


def _format_answers(answers: list[dict]) -> str:
    """Render answered ledger entries for a revision prompt.

    Each answer is marked AUTHORITATIVE so the model prefers it over any
    conflicting document/code evidence.

    Args:
        answers: Answered ledger entries (``human_answer`` populated).

    Returns:
        Multi-line bullets pairing each question with its human answer and
        the default assumption it replaces.
    """
    return "\n".join(
        f"- {a['id']}: Q: {a['question']}\n  Human answer (AUTHORITATIVE): {a['human_answer']}\n"
        f"  (replaces default assumption: {a['default_assumption']})"
        for a in answers
    )


def build_initial_prompt(spec: PhaseSpec, sources: str, memory_text: str,
                         open_questions: list[dict]) -> str:
    """Render the full first-run prompt for a phase via ``prompts/initial.md``.

    Args:
        spec: The phase being run (supplies the instructions).
        sources: Pre-loaded source material from ``spec.load_sources()``.
        memory_text: Current ``memory.json`` as pretty-printed JSON.
        open_questions: Open ledger entries (for duplicate suppression).

    Returns:
        The complete prompt string handed to the phase agent.

    Example:
        >>> spec = build_phase_specs(Path("sample_data"))[0]
        >>> "OPEN QUESTIONS" in build_initial_prompt(spec, "src", "{}", [])
        True
    """
    return PROMPT_ENV.get_template("initial.md").render(
        instructions=spec.instructions,
        open_questions=_format_open_questions(open_questions),
        memory=memory_text,
        sources=sources,
    )


def build_revision_prompt(spec: PhaseSpec, sources: str, memory_text: str,
                          open_questions: list[dict], answers: list[dict],
                          previous_report: str) -> str:
    """Render the post-review re-run prompt via ``prompts/revision.md``.

    Args:
        spec: The phase being revised.
        sources: Pre-loaded source material from ``spec.load_sources()``.
        memory_text: Current ``memory.json`` as pretty-printed JSON.
        open_questions: Still-open ledger entries (for duplicate suppression).
        answers: Answered ledger entries for this phase - rendered as
            authoritative human decisions.
        previous_report: The phase's earlier report, so the model rewrites
            rather than starts from scratch.

    Returns:
        The complete revision prompt string.
    """
    return PROMPT_ENV.get_template("revision.md").render(
        instructions=spec.instructions,
        open_questions=_format_open_questions(open_questions),
        answers=_format_answers(answers),
        previous_report=previous_report,
        memory=memory_text,
        sources=sources,
    )


# -- executor -----------------------------------------------------------------

# NOT str.format-ed: reports routinely contain literal braces (code, JSON).
_NUDGE_PREFIX = """You produced your phase report but recorded no findings in shared
memory. Call the update_memory tool now for each of the 3-8 key findings in
the report below, then reply "done".

"""

_REPORT_RETRY = (
    "You explored the sources but did not produce the phase report. Write your "
    "complete phase report now as your final answer: well-structured markdown, "
    "headings, concise, evidence-cited."
)

_MEMORY_GAP_NOTE = "\n\n> NOTE: agent recorded no memory entries for this phase."

# Local models occasionally leak chat-template specials (e.g. "<|tool_response>")
# as final text after a tool-heavy run; strip them and treat the rest as the text.
_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|>]{1,32}\|?>")


def _clean_text(text: str | None) -> str:
    """Strip leaked chat-template special tokens and surrounding whitespace.

    Args:
        text: Raw model text (may be ``None``).

    Returns:
        Cleaned text; empty string when nothing meaningful remains.

    Example:
        >>> _clean_text("<|tool_response|>")
        ''
        >>> _clean_text("ok <|eot|> done")
        'ok  done'
    """
    return _SPECIAL_TOKEN_RE.sub("", text or "").strip()


class PhaseExecutor(Executor):
    """One workflow node per phase (and per repo for deep_analysis).

    Lifecycle per node:

    * ``on_start``/``on_upstream`` -> :meth:`_run_initial`: render the initial
      prompt, run the agent (tools do the side effects: scratchpad, ledger,
      memory, repo exploration), write the report, emit ``AnalysisDone`` (has
      ``unit``) or ``PhaseDone``.
    * ``on_revision``: re-run with human answers marked authoritative,
      overwrite the report, emit ``RevisionDone`` back to the review gate.

    Each run cycle mints ONE ``AgentSession`` shared by every turn in it, which
    is what makes the bounded memory nudge and report retry effective - the
    follow-up turn sees the whole earlier exploration. This must be explicit:
    ``Agent.run(session=None)`` is stateless per call. Revisions mint a fresh
    session, since the revision prompt is self-contained by design.

    Scratchpad edits made mid-run are pushed into the live session by the
    steering middleware (see :mod:`hotl_demo.steering`).

    Example:
        >>> spec = build_phase_specs(Path("sample_data"))[0]
        >>> executor = PhaseExecutor(spec, store)  # doctest: +SKIP
    """

    def __init__(self, spec: PhaseSpec, store: ArtifactStore,
                 scratchpad_path=SCRATCHPAD_PATH, agent: Any | None = None) -> None:
        """Wire the phase's agent with its tool belt.

        Args:
            spec: Phase definition from :func:`build_phase_specs`.
            store: Shared artifact store for this run.
            scratchpad_path: Steering file surfaced by ``read_scratchpad``.
            agent: Test seam - a scripted stand-in replaces the real
                Ollama-backed ``Agent`` when provided.
        """
        super().__init__(id=spec.executor_id)
        self._spec = spec
        self._store = store
        tools = make_tools(store, spec.name, spec.unit, scratchpad_path)
        if spec.repo_dir is not None:
            # Analyzers explore their repo agentically instead of receiving it.
            tools += make_repo_tools(spec.repo_dir)
        # MessageInjectionMiddleware (chat) does delivery; steering_mw (function)
        # does detection. A mixed middleware list is a supported MiddlewareTypes
        # shape, not a workaround.
        injector = MessageInjectionMiddleware()
        steering_mw = make_steering_middleware(
            ScratchpadWatch(scratchpad_path), injector, spec.executor_id
        )
        self._agent = agent or Agent(
            client=OllamaChatClient(),  # model comes from OLLAMA_MODEL env var
            name=spec.executor_id.replace(":", "_"),
            instructions="You are one phase of a multi-agent assessment pipeline.",
            tools=tools,
            middleware=[injector, steering_mw],
        )
        # Set per run cycle by _run_initial/on_revision; not shared across cycles.
        self._session = None

    @handler
    async def on_start(self, go: str, ctx: WorkflowContext[PhaseDone | AnalysisDone]) -> None:
        """Entry point for the workflow's start executor (discovery only).

        Args:
            go: The initial ``workflow.run("start")`` payload; ignored.
            ctx: Workflow context used to emit the completion message.
        """
        await self._run_initial(ctx)

    @handler
    async def on_upstream(self, done: PhaseDone,
                          ctx: WorkflowContext[PhaseDone | AnalysisDone]) -> None:
        """React to the previous phase's initial completion.

        Args:
            done: Upstream completion (contents unused; arrival is the signal).
            ctx: Workflow context used to emit the completion message.
        """
        await self._run_initial(ctx)

    @handler
    async def on_revision(self, trig: RevisionTrigger,
                          ctx: WorkflowContext[RevisionDone]) -> None:
        """Re-run this phase with human answers and overwrite its report.

        Args:
            trig: The review gate's targeted instruction, carrying the
                answered ledger entries for this ``(phase, unit)``.
            ctx: Workflow context; emits ``RevisionDone`` back to the gate.
        """
        self._session = self._agent.create_session()
        prompt = build_revision_prompt(
            self._spec, self._spec.load_sources(), self._store.memory_text(),
            self._store.open_questions(), trig.answers,
            self._store.read_report(self._spec.report_filename),
        )
        text = await self._invoke_report(prompt)
        self._store.write_report(self._spec.report_filename, text)
        print(f"  revised: {self._spec.executor_id}")
        await ctx.send_message(RevisionDone(self._spec.name, self._spec.unit))

    async def _run_initial(self, ctx) -> None:
        """First run of the phase: report + memory + (maybe) ledger questions.

        Core flow: snapshot the phase's memory-key count, run the agent, and
        if the count did not move, nudge exactly once - the deal is "report
        comes back as text; memory/ledger arrive via tool calls", and local
        models sometimes skip the tool half. A phase that still refuses gets
        its report annotated rather than failing the pipeline.

        Args:
            ctx: Workflow context used to emit ``AnalysisDone``/``PhaseDone``.
        """
        self._session = self._agent.create_session()
        before = self._store.memory_key_count(self._spec.name, self._spec.unit)
        prompt = build_initial_prompt(
            self._spec, self._spec.load_sources(), self._store.memory_text(),
            self._store.open_questions(),
        )
        text = await self._invoke_report(prompt)
        if self._store.memory_key_count(self._spec.name, self._spec.unit) == before:
            # ponytail: one bounded nudge, then proceed and note the gap
            await self._invoke(_NUDGE_PREFIX + text)
            if self._store.memory_key_count(self._spec.name, self._spec.unit) == before:
                text += _MEMORY_GAP_NOTE
        self._store.write_report(self._spec.report_filename, text)
        raised = [q for q in self._store.read_ledger()
                  if q["phase"] == self._spec.name and q["unit"] == self._spec.unit]
        print(f"  {self._spec.executor_id}: report written ({len(raised)} questions raised)")
        # Message TYPE tells the graph where to route: analyzers feed the
        # fan-in join, everything else feeds the next phase in the chain.
        if self._spec.unit is not None:
            await ctx.send_message(AnalysisDone(self._spec.unit))
        else:
            await ctx.send_message(PhaseDone(self._spec.name))

    async def _invoke_report(self, prompt: str) -> str:
        """Invoke expecting a report back; retry once if the model produced none.

        Tool-heavy runs sometimes end with empty/junk final text. The retry
        sees the full exploration because both turns share this cycle's
        session - which only works because ``_invoke`` passes it explicitly
        (``Agent.run(session=None)`` is stateless per call).

        Args:
            prompt: The rendered initial or revision prompt.

        Returns:
            Non-empty report text, or a placeholder if both attempts came
            back empty.
        """
        text = await self._invoke(prompt)
        if not text:
            text = await self._invoke(_REPORT_RETRY)
        return text or "(no report produced by the model)"

    async def _invoke(self, prompt: str) -> str:
        """Run one agent turn in the current cycle's session and return its text.

        Args:
            prompt: Prompt for this turn.

        Returns:
            Final text with leaked special tokens stripped; may be empty.
        """
        result = await self._agent.run(prompt, session=self._session)
        return _clean_text(result.text)
