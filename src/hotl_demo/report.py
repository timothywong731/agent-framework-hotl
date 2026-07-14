"""Final report: LLM-composed verdict + deterministic adjudication log."""
from __future__ import annotations

from typing import Any

from agent_framework import Agent, Executor, WorkflowContext, handler
from agent_framework.ollama import OllamaChatClient
from typing_extensions import Never

from .artifacts import ArtifactStore
from .phases import ReportTrigger


def _cell(text: str) -> str:
    # LLM-authored strings may contain newlines/pipes that would break the row.
    return " ".join(str(text).split()).replace("|", "\\|")


def render_adjudication_log(ledger: list[dict]) -> str:
    if not ledger:
        return "No questions were raised during this run.\n"
    lines = [
        "| id | phase | question | resolution |",
        "|---|---|---|---|",
    ]
    for e in ledger:
        where = f"{e['phase']}[{e['unit']}]" if e["unit"] else e["phase"]
        if e["status"] == "answered":
            resolution = f"answered: {e['human_answer']}"
        elif e["status"] == "declined":
            resolution = f"declined - default applied: {e['default_assumption']}"
        else:
            resolution = f"open - default assumption applied: {e['default_assumption']}"
        lines.append(f"| {e['id']} | {where} | {_cell(e['question'])} | {_cell(resolution)} |")
    return "\n".join(lines) + "\n"


def build_report_prompt(memory_text: str, reports: dict[str, str], ledger: list[dict]) -> str:
    report_blocks = "\n\n".join(f"--- {name} ---\n{text}" for name, text in reports.items())
    adjudications = "\n".join(
        f"- {e['id']} [{e['status']}] {e['question']}"
        + (f" -> HUMAN: {e['human_answer']}" if e["human_answer"] else
           f" -> default: {e['default_assumption']}")
        for e in ledger
    ) or "none"
    return f"""You are writing the final cloud migration readiness report for
Meridian Retail's Order Management System, synthesizing the phase reports and
shared memory below. Human adjudications are authoritative.

Structure your markdown report exactly as:
# Cloud Migration Readiness Report - Meridian Retail OMS
## Executive summary
## Readiness scorecard
(a table scoring: compute, data, integrations, security, operations - one of
Ready / Ready with conditions / Not ready, each with a one-line reason)
## Migration recommendation
(6R approach, target services, sequencing, prerequisites, key risks)

## HUMAN ADJUDICATIONS
{adjudications}

## SHARED MEMORY
```json
{memory_text}
```

## PHASE REPORTS
{report_blocks}

Write the report now. Do not include an adjudication log section - it is
appended automatically.
"""


class FinalReportExecutor(Executor):
    def __init__(self, store: ArtifactStore, agent: Any | None = None) -> None:
        super().__init__(id="final_report")
        self._store = store
        self._agent = agent or Agent(
            client=OllamaChatClient(),
            name="final_report",
            instructions="You write crisp executive assessment reports.",
        )

    @handler
    async def on_report(self, trig: ReportTrigger, ctx: WorkflowContext[Never, str]) -> None:
        ledger = self._store.read_ledger()
        prompt = build_report_prompt(
            self._store.memory_text(), self._store.read_all_reports(), ledger
        )
        result = await self._agent.run(prompt)
        text = (result.text or "(no text returned by the model)")
        text += "\n\n## Adjudication log\n\n" + render_adjudication_log(ledger)
        path = self._store.write_report("final_report.md", text)
        print("  final_report: written")
        await ctx.yield_output(str(path))
