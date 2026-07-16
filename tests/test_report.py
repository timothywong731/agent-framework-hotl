"""Final report: adjudication-log rendering, prompt assembly, executor output."""
import pytest

from conftest import FakeAgent, FakeCtx

from hotl_demo.artifacts import REPOS, ArtifactStore
from hotl_demo.phases import ReportTrigger
from hotl_demo.report import FinalReportExecutor, build_report_prompt, render_adjudication_log

LEDGER = [
    {"id": "q-1", "phase": "discovery", "unit": None, "question": "Scope?",
     "context": "c", "default_assumption": "in scope", "status": "answered",
     "human_answer": "recon in scope", "asked_at": "t"},
    {"id": "q-2", "phase": "enterprise_context", "unit": None, "question": "Region?",
     "context": "c", "default_assumption": "EU", "status": "declined",
     "human_answer": None, "asked_at": "t"},
    {"id": "q-3", "phase": "deep_analysis", "unit": "oms-monolith", "question": "RTO?",
     "context": "c", "default_assumption": "4h", "status": "open",
     "human_answer": None, "asked_at": "t"},
]


def test_render_adjudication_log_covers_all_statuses():
    table = render_adjudication_log(LEDGER)
    assert "| q-1 |" in table and "answered" in table and "recon in scope" in table
    assert "| q-2 |" in table and "declined - default applied: EU" in table
    assert "| q-3 |" in table and "open - default assumption applied: 4h" in table
    assert "deep_analysis[oms-monolith]" in table


def test_render_adjudication_log_empty():
    assert "No questions" in render_adjudication_log([])


def test_render_adjudication_log_sanitizes_newlines_and_pipes():
    entry = {"id": "q-1", "phase": "discovery", "unit": None,
             "question": "Scope?\nRecon | batch too?", "context": "c",
             "default_assumption": "in | scope", "status": "open",
             "human_answer": None, "asked_at": "t"}
    table = render_adjudication_log([entry])
    rows = [ln for ln in table.strip().splitlines() if ln]
    assert len(rows) == 3                              # header + separator + ONE row
    assert all(ln.startswith("|") and ln.endswith("|") for ln in rows)
    assert "Scope? Recon \\| batch too?" in rows[2]    # newline collapsed, pipe escaped
    assert "in \\| scope" in rows[2]
    assert rows[2].count(" | ") + 2 == rows[0].count("|")  # column count intact


def test_build_report_prompt_includes_everything():
    prompt = build_report_prompt("{\"mem\": 1}", {"phase_01_discovery.md": "D-REPORT"}, LEDGER)
    assert "{\"mem\": 1}" in prompt
    assert "phase_01_discovery.md" in prompt and "D-REPORT" in prompt
    assert "q-1" in prompt and "recon in scope" in prompt
    assert "scorecard" in prompt.lower()


@pytest.mark.asyncio
async def test_executor_writes_report_and_yields_path(tmp_path):
    store = ArtifactStore(tmp_path / "run", repos=REPOS)
    store.write_report("phase_01_discovery.md", "D")
    for e in LEDGER:
        store.raise_question(e["phase"], e["unit"], e["question"], e["context"],
                             e["default_assumption"],
                             importance="medium", impact="swings the verdict")
    store.resolve_question("q-1", "answered", "recon in scope")
    store.resolve_question("q-2", "declined", None)
    agent = FakeAgent(["# Readiness Report\nverdict..."])
    ctx = FakeCtx()
    await FinalReportExecutor(store, agent=agent).on_report(ReportTrigger(), ctx)
    text = store.read_report("final_report.md")
    assert text.startswith("# Readiness Report")
    assert "## Adjudication log" in text
    assert "recon in scope" in text            # deterministic table present even though
    assert "q-3" in text                       # the fake agent never mentioned the ledger
    assert ctx.outputs == [str(store.run_dir / "final_report.md")]


def test_adjudication_log_deferred_branch_is_distinct_from_open():
    log = render_adjudication_log([
        {"id": "q-1", "phase": "discovery", "unit": None, "question": "A?",
         "status": "deferred", "human_answer": None, "default_assumption": "da"},
        {"id": "q-2", "phase": "questionnaire", "unit": None, "question": "B?",
         "status": "open", "human_answer": None, "default_assumption": "db"},
    ])
    assert "deferred (over slot limit) - default applied: da" in log
    assert "open - default assumption applied: db" in log
