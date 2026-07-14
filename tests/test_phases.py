from pathlib import Path

from hotl_demo.phases import (
    PROMPTS_DIR,
    PhaseSpec,
    build_initial_prompt,
    build_phase_specs,
    build_revision_prompt,
    load_pdf_text,
    parse_prompt_file,
    repo_listing,
)

BASE = Path("sample_data")


def _spec(name="discovery", unit=None):
    return PhaseSpec(
        name=name, unit=unit, executor_id=name,
        report_filename=f"phase_x_{name}.md",
        instructions="INSTRUCTIONS-SENTINEL",
        load_sources=lambda: "SOURCES-SENTINEL",
    )


def test_load_pdf_text_extracts_planted_content():
    text = " ".join(load_pdf_text(BASE / "docs" / "02_enterprise_cloud_strategy.pdf").split())
    assert "Microsoft Azure is the approved strategic" in text


def test_repo_listing():
    listing = repo_listing(BASE / "repos" / "oms-monolith")
    assert "s3_uploader.py" in listing and "README.md" in listing


def test_parse_prompt_file_frontmatter_and_body():
    meta, body = parse_prompt_file(PROMPTS_DIR / "deep_analysis.md")
    assert meta["name"] == "deep_analysis"
    assert meta["per_repo"] is True
    assert meta["order"] == 2
    assert meta["report_filename"] == "phase_02_deep_analysis_{unit}.md"
    assert "{{ unit }}" in body                 # body is a Jinja2 template
    assert "---" not in body.split("\n")[0]     # frontmatter stripped


def test_analyzer_specs_render_unit_and_carry_repo_dir():
    by_id = {s.executor_id: s for s in build_phase_specs(BASE)}
    mono = by_id["analyze:oms-monolith"]
    assert "oms-monolith" in mono.instructions          # {{ unit }} rendered
    assert "list_files" in mono.instructions            # exploration directive
    assert mono.repo_dir == BASE / "repos" / "oms-monolith"
    assert by_id["discovery"].repo_dir is None


def test_build_phase_specs_order_units_reports():
    specs = build_phase_specs(BASE)
    assert [(s.name, s.unit) for s in specs] == [
        ("discovery", None),
        ("deep_analysis", "oms-monolith"),
        ("deep_analysis", "oms-batch-recon"),
        ("enterprise_context", None),
        ("questionnaire", None),
    ]
    assert [s.executor_id for s in specs] == [
        "discovery", "analyze:oms-monolith", "analyze:oms-batch-recon",
        "enterprise_context", "questionnaire",
    ]
    assert specs[1].report_filename == "phase_02_deep_analysis_oms-monolith.md"
    assert specs[4].report_filename == "phase_04_questionnaire.md"


def test_spec_sources_contain_the_right_material():
    # normalize whitespace: PDF extraction wraps lines mid-phrase
    def norm(text: str) -> str:
        return " ".join(text.split())

    by_id = {s.executor_id: s for s in build_phase_specs(BASE)}
    discovery = norm(by_id["discovery"].load_sources())
    assert "Microsoft Azure is the approved strategic" in discovery   # strategy pdf
    assert "financial reconciliation" in discovery                    # recon README
    assert "=== config.py ===" not in discovery                       # listings only, not full repo text
    monolith = norm(by_id["analyze:oms-monolith"].load_sources())
    assert "Supporting batch processes" in monolith                   # PDF 1 included
    assert "boto3" not in monolith                                    # repo NOT pre-loaded (rev 4)
    assert "list_files" in monolith                                   # pointer to exploration tools
    ec = norm(by_id["enterprise_context"].load_sources())
    assert "remain in-region" in ec                                   # security pdf
    assert "Supporting batch processes" not in ec                     # PDF 1 excluded
    questionnaire = norm(by_id["questionnaire"].load_sources())
    assert "## 6. Recovery objectives (RTO / RPO)" in questionnaire   # template text


def test_initial_prompt_assembly():
    open_qs = [{"id": "q-1", "phase": "discovery", "unit": None,
                "question": "Scope?", "status": "open"}]
    prompt = build_initial_prompt(_spec(), "SOURCES-SENTINEL", '{"m": 1}', open_qs)
    assert "INSTRUCTIONS-SENTINEL" in prompt
    assert "SOURCES-SENTINEL" in prompt
    assert '{"m": 1}' in prompt
    assert "q-1" in prompt and "Scope?" in prompt        # open ledger for dup suppression
    assert "read_scratchpad" in prompt                   # scratchpad reminder
    assert "raise_question" in prompt and "update_memory" in prompt


def test_initial_prompt_with_empty_ledger_says_so():
    prompt = build_initial_prompt(_spec(), "S", "{}", [])
    assert "No questions raised so far" in prompt


def test_revision_prompt_assembly():
    answers = [{"id": "q-2", "question": "Which cloud?", "human_answer": "AWS, actually",
                "default_assumption": "Azure"}]
    open_qs = [{"id": "q-7", "phase": "enterprise_context", "unit": None,
                "question": "Region?", "status": "open"}]
    prompt = build_revision_prompt(_spec(), "SOURCES-SENTINEL", "{}", open_qs,
                                   answers, "OLD-REPORT")
    assert "AWS, actually" in prompt
    assert "authoritative" in prompt.lower()
    assert "OLD-REPORT" in prompt
    assert "q-2" in prompt
    assert "q-7" in prompt and "Region?" in prompt   # open ledger for dup suppression


def test_revision_prompt_with_empty_ledger_says_so():
    prompt = build_revision_prompt(_spec(), "S", "{}", [], [], "OLD")
    assert "No questions raised so far" in prompt
