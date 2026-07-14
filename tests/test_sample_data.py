"""Guardrail: every planted gap/conflict survives in the committed corpus."""
from pathlib import Path

from pypdf import PdfReader

DOCS = Path("sample_data/docs")
REPOS = Path("sample_data/repos")


def _pdf_text(name: str) -> str:
    raw = "\n".join(page.extract_text() for page in PdfReader(DOCS / name).pages)
    return " ".join(raw.split())  # normalize: extraction wraps lines mid-phrase


def test_pdfs_exist_and_extract():
    assert sorted(p.name for p in DOCS.glob("*.pdf")) == [
        "01_oms_application_architecture.pdf",
        "02_enterprise_cloud_strategy.pdf",
        "03_cybersecurity_standards.pdf",
    ]


def test_planted_gaps_in_pdfs():
    arch = _pdf_text("01_oms_application_architecture.pdf")
    strategy = _pdf_text("02_enterprise_cloud_strategy.pdf")
    security = _pdf_text("03_cybersecurity_standards.pdf")
    # planting 1 fuel: vague batch mention, recon absent
    assert "Supporting batch processes" in arch
    assert "reconciliation" not in arch.lower()
    # planting 3: no RTO/RPO numbers anywhere in the corpus
    assert "RTO" not in arch and "RPO" not in arch
    # planting 2: Azure mandate
    assert "Microsoft Azure is the approved strategic" in strategy
    # planting 5: MQ retirement date TBC
    assert "retirement (date TBC)" in strategy
    # planting 4: residency region unspecified
    assert "remain in-region" in security
    # planting 6 fuel: Oracle named in arch doc
    assert "Oracle Database 11g" in arch


def test_planted_conflicts_in_repos():
    s3 = (REPOS / "oms-monolith" / "s3_uploader.py").read_text(encoding="utf-8")
    assert "boto3" in s3  # planting 2: AWS in code vs Azure mandate
    cfg = (REPOS / "oms-batch-recon" / "config.py").read_text(encoding="utf-8")
    assert 'ORACLE_PASSWORD = "Rec0n#2011!"' in cfg  # planting 7: hardcoded cred
    readme = (REPOS / "oms-batch-recon" / "README.md").read_text(encoding="utf-8")
    assert "financial reconciliation" in readme  # planting 1: undocumented scope
    fs = (REPOS / "oms-monolith" / "file_store.py").read_text(encoding="utf-8")
    assert "/mnt/nfs/orders" in fs
