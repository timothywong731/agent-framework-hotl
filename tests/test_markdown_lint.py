"""Markdown lint gate: keeps the living markdown (README, CLAUDE.md, prompts) clean.

Historical documents under docs/ (spec, plan) are deliberately out of scope.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_markdown_lint():
    """The linted set passes pymarkdown with the repo's .pymarkdown.json config."""
    targets = ["README.md", "src/hotl_demo/prompts"]
    if (ROOT / "CLAUDE.md").exists():
        targets.append("CLAUDE.md")
    proc = subprocess.run(
        [sys.executable, "-m", "pymarkdown", "--config", ".pymarkdown.json", "scan", *targets],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert proc.returncode == 0, f"markdown lint failures:\n{proc.stdout}\n{proc.stderr}"
