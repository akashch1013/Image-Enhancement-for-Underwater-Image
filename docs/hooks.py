"""MkDocs hook: make the experiment figures and the CA4 report available to the docs.

MkDocs only serves files inside docs/, so before each build the files the pages reference are
copied into docs/assets/generated/ (git-ignored). Unchanged files are skipped so `mkdocs serve`
does not rebuild in a loop.
"""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "docs" / "assets" / "generated"
SOURCES = [*sorted((ROOT / "results" / "figures").glob("*.png")),
           ROOT / "report" / "CA4_Result_Analysis_Report.pdf",
           ROOT / "report" / "CA4_Result_Analysis_Report.docx"]


def on_pre_build(config, **kwargs):
    DEST.mkdir(parents=True, exist_ok=True)
    for src in SOURCES:
        if not src.exists():
            continue
        dst = DEST / src.name
        if dst.exists() and dst.stat().st_size == src.stat().st_size and dst.stat().st_mtime >= src.stat().st_mtime:
            continue
        shutil.copy2(src, dst)
