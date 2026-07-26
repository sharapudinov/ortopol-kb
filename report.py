"""Coverage report: how much of the corpus is actually usable text, by name.

The whole point of this report is that files which failed to extract show
up explicitly. A summary count that silently drops the six broken files
would read as "corpus covered" when it isn't.

`extraction_state()` also defines the mapping from the four fine-grained
`encoding.Category` values down to the three-value `extraction_state`
column used by the Postgres schema (clean / recoded / unreadable): a
no-text-layer scan and a broken-font PDF are both "unreadable" as far as
the database's coverage contract is concerned, even though the report here
keeps them distinguishable for humans.
"""
from __future__ import annotations

import json
from pathlib import Path

from encoding import Category
from pdf_extract import DocumentExtraction

_EXTRACTION_STATE_BY_CATEGORY = {
    Category.CLEAN: "clean",
    Category.MOJIBAKE_RECOVERED: "recoded",
    Category.DEGRADED: "degraded",
    Category.BROKEN: "unreadable",
    Category.NO_TEXT_LAYER: "unreadable",
}

# DEGRADED считается покрытым: его страницы В ИНДЕКСЕ. Потери записаны в note,
# а не спрятаны — но объявлять документ непокрытым из-за 0.3% слов значит
# отчитываться о дыре, которой нет.
_COVERED_CATEGORIES = (Category.CLEAN, Category.MOJIBAKE_RECOVERED, Category.DEGRADED)


def extraction_state(category: Category) -> str:
    return _EXTRACTION_STATE_BY_CATEGORY[category]


def build_report(extractions: list[DocumentExtraction]) -> dict:
    counts = {c.value: 0 for c in Category}
    for extraction in extractions:
        counts[extraction.category.value] += 1

    not_covered = [
        {
            "id": e.stem,
            "filename": e.filename,
            "category": e.category.value,
            "note": e.note,
        }
        for e in extractions
        if e.category not in _COVERED_CATEGORIES
    ]

    return {
        "total_documents": len(extractions),
        "counts": counts,
        "covered": len(extractions) - len(not_covered),
        "not_covered": not_covered,
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# Corpus coverage report",
        "",
        f"Total documents: {report['total_documents']}",
        f"Covered (clean + recoded): {report['covered']}",
        "",
        "| category | count |",
        "|---|---|",
    ]
    for category, count in report["counts"].items():
        lines.append(f"| {category} | {count} |")

    lines += ["", "## Not covered (by name)", ""]
    if not report["not_covered"]:
        lines.append("(none)")
    else:
        lines.append("| id | filename | category | note |")
        lines.append("|---|---|---|---|")
        for entry in report["not_covered"]:
            lines.append(
                f"| {entry['id']} | {entry['filename']} | {entry['category']} | {entry['note']} |"
            )

    return "\n".join(lines) + "\n"


def write_coverage_report(
    extractions: list[DocumentExtraction], output_dir: Path
) -> tuple[Path, Path]:
    report = build_report(extractions)
    json_path = output_dir / "coverage.json"
    md_path = output_dir / "coverage.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path
