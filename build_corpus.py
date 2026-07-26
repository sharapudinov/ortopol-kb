"""Extract every PDF under theory/iis/ into plain text, outside git.

Usage:
    python3 build_corpus.py [--pdf-dir DIR] [--output-dir DIR]

Storage-agnostic on purpose: this step only writes plain files (text/*.txt,
manifest.json, coverage.{json,md}) to --output-dir. Loading the result into
Postgres is a separate step (pg_load.py) so extraction can be re-run,
inspected, or fed to a different backend without re-running pdftotext.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from paths import default_corpus_dir, default_pdf_dir
from pdf_extract import DocumentExtraction, extract_document
from report import write_coverage_report


def build_corpus(pdf_dir: Path, output_dir: Path) -> list[DocumentExtraction]:
    text_dir = output_dir / "text"
    text_dir.mkdir(parents=True, exist_ok=True)

    extractions = []
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        extraction = extract_document(pdf_path)
        extractions.append(extraction)
        if extraction.pages:
            text_path = text_dir / f"{extraction.stem}.txt"
            text_path.write_text("\f".join(extraction.pages), encoding="utf-8")

    return extractions


def write_manifest(extractions: list[DocumentExtraction], output_dir: Path) -> Path:
    manifest = {
        "documents": [
            {
                "id": e.stem,
                "filename": e.filename,
                "category": e.category.value,
                "chars_extracted": e.chars_extracted,
                "pages_count": len(e.pages),
                "text_path": f"text/{e.stem}.txt" if e.pages else None,
                "note": e.note,
            }
            for e in extractions
        ]
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf-dir", type=Path, default=default_pdf_dir())
    parser.add_argument("--output-dir", type=Path, default=default_corpus_dir())
    args = parser.parse_args(argv)

    extractions = build_corpus(args.pdf_dir, args.output_dir)
    write_manifest(extractions, args.output_dir)
    _, md_path = write_coverage_report(extractions, args.output_dir)

    print(f"Extracted {len(extractions)} documents from {args.pdf_dir}")
    print(f"Manifest: {args.output_dir / 'manifest.json'}")
    print(f"Coverage report: {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
