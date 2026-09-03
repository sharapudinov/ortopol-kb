#!/usr/bin/env python3
"""Static verification that an artifact's CONTENT matches the profile and
the legal classification its own manifest declares.

Static means: no Docker, no Postgres, no network -- the dump's bytes are
read directly (dump_scan.py) and compared against manifest.json. That is
deliberately the whole point. The question "did the public artifact ship a
paper it may not ship" must be answerable before the artifact is restored
anywhere, by anyone who has only the file, and it must be answered from the
shipped bytes rather than from the packager's intentions.

This module owns the single streaming pass over the dump and the ORDER the
checks run in; the checks themselves live one module per subject, each
contributing row visitors to that same pass:

  profile/manifest agreement   here: the schemas the profile+mode rule
                               requires (manifest_contract.schemas_for) are
                               the ones the manifest declares AND the ones
                               the dump contains (public: no measurements
                               schema at all, not merely no measurements
                               rows) -- three-way, since the dump and the
                               declaration are both the producer's word;
                               read off the same pass, since schema names
                               and COPY headers are the same lines
  manifest version matches     manifest_gates.check_manifest_version --
                               manifest.schema_version is the version this
                               reader knows; anything else stops the pass
                               instead of reading missing fields as
                               satisfied checks
  profile is in the vocabulary manifest_classes.check_profile_is_known --
                               every check below picks its strictness off
                               it, so an unknown one stops the pass
  legal vocabulary is known    manifest_classes.
                               check_legal_vocabulary_is_known -- the two
                               distribution lists the legal checks derive
                               their whole expectation from name only classes
                               this profile may carry, and neither is empty
  citation block is a block    citation_policy_check.
                               check_citation_block_is_shaped -- the pass
                               reads manifest.citation.mode before any
                               check runs, so a field that is not a mapping
                               has to stop the pass rather than raise
                               through it
  dump block names a file      manifest_gates.
                               check_dump_block_is_shaped -- the same, one
                               key over: the pass builds the path it opens
                               out of manifest.dump, so a block naming
                               nothing (or a file the package does not
                               carry) stops the pass with a row, not a
                               traceback
  every column is classified   column_class_checks.py: each COPY block of
                               schema corpus/citation carries only columns
                               the bundled classification maps name -- the
                               producer-side polarity (an unnamed column
                               stops the build) asked of the finished file,
                               which is the side an unsigned manifest
                               leaves to the recipient
  every table is declared      table_rows_check.py: each classified
                               schema's manifest block names every table of
                               it the dump carries, with exactly that many
                               rows, and carries every table it names --
                               both schemas from one engine, because a
                               check that finds no COPY block cannot
                               otherwise tell "cut" from "never shipped"
  corpus content holds         corpus_content_checks.py (module size):
                               classification complete, excluded left no
                               trace, metadata-only stripped, full-text
                               intact, every page embedded, no generated
                               column in the dump
  sequences arrive moved       sequence_checks.py: every sequence-owning
                               column whose table shipped rows is followed
                               by a setval, and it comes after the block.
                               The one breach that is silent at restore
                               time and lands on the recipient's next insert
  legal vocabulary             which ids the manifest says are carried, and
                               in which shape, is manifest_classes.py
                               (module size): a manifest-only reading, with
                               no dump byte in it
  citation policy is owner's   manifest.citation.policy_source == "owner":
                               an artifact whose citation mode was forced
                               with --policy-override fails here rather
                               than being certified as publishable, and one
                               whose manifest names no policy at all fails
                               too (citation_policy_check.py)
  citation content holds       citation_content_checks.py (module size)
                               owns the citation row visitors on this pass
                               and the hunt for content a mode was not
                               allowed to ship
  citation cut holds           citation_cut_checks.py (module size): the
                               counts and the kind census against the
                               manifest, and the three checks that hold the
                               citation cut to the DOCUMENT cut -- no work
                               row names a document this dump does not
                               carry, no edge a work it does not carry, and
                               no journal row a document the artifact drops

Same (ok, detail) contract as smoke_checks.py, so smoke_test.py can list
these beside its live checks; runnable standalone as well:

    python3 profile_checks.py                     # artifact beside this file
    python3 profile_checks.py --artifact-dir DIR
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import NamedTuple

import citation_content_checks
import citation_cut_checks
import citation_policy_check
import column_class_checks
import corpus_content_checks
import dump_scan
import sequence_checks
import table_rows_check
from manifest_classes import check_legal_vocabulary_is_known, check_profile_is_known
from manifest_gates import check_dump_block_is_shaped, check_manifest_version
from manifest_contract import required_schemas
from manifest_keys import Key


def check_schemas(contents: dump_scan.DumpContents, manifest: dict) -> tuple[bool, str]:
    """The schemas the dump names, the ones the manifest declares and the
    ones the profile+mode RULE requires are one and the same set.

    Three-way, because two of them are the same side: manifest.json is not
    signed and this module travels inside the artifact precisely so a
    recipient can certify it without trusting the producer
    (ARTIFACT_SIDE_FAILS_CLOSED). Compared with the declaration alone, a
    build that resolved "which schemas does this profile ship" wrongly
    agrees with itself -- one decision wrote both the dump and the claim --
    and the recipient certifies the mistake. The rule is
    manifest_contract.schemas_for(), the authority the dumpers and the
    manifest are written by, re-derived from the two manifest fields the
    gates above have already validated the shape of.

    Read off the pass _visit() already made, not off a pass of its own: the
    schema names and the COPY headers are the same lines, and the full
    profile's dump carries every source PDF as hex, so a second inflate of
    it is the most expensive way to re-answer a question already answered.
    """
    declared = set(manifest.get(Key.SCHEMAS, []))
    present = contents.schemas
    required, problem = required_schemas(manifest)
    if required is None:
        return False, (f"по манифесту нельзя вывести состав схем: {problem}; "
                       f"дамп несёт {sorted(present)}, манифест объявляет {sorted(declared)}")
    ok = present == declared == set(required)
    return ok, (f"dump carries {sorted(present)}, manifest declares {sorted(declared)}, "
                f"profile+mode rule requires {sorted(required)}")


class DumpFacts(NamedTuple):
    """What the pass collected, one record per subject.

    A pair rather than one merged dict of string keys. Every check below
    reads its input by NAME, so a fact that never arrived -- a visitor that
    did not fire, a key renamed on one side of a module split, a bundled
    checker older than the package it travels in -- raises where it is read
    instead of resolving to an empty set. An empty set is what makes
    "absent from the dump: none" and "leaked 0 row(s)" true, i.e. the shape
    in which this package certifies nothing and says [OK].
    """

    corpus: corpus_content_checks.CorpusFacts
    citation: citation_content_checks.CitationFacts


def _visit(dump_path: Path, manifest: dict) -> tuple[dump_scan.DumpContents, DumpFacts]:
    """One streaming pass over the dump, with every subject's visitors on it.

    Called only after run_checks() has gated the fields the wiring itself
    reads: manifest.citation must be a mapping before its mode can be handed
    to the citation visitors, and manifest.dump must name a file that exists
    before this path can be opened.
    """
    row_visitors: dict = {}
    facts = DumpFacts(
        corpus=corpus_content_checks.attach_visitors(row_visitors),
        citation=citation_content_checks.attach_visitors(
            row_visitors, manifest.get(Key.CITATION, {}).get(Key.CITATION_MODE)),
    )
    return dump_scan.scan(dump_path, row_visitors), facts


def run_checks(artifact_dir: Path) -> list[tuple[str, bool, str]]:
    manifest = json.loads((artifact_dir / "manifest.json").read_text())
    version = ("версия манифеста = версия проверяльщика", *check_manifest_version(manifest))
    if not version[1]:
        # Nothing below is meaningful against a manifest this reader cannot
        # read, and a list of passes underneath a failed gate reads as a
        # certification. The gate is the whole answer.
        return [version]
    known = ("манифест называет известный профиль", *check_profile_is_known(manifest))
    if not known[1]:
        # Every check below picks its strictness off this string; read as
        # anything but a declared profile they all take the lenient branch
        # at once, and a column of passes underneath is a certification of
        # nothing.
        return [version, known]
    vocabulary = ("правовой словарь манифеста известен",
                  *check_legal_vocabulary_is_known(manifest))
    if not vocabulary[1]:
        # The same polarity one field further in, and the field the legal
        # checks below derive their whole expectation from: an unknown or
        # over-broad distribution shrinks what they look for, and a shrunken
        # expectation is satisfied by an artifact nobody verified.
        return [version, known, vocabulary]
    shaped = ("манифест несёт блок citation словарём",
              *citation_policy_check.check_citation_block_is_shaped(manifest))
    if not shaped[1]:
        # The same polarity one field further in, and this one is not merely
        # about strictness: _visit() reads manifest.citation.mode to wire the
        # citation visitors, so a field that is not a mapping raises out of
        # run_checks() before a single result exists. A caller that extends
        # its own list with ours (smoke_test.py) then aborts with a traceback
        # and no results at all -- the failure mode the profile gate above
        # exists to prevent, one key over.
        return [version, known, vocabulary, shaped]
    named = ("манифест несёт блок dump с существующим файлом",
             *check_dump_block_is_shaped(manifest, artifact_dir))
    if not named[1]:
        # The same polarity one key over from the citation block, and the
        # same failure it prevents: the line below builds the path it opens
        # out of this block, so a `dump` that is absent, a string or a list
        # raises out of run_checks() before a single result exists, and a
        # name pointing at nothing raises from inside gzip.open. A caller
        # that extends its own list with ours aborts with a traceback and
        # no results at all.
        return [version, known, vocabulary, shaped, named]
    dump_path = artifact_dir / manifest[Key.DUMP][Key.FILE]
    contents, facts = _visit(dump_path, manifest)
    scans = contents.tables
    profile = manifest.get(Key.PROFILE)
    return [
        version,
        known,
        vocabulary,
        shaped,
        named,
        (f"профиль {profile!r}: схемы дампа = манифест", *check_schemas(contents, manifest)),
        ("каждая колонка дампа классифицирована",
         *column_class_checks.check_columns_are_classified(scans)),
        *[(f"{schema}: каждая заявленная таблица приехала целиком",
           *table_rows_check.check_every_declared_table_shipped(manifest, scans, schema))
          for schema in table_rows_check.DECLARED_SCHEMAS],
        ("правовая классификация полна",
         *corpus_content_checks.check_classification_complete(manifest, scans)),
        ("excluded: ни строки документа, ни страниц",
         *corpus_content_checks.check_excluded_absent(manifest, facts)),
        ("metadata-only: ни блоба, ни текста",
         *corpus_content_checks.check_metadata_only_stripped(manifest, facts)),
        ("full-text: блоб и текст на месте",
         *corpus_content_checks.check_full_content_intact(manifest, facts)),
        ("векторы у всех страниц",
         *corpus_content_checks.check_pages_embedded(manifest, scans, facts)),
        ("нет generated-колонок в дампе",
         *corpus_content_checks.check_no_generated_columns(scans)),
        ("последовательности переставлены после своих строк",
         *sequence_checks.check_sequences_are_repositioned(contents)),
        ("citation: режим — решение владельца, не --policy-override",
         *citation_policy_check.check_policy_is_the_owners(manifest)),
        ("citation: схема/счётчики совпадают с манифестом",
         *citation_cut_checks.check_citation_schema_matches_mode(manifest, scans)),
        ("citation: content-колонки вырезаны вне full-skeleton",
         *citation_content_checks.check_content_is_stripped(manifest, facts)),
        ("citation.work ссылается только на документы этого пакета",
         *citation_cut_checks.check_work_documents_are_in_the_dump(manifest, facts)),
        ("citation.cites ссылается только на узлы этого пакета",
         *citation_cut_checks.check_edges_reference_shipped_works(manifest, facts)),
        ("citation.work: перепись по kind совпадает с манифестом",
         *citation_cut_checks.check_kind_census_matches_manifest(manifest, facts)),
        ("citation.crawl_step не называет вырезанных документов",
         *citation_cut_checks.check_journal_names_nothing_cut(manifest, facts)),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=Path(__file__).resolve().parent,
                        help="extracted artifact directory (default: this script's own)")
    args = parser.parse_args(argv)

    manifest_path = args.artifact_dir / "manifest.json"
    if not manifest_path.is_file():
        print(f"no manifest.json under {args.artifact_dir}", file=sys.stderr)
        return 2

    all_ok = True
    for name, ok, detail in run_checks(args.artifact_dir):
        print(f"[{'OK' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
        all_ok = all_ok and ok
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
