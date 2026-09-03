#!/usr/bin/env python3
"""Packages a deployable snapshot of the knowledge base from the live instance.

Two profiles, `--profile full` (the default) and `--profile public`:

  full    every schema, every byte: both corpus and measurements, full page
          text, every source PDF/djvu blob. NEVER published -- it carries
          other people's copyright in full and exists so the corpus can move
          between the owner's own machines.
  public  schema corpus only, content cut per the legal classification
          stored in the database (corpus.documents.public_distribution, see
          legal_profile.py / public_dump.py): full content for documents
          cleared for redistribution, metadata + page vectors only for the
          rest. Still not published automatically -- the owner approves the
          act of publication; this only makes an artifact that CAN be.

Output: corpus/deploy/kb-<profile>-<date>.tar.zst, self-contained -- a
foreign agent with no access to this repository can extract it and both
deploy and self-verify the result (see AGENT_GUIDE.md, bundled inside).
Contents:
  - 01_dump.sql.gz     the profile's dump (data + DDL; no roles/privileges/
                       tablespaces -- portable by design)
  - manifest.json      the profile, the schemas the dump carries, the legal
                       classification it applied (counts per class, id lists
                       per distribution, the verify query), counts, embedding
                       model (+ its ollama digest), two reference probes
                       (fulltext, vector) recorded FOR THIS PROFILE against
                       this run of the live data, and files{} (sha256 of
                       every bundled runtime/script file) -- see
                       manifest_probe.gather_manifest.
  - docker-compose.yml, init/, ollama-entrypoint.sh, .pgenv.example,
    AGENT_GUIDE.md      the whole runtime this package's own docs describe.
  - smoke_test.py, every deploy module it imports, and copies of the
    repository-root modules those import (under corpus_lib/), so `python3
    smoke_test.py` run from inside the extracted directory works with zero
    repository access. WHICH files those are is artifact_bundle.py's
    DEPLOY_FILES / CORPUS_LIB_FILES, and the manifest's files{} in the
    artifact itself -- not a list here: a second enumeration in prose goes
    stale silently, and this one had, by twenty names.

Never writes inside this repository: even the public artifact carries whole
texts and source files of third-party publications (the ones cleared for it),
and the full artifact carries all of them. --out-dir defaults to
corpus/deploy/, a sibling of theory/, outside git by construction (see
paths.py).
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from deploy_pathfix import ensure_corpus_importable

ensure_corpus_importable()

from paths import default_corpus_dir  # noqa: E402
from pg_common import PostgresUnavailable, load_pgenv  # noqa: E402

from artifact_bundle import bundle_runtime_files, dump_schemas, package  # noqa: E402
from citation_profile import (CitationUnclassified, full_profile_mode,  # noqa: E402
                              resolve_citation_mode)
from dump_integrity import sha256_file  # noqa: E402
from legal_profile import Unclassified  # noqa: E402
from manifest_keys import Key  # noqa: E402
from manifest_contract import CitationMode, PolicySource, Profile  # noqa: E402
from manifest_rows import stamp_dumped_rows  # noqa: E402
from manifest_probe import gather_manifest  # noqa: E402
from public_dump import dump_public  # noqa: E402

OLLAMA_URL = "http://127.0.0.1:5471/api/embed"


def write_dump(profile: str, env: dict, gz_path: Path, *, citation_mode: str):
    """Dispatches to the profile's dump writer and returns the rows it wrote
    (copy_rows.DumpedRows), per schema and per table.

    Here rather than inside artifact_bundle.dump_schemas so the two writers
    stay independent modules: public_dump.py imports artifact_bundle (for the
    shared compression level), and making artifact_bundle choose between them
    would close that loop. A function resolving module globals per call, not
    a dict of function objects built at import time -- the dict captured the
    originals, so it silently ignored any later rebinding of these names
    (which is exactly how the tests substitute a stub for the real pg_dump).
    Both writers take citation_mode, and none of the three signatures
    gives it a default: the full profile applies no cut to the schema's
    CONTENT, but whether the schema exists to be dumped at all is the same
    resolved fact the manifest declares (manifest_contract.schemas_for),
    and a dump that assumed one would quietly disagree with a manifest that
    was told the other.
    """
    if profile == Profile.PUBLIC:
        return dump_public(env, gz_path, citation_mode=citation_mode)
    return dump_schemas(env, gz_path, citation_mode=citation_mode)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pgenv", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--profile", choices=Profile.ALL, default=Profile.FULL,
        help="full (default): everything, never published. public: content cut "
             "per corpus.documents.public_distribution -- see the module docstring",
    )
    parser.add_argument(
        "--ollama-url", default=OLLAMA_URL,
        help="Ollama /api/embed endpoint for the vector probe; override to build "
             "from a host other than the developer's local one",
    )
    parser.add_argument(
        "--policy-override", choices=CitationMode.ALL, default=None, dest="policy_override",
        help="TEST ONLY: force the citation schema's public-artifact mode instead of "
             "reading citation.public_policy. This is NEVER the owner's decision -- it "
             "exists so the packaging/smoke pipeline can be exercised before that decision "
             "is made. Valid only with --profile public. The build records "
             "citation.policy_source='override' IN THE MANIFEST, which profile_checks.py "
             "fails on, and names the file kb-override-<profile>-<date> -- OUTSIDE the "
             "kb-public-* namespace, so no name a consumer globs for can reach it.",
    )
    args = parser.parse_args(argv)

    if args.policy_override and args.profile != Profile.PUBLIC:
        print("--policy-override only applies to --profile public", file=sys.stderr)
        return 2

    corpus_dir = default_corpus_dir()
    pgenv_path = args.pgenv or (corpus_dir / ".pgenv")
    out_dir = args.out_dir or (corpus_dir / "deploy")
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        env = load_pgenv(pgenv_path)
    except PostgresUnavailable as exc:
        print(f"Postgres unavailable: {exc}", file=sys.stderr)
        return 1

    # An override build is named OUTSIDE the profile's namespace --
    # kb-override-<profile>-<date>, not kb-<profile>-override-<date> -- so
    # that no `kb-public-*` selection (a glob, a publish script, a human
    # reading a directory listing) can reach it at all. A suffix inside the
    # namespace was reachable by every one of them, and sorted AFTER the
    # genuine same-date artifact ('o' > '2'), so "the newest public
    # artifact" resolved to the override build every time.
    #
    # The name is still only the outer line: it survives neither a copy
    # under another name nor a recipient who only ever reads manifest.json.
    # What travels INSIDE the package is manifest.citation.policy_source,
    # and profile_checks.py fails on PolicySource.OVERRIDE, so such a build
    # cannot be certified as publishable by any consumer, however it arrived.
    name_tag = args.profile
    if args.policy_override:
        name_tag = f"override-{args.profile}"
        print(
            f"ВНИМАНИЕ: --policy-override={args.policy_override} -- НЕ решение владельца. "
            f"Артефакт назван kb-{name_tag}-..., несёт в манифесте "
            f"citation.policy_source='{PolicySource.OVERRIDE}' и НЕ ПРОЙДЁТ "
            f"deploy/profile_checks.py: публиковать его нельзя, он существует только "
            f"чтобы прогнать конвейер до решения владельца.",
            file=sys.stderr,
        )

    # What the citation schema contributes is decided ONCE per build, here,
    # and the resolved pair is handed to every consumer below (manifest and
    # dump). Two resolutions of the same question can disagree; one cannot
    # -- and that holds for WHOSE decision it was as strongly as for the
    # mode itself, which is why the provenance comes out of the same call
    # rather than being recomputed here from the profile and the flag.
    #
    # The branch is on the profile and it is written here, in the open,
    # because the two profiles ask genuinely different questions: public
    # resolves the owner's POLICY (and refuses when there is none), full
    # applies no policy at all and only reports whether the database has
    # the schema. Folded into one function keyed on `!= PUBLIC`, the second
    # answered a mechanical fact with a value out of the policy vocabulary.
    try:
        citation_mode, policy_source = (
            resolve_citation_mode(env, args.profile, args.policy_override)
            if args.profile == Profile.PUBLIC else full_profile_mode(env))
    except CitationUnclassified as exc:
        # Same refusal as an unclassified document, for the citation schema
        # as a whole (see citation_profile.py): the crawl's own record names
        # third-party titles/abstracts, and shipping or stripping it by
        # default would both be the packager deciding a question only the
        # owner can answer.
        print(f"схема citation не классифицирована для публичного артефакта: {exc}",
              file=sys.stderr)
        return 1

    print(f"профиль {args.profile}; собираю манифест против живой базы...")
    try:
        manifest = gather_manifest(env, args.ollama_url, profile=args.profile,
                                    citation_mode=citation_mode,
                                    policy_source=policy_source)
    except Unclassified as exc:
        # Not a crash: a document with no legal classification is a decision
        # the owner has to make, and the packager must never make it by
        # default. Reported like any other refusal to build.
        print(f"правовая классификация неполна: {exc}", file=sys.stderr)
        return 1

    date_tag = datetime.now(timezone.utc).strftime("%Y%m%d")
    out_path = out_dir / f"kb-{name_tag}-{date_tag}.tar.zst"

    with tempfile.TemporaryDirectory(prefix="kb-build-") as workdir_str:
        workdir = Path(workdir_str)
        print("копирую runtime-файлы (compose, init, скрипты смока) в артефакт...")
        manifest[Key.FILES] = bundle_runtime_files(workdir)
        print(f"дамп схем {', '.join(manifest[Key.SCHEMAS])} (профиль {args.profile})...")
        dump_gz = workdir / "01_dump.sql.gz"
        # Which tables the dump carried and how many rows each got: known
        # only once the dump is written, and known from the writing itself
        # -- rows counted as they streamed into the file, or read back off
        # it (copy_rows.py). Stamped into the keys the manifest already
        # declares, like dump{} below and for the same reason
        # (MANIFEST_DESCRIBES_ARTIFACT). The recipient's gate requires every
        # declared table to be in the file with exactly that many rows, and
        # that equality now holds by construction instead of resting on two
        # readings of a live instance that nothing held still.
        stamp_dumped_rows(manifest,
                          write_dump(args.profile, env, dump_gz,
                                     citation_mode=citation_mode))
        manifest[Key.DUMP] = {
            Key.FILE: dump_gz.name,
            Key.BYTES: dump_gz.stat().st_size,
            Key.SHA256: sha256_file(dump_gz),
        }
        (workdir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
        )
        print(f"упаковываю в {out_path}...")
        package(workdir, out_path)

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"артефакт: {out_path} ({out_path.stat().st_size} байт)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
