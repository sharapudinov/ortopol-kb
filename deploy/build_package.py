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
  - smoke_test.py and the deploy modules it imports (smoke_stack.py,
    smoke_checks.py, vector_probe_check.py, blob_integrity_checks.py,
    bundled_files_check.py, manifest_contract.py, compose_lifecycle.py,
    dump_integrity.py, deploy_pathfix.py, ollama_registry.py,
    drift_probe.py, profile_checks.py, dump_scan.py,
    pg_rank_probe.py -- deploy-only despite living beside
    pg_search/pg_common, see that module's own docstring), plus
    corpus_lib/{pg_common,pg_search}.py -- copies of the deploy scripts and
    the two genuinely shared repository-root modules they import, so
    `python3 smoke_test.py` run from inside the extracted directory works
    with zero repository access -- see artifact_bundle.py's DEPLOY_FILES/
    CORPUS_LIB_FILES for the definitive, machine-checked list
    (test_artifact_bundle.py's import-closure test fails if this comment
    or that list ever drifts from what smoke_test.py actually imports).

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
from citation_profile import CitationUnclassified  # noqa: E402
from dump_integrity import sha256_file  # noqa: E402
from legal_profile import Unclassified  # noqa: E402
from manifest_contract import CitationMode, Key, Profile  # noqa: E402
from manifest_probe import gather_manifest  # noqa: E402
from public_dump import dump_public  # noqa: E402

OLLAMA_URL = "http://127.0.0.1:5471/api/embed"


def write_dump(profile: str, env: dict, gz_path: Path, citation_mode_override: str | None = None) -> None:
    """Dispatches to the profile's dump writer, both (env, gz_path) -> None.

    Here rather than inside artifact_bundle.dump_schemas so the two writers
    stay independent modules: public_dump.py imports artifact_bundle (for the
    shared compression level), and making artifact_bundle choose between them
    would close that loop. A function resolving module globals per call, not
    a dict of function objects built at import time -- the dict captured the
    originals, so it silently ignored any later rebinding of these names
    (which is exactly how the tests substitute a stub for the real pg_dump).
    citation_mode_override is meaningless for the full writer (dump_schemas
    always carries the whole citation schema, like every other table) and is
    passed only to dump_public.
    """
    if profile == Profile.PUBLIC:
        dump_public(env, gz_path, citation_mode_override=citation_mode_override)
    else:
        dump_schemas(env, gz_path)


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
             "is made. Valid only with --profile public; the output artifact is named "
             "kb-public-override-<date>, never kb-public-<date>, so an override build can "
             "never be mistaken for one the owner classified.",
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

    profile_tag = args.profile
    if args.policy_override:
        profile_tag = f"{args.profile}-override"
        print(
            f"ВНИМАНИЕ: --policy-override={args.policy_override} -- override, not the "
            f"owner's decision; артефакт назван kb-{profile_tag}-..., не kb-public-...",
            file=sys.stderr,
        )

    print(f"профиль {args.profile}; собираю манифест против живой базы...")
    try:
        manifest = gather_manifest(env, args.ollama_url, profile=args.profile,
                                    citation_mode_override=args.policy_override)
    except Unclassified as exc:
        # Not a crash: a document with no legal classification is a decision
        # the owner has to make, and the packager must never make it by
        # default. Reported like any other refusal to build.
        print(f"правовая классификация неполна: {exc}", file=sys.stderr)
        return 1
    except CitationUnclassified as exc:
        # Same refusal, for the citation schema as a whole (see
        # citation_profile.py): the crawl's own record names third-party
        # titles/abstracts, and shipping or stripping it by default would
        # both be the packager deciding a question only the owner can answer.
        print(f"схема citation не классифицирована для публичного артефакта: {exc}",
              file=sys.stderr)
        return 1

    date_tag = datetime.now(timezone.utc).strftime("%Y%m%d")
    out_path = out_dir / f"kb-{profile_tag}-{date_tag}.tar.zst"

    with tempfile.TemporaryDirectory(prefix="kb-build-") as workdir_str:
        workdir = Path(workdir_str)
        print("копирую runtime-файлы (compose, init, скрипты смока) в артефакт...")
        manifest[Key.FILES] = bundle_runtime_files(workdir)
        print(f"дамп схем {', '.join(manifest[Key.SCHEMAS])} (профиль {args.profile})...")
        dump_gz = workdir / "01_dump.sql.gz"
        write_dump(args.profile, env, dump_gz, citation_mode_override=args.policy_override)
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
