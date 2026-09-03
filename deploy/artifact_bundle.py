"""Assembles the tar.zst artifact build_package.py produces: the pg_dump
(streamed straight through gzip, see dump_schemas) and a self-contained
copy of every runtime/script file a foreign agent needs to deploy and
self-verify the package with no access to this repository (see
deploy_pathfix.py and AGENT_GUIDE.md).
"""
from __future__ import annotations

import gzip
import shutil
import subprocess
from pathlib import Path

from citation_dump import live_row_counts
from dump_integrity import sha256_file
from manifest_contract import Profile, schemas_for, ships_citation
from pg_stream import CommandFailed, stream_stdout

# Paths are relative to this file's own directory (deploy/).
DEPLOY_DIR = Path(__file__).resolve().parent
CORPUS_DIR = DEPLOY_DIR.parent

DEPLOY_FILES = [
    "docker-compose.yml",
    "init/00_extensions.sql",
    # kb-pg is built, not pulled (see docker-compose.yml): the recipient
    # needs the Dockerfile to get the same AGE+pgvector pairing.
    "pg/Dockerfile",
    "pg/README.md",
    "ollama-entrypoint.sh",
    ".pgenv.example",
    "AGENT_GUIDE.md",
    "smoke_test.py",
    "smoke_stack.py",
    "smoke_checks.py",
    "vector_probe_check.py",
    "blob_integrity_checks.py",
    "bundled_files_check.py",
    "compose_lifecycle.py",
    "dump_integrity.py",
    "deploy_pathfix.py",
    "ollama_registry.py",
    "drift_probe.py",
    "manifest_contract.py",
    # ... and, beside it, the manifest's own key names and version, which
    # every bundled verifier reads and none of them may spell itself.
    "manifest_keys.py",
    # The probe question itself, which drift_probe.py defaults to.
    "probe_query.py",
    "pg_rank_probe.py",
    # The static profile/legal verification, and the dump reader it uses:
    # bundled so the recipient of an artifact can re-answer "does this
    # package's content match the classification it declares" from the
    # package alone, with no database and no repository. legal_profile.py and
    # public_dump.py are deliberately NOT here -- they read and cut the live
    # corpus, which is the builder's job, not the recipient's.
    "profile_checks.py",
    "dump_scan.py",
    # ... and the corpus half of what it checks: the document/page visitors
    # and the six checks that hold the dump to the legal classification.
    # Split off for module size, like the citation half below -- without it
    # profile_checks.py does not import at all on the recipient's side.
    "corpus_content_checks.py",
    # ... and the manifest-only legal vocabulary its checks reason over,
    # split off for module size: without it profile_checks.py does not
    # import at all on the recipient's side.
    "manifest_classes.py",
    # Static verification of the citation-schema slice of the dump:
    # profile_checks.py's run_checks() calls into it ...
    "citation_content_checks.py",
    # ... beside the module that asks the other question of the facts it
    # collects: does everything that shipped name only what this package
    # carries (work -> document, edge -> work, journal -> neither).
    "citation_cut_checks.py",
    # ... and, beside it, the one citation check that reads no dump byte:
    # whose decision the mode was (manifest.citation.policy_source). Split
    # off for module size and because it answers a different question --
    # without it profile_checks.py does not import at all.
    "citation_policy_check.py",
    # ... and the topology/content classification it checks against. The
    # SAME map the builder's citation_dump.py strips by: a second copy on
    # this side could only agree with the producer by accident, and the
    # check exists precisely to catch the case where it does not.
    "citation_columns.py",
    # Rebuilds citation_graph after the dump restores (see its own comment
    # on why LOAD 'age' cannot be a bare statement inside the DO block).
    "init/02_project_graph.sql",
]
CORPUS_LIB_FILES = [
    "pg_common.py",
    "pg_search.py",
    # smoke_checks.check_citation_projection reuses graph_exists/projection_reading/
    # compare_counts rather than reimplementing the |V|=|work|/|E|=|cites|
    # comparison a second time; that plumbing, and graph_sql()'s
    # AGE-activation contract, live here (see the module's own docstring).
    "pg_graph_common.py",
    # The CLI over it, because AGENT_GUIDE.md documents `pg_graph.py
    # project --check` and the four query subcommands to the recipient ...
    "pg_graph.py",
    # ... and the query layer behind citers/candidates/cocitation/hybrid:
    # the relational two here, the two Cypher ones in pg_graph_cypher.py,
    # each imported by the CLI from the module that owns it. Bundling only
    # the CLI left every documented graph query raising ModuleNotFoundError
    # on a package whose own guide says it can answer them.
    "pg_graph_candidates.py",
    # ... and the two closed vocabularies of the schema those queries name
    # (citation.work.kind here), declared once for both sides of the
    # boundary -- see citation_vocab.py's own docstring for why it is a
    # root module rather than part of citations/, which deliberately does
    # not travel.
    "citation_vocab.py",
    "pg_graph_cocitation.py",
    "pg_graph_cypher.py",
    # ... and the four schema files `pg_graph.py init` applies, in order
    # (data definition, idempotent constraint migrations, AGE projection,
    # journal backfill -- kb/CLAUDE.md FILE_SIZE split pg_schema_citation.sql
    # along those seams). pg_graph_common.SCHEMA_PATHS resolves each NEXT TO
    # THE MODULE, so all four must land in corpus_lib/ or the artifact ships
    # a documented first subcommand that cannot run -- a module-relative
    # dependency the bundle list has to model, since nothing in an import
    # graph mentions it.
    "pg_schema_citation.sql",
    "pg_schema_citation_constraints.sql",
    "pg_schema_citation_graph.sql",
    "pg_schema_citation_backfill.sql",
]


# gzip.open()'s default compresslevel is 9 (max), not the level 6 the gzip
# CLI itself defaults to. Level 9 buys ~1-2% smaller output for 2-4x slower
# deflate, and since compression runs single-threaded and in-line with
# pg_dump's own pipe (the dump streams straight through, see dump_schemas),
# that difference is the whole pipeline's bottleneck on a multi-GB dump. 6
# is the deliberate choice, not an oversight. public_dump.py imports this
# constant rather than picking its own level.
DUMP_COMPRESSLEVEL = 6


def dump_schemas(env: dict, gz_path: Path, citation_mode: str) -> dict[str, int]:
    """The FULL profile's dump: pg_dump of every schema the profile carries
    (manifest_contract.schemas_for -- the same list manifest.json declares),
    streamed straight through gzip into gz_path -- one pass, no
    uncompressed intermediate file. Returns {table: rows} for the citation
    tables it carried, the same answer public_dump.dump_public() gives and
    for the same consumer: manifest.citation.table_rows, which the
    recipient's gate holds the shipped bytes to.

    pg_dump applies no cut, so those numbers are the catalog's whole answer
    (citation_dump.live_row_counts) rather than the classification's: this
    profile writes a table nobody classified too, and a manifest that
    quietly omitted it would leave that table undeclared and unchecked on
    the other side. The dump embeds every source PDF/djvu blob in the corpus (hundreds
    of MB to several GB compressed), so writing it out uncompressed first
    and re-reading it to compress would roughly double both wall-clock and
    peak disk usage for no benefit.

    The public profile's filtered equivalent lives in public_dump.py; both
    are dispatched from build_package.py by --profile, and both stream
    through pg_stream.stream_stdout (see that module on why stderr must be
    drained concurrently).
    """
    schema_args = [f"--schema={name}" for name in schemas_for(Profile.FULL, citation_mode)]
    try:
        with gzip.open(gz_path, "wb", compresslevel=DUMP_COMPRESSLEVEL) as dst:
            stream_stdout(
                ["pg_dump", "--no-owner", "--no-privileges", "--no-tablespaces", *schema_args,
                 # Defensive, on top of --schema already being a whitelist:
                 # citation_graph (AGE's own schema for graph 'citation_graph')
                 # must never be dumped -- apache/age issue #2503, restoring a
                 # dumped ag_graph.graphid breaks Cypher against it (see
                 # pg_schema_citation.sql's header comment). ag_catalog is
                 # excluded for the same reason, belt-and-braces.
                 "--exclude-schema=citation_graph", "--exclude-schema=ag_catalog"],
                env, dst,
            )
    except CommandFailed as exc:
        gz_path.unlink(missing_ok=True)
        raise RuntimeError(str(exc)) from exc
    return live_row_counts(env) if ships_citation(citation_mode) else {}


def bundle_runtime_files(workdir: Path) -> dict[str, str]:
    """Copies DEPLOY_FILES and CORPUS_LIB_FILES into workdir, mirroring the
    layout deploy_pathfix.py expects at extraction time. Returns {relative_path:
    sha256} for manifest["files"], so a tampered or partially-extracted
    artifact is detectable file-by-file, not just as a single whole-dump
    hash.
    """
    files: dict[str, str] = {}
    for rel in DEPLOY_FILES:
        dst = workdir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(DEPLOY_DIR / rel, dst)
        files[rel] = sha256_file(dst)
    lib_dir = workdir / "corpus_lib"
    lib_dir.mkdir(exist_ok=True)
    for rel in CORPUS_LIB_FILES:
        dst = lib_dir / rel
        shutil.copy2(CORPUS_DIR / rel, dst)
        files[f"corpus_lib/{rel}"] = sha256_file(dst)
    return files


def package(workdir: Path, out_path: Path) -> None:
    subprocess.run(
        ["tar", "--zstd", "-cf", str(out_path), "-C", str(workdir), "."],
        check=True,
    )
