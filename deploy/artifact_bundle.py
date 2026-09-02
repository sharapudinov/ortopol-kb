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

from dump_integrity import sha256_file
from manifest_contract import Profile, schemas_for
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
    "pg_rank_probe.py",
    # The static profile/legal verification, and the dump reader it uses:
    # bundled so the recipient of an artifact can re-answer "does this
    # package's content match the classification it declares" from the
    # package alone, with no database and no repository. legal_profile.py and
    # public_dump.py are deliberately NOT here -- they read and cut the live
    # corpus, which is the builder's job, not the recipient's.
    "profile_checks.py",
    "dump_scan.py",
    # Static verification of the citation-schema slice of the dump:
    # profile_checks.py's run_checks() calls into it ...
    "citation_content_checks.py",
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
    # smoke_checks.check_citation_projection reuses graph_exists/graph_counts/
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
    "pg_graph_queries.py",
    "pg_graph_cypher.py",
    # ... and the schema `pg_graph.py init` applies. pg_graph_common.
    # SCHEMA_PATH resolves it NEXT TO THE MODULE, so it must land in
    # corpus_lib/ or the artifact ships a documented first subcommand that
    # cannot run -- a module-relative dependency the bundle list has to
    # model, since nothing in an import graph mentions it.
    "pg_schema_citation.sql",
]


# gzip.open()'s default compresslevel is 9 (max), not the level 6 the gzip
# CLI itself defaults to. Level 9 buys ~1-2% smaller output for 2-4x slower
# deflate, and since compression runs single-threaded and in-line with
# pg_dump's own pipe (the dump streams straight through, see dump_schemas),
# that difference is the whole pipeline's bottleneck on a multi-GB dump. 6
# is the deliberate choice, not an oversight. public_dump.py imports this
# constant rather than picking its own level.
DUMP_COMPRESSLEVEL = 6


def dump_schemas(env: dict, gz_path: Path, citation_mode: str) -> None:
    """The FULL profile's dump: pg_dump of every schema the profile carries
    (manifest_contract.schemas_for -- the same list manifest.json declares),
    streamed straight through gzip into gz_path -- one pass, no
    uncompressed intermediate file. The dump embeds every source PDF/djvu blob in the corpus (hundreds
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
