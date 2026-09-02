"""The throwaway kb-smoke deploy target smoke_test.py exercises: where its
data comes from (artifact_data_dir, latest_artifact), what environment its
docker compose stack runs under (smoke_env, psql_env), and confirming the
live ortopol-pg/ortopol-ollama instance was untouched by it
(check_live_instance_intact).

Split out of smoke_test.py (module size): these functions describe the
STACK smoke_test.py's main() exercises, not the orchestration of exercising
it (bringing it up, running the checks, tearing it down, reporting).
"""
from __future__ import annotations

import contextlib
import os
import re
import secrets
import subprocess
import tempfile
from pathlib import Path

from deploy_pathfix import ensure_corpus_importable, try_default_corpus_dir

ensure_corpus_importable()

from pg_common import PostgresUnavailable, load_pgenv, run_sql  # noqa: E402

PROJECT = "kb-smoke"
PG_PORT = "5474"
OLLAMA_PORT = "5475"


class ArtifactUnavailable(RuntimeError):
    """No artifact could be resolved (empty/missing deploy dir, no artifact
    given and no repository context to auto-discover one from). A domain
    error, not a process-exit decision: this module describes the STACK,
    not the orchestration (see the module docstring) -- deciding whether an
    unresolved artifact is fatal, and how to report it, belongs to
    smoke_test.py's main(), the only place with the (ok, detail) results
    list every other failure mode reports through.
    """


# An artifact's name in full: kb-<profile>-<8-digit date>.tar.zst, with the
# profile a single word. Matched rather than globbed, and anchored at both
# ends, because everything a glob would let through here is a real name a
# build can produce. `kb-{profile}-*` matched kb-public-override-<date>,
# the name override builds used to carry -- and then sorted it AFTER the
# genuine same-date artifact ('o' > '2'), so "the newest public artifact"
# resolved to the override every time. build_package.py now names such a
# build kb-override-<profile>-<date>, outside the profile's namespace
# altogether; this pattern is the second line rather than the only one, and
# it still holds for the names already lying in a deploy directory: the
# profile group cannot contain a hyphen, so no tag glued to a profile can
# ever answer to that profile's name.
_ARTIFACT_NAME = re.compile(r"\Akb-(?P<profile>[a-z]+)-[0-9]{8}\.tar\.zst\Z")


def latest_artifact(corpus_dir: Path, profile: str | None = None) -> Path:
    """The newest artifact under corpus_dir/deploy/, optionally restricted to
    one profile.

    The profile filter is not cosmetic: artifacts are named
    kb-<profile>-<date>.tar.zst, so an unrestricted sort returns
    kb-public-<date> ahead of kb-full-<date> for the SAME date (lexical
    order, 'p' > 'f') -- auto-discovery would silently smoke-test the public
    package when asked for nothing in particular. Callers say which one they
    mean; smoke_test.py defaults to full.
    """
    deploy_dir = corpus_dir / "deploy"
    candidates = []
    # An absent deploy/ is "no artifact", not a crash: before the first
    # build nothing has created it, and iterdir() raises there where glob()
    # simply yielded nothing. The refusal below is the one a caller can
    # act on -- it names what it was looking for and where.
    try:
        entries = list(deploy_dir.iterdir())
    except FileNotFoundError:
        entries = []
    for path in entries:
        match = _ARTIFACT_NAME.match(path.name)
        if match and (profile is None or match.group("profile") == profile):
            candidates.append(path)
    if not candidates:
        wanted = f"kb-{profile or '<profile>'}-<YYYYMMDD>.tar.zst"
        raise ArtifactUnavailable(f"no {wanted} found under {deploy_dir}")
    return sorted(candidates)[-1]


@contextlib.contextmanager
def artifact_data_dir(artifact: Path | None, artifact_dir: Path | None,
                      profile: str | None = None):
    """Yields (extract_dir, pristine): a directory containing manifest.json
    + 01_dump.sql.gz, and whether it is a fresh extraction nothing else has
    touched yet.

    artifact_dir: already extracted (standalone run pointed at some other
    directory), used as-is, never deleted. NOT pristine -- this is exactly
    the mode a foreign agent points at a live, already-deployed directory.
    artifact: a packed tar.zst, extracted into a fresh temp dir that is
    cleaned up on exit. pristine -- nothing has written into that directory
    except the extraction itself.
    neither: this script's own directory if a manifest.json sits beside it
    (the ordinary standalone case -- extract-and-run-in-place), otherwise
    the newest tar.zst of `profile` under the repository's corpus/deploy/
    (dev-loop convenience, requires paths.py -- see the import above; the
    profile matters, see latest_artifact). The in-place
    case is NOT pristine: AGENT_GUIDE.md's own documented sequence has the
    operator `cp .pgenv.example .pgenv` in this same directory before
    running this script (see bundled_files_check.check_bundled_files's
    pristine parameter for what that changes).

    Explicit Path | None parameters, not an argparse.Namespace: this keeps
    the function callable (and unit-testable) without fabricating a parsed
    CLI namespace -- main() is the only caller and unpacks args.artifact/
    args.artifact_dir itself. Raises ArtifactUnavailable rather than
    exiting the process; see that exception's own docstring.
    """
    if artifact_dir is not None:
        yield artifact_dir, False
        return

    tar_path = artifact
    if tar_path is None:
        here = Path(__file__).resolve().parent
        if (here / "manifest.json").is_file():
            yield here, False
            return
        corpus_dir = try_default_corpus_dir()
        if corpus_dir is None:
            raise ArtifactUnavailable(
                "no artifact given and no repository context to auto-discover one -- "
                "pass --artifact <tar.zst> or --artifact-dir <extracted dir>"
            )
        tar_path = latest_artifact(corpus_dir, profile)

    with tempfile.TemporaryDirectory(prefix="kb-smoke-extract-") as extract_dir_str:
        extract_dir = Path(extract_dir_str)
        # Python's tarfile has no zstd support until 3.14; build_package.py
        # packs with the external tar --zstd, so extraction mirrors that.
        subprocess.run(["tar", "--zstd", "-xf", str(tar_path), "-C", str(extract_dir)], check=True)
        yield extract_dir, True


def smoke_env(artifact_dir: Path, bge_m3_digest: str, embed_model: str) -> dict:
    env = os.environ.copy()
    env.update(
        PGUSER="ortopol",
        PGPASSWORD=secrets.token_urlsafe(24),
        PGDATABASE="ortopol",
        KB_PG_PORT=PG_PORT,
        KB_OLLAMA_PORT=OLLAMA_PORT,
        KB_ARTIFACT_DIR=str(artifact_dir),
        BGE_M3_DIGEST=bge_m3_digest,
        # Sourced from the artifact's own manifest, not docker-compose.yml's
        # "bge-m3" default: a kb-smoke run against a package built with a
        # differently-named model must pull/grep for THAT name, or
        # kb-ollama's healthcheck never goes green.
        KB_EMBED_MODEL=embed_model,
    )
    return env


def psql_env(compose_env: dict) -> dict:
    env = compose_env.copy()
    env.update(PGHOST="127.0.0.1", PGPORT=PG_PORT)
    return env


def check_live_instance_intact(live_pgenv: Path | None) -> tuple[bool | None, str]:
    if live_pgenv is None:
        return None, "--live-pgenv не указан -- пропущено (самодостаточный прогон без доступа к репозиторию)"
    try:
        env = load_pgenv(live_pgenv)
    except PostgresUnavailable as exc:
        return False, f"could not load {live_pgenv}: {exc}"
    try:
        out = run_sql(env, "SELECT 1;", extra_args=["-t", "-A"]).stdout.strip()
    except RuntimeError as exc:
        return False, f"live instance unreachable after smoke run: {exc}"
    return out == "1", f"SELECT 1 on live instance -> {out!r}"
