"""The two manifest fields profile_checks.py's own wiring reads before any
check can run: the version it is written against, and the dump block whose
path it opens.

Split out of profile_checks.py for module size (kb/CLAUDE.md FILE_SIZE)
along the seam the other gates already have -- check_profile_is_known lives
with the readings that branch on the profile (manifest_classes.py) and
check_citation_block_is_shaped with the module that owns the citation
policy (citation_policy_check.py). These two belong to no subject: nothing
downstream reasons about them, the PASS does.

Both are gates rather than checks, and the difference is the failure they
prevent. A check that fails prints a red row; a field the wiring itself
reads and does not find raises out of run_checks() before a single result
exists, and a caller that extends its own list with ours (smoke_test.py,
with no try/except) then aborts with a traceback and no results at all --
strictly worse than a red row, because nothing was checked and nothing said
so (ARTIFACT_SIDE_FAILS_CLOSED). Answering "the version is not mine" or
"there is no dump block" as a row is the whole of their job.
"""
from __future__ import annotations

from pathlib import Path

from manifest_keys import MANIFEST_SCHEMA_VERSION, Key


def check_manifest_version(manifest: dict) -> tuple[bool, str]:
    """The manifest is the one this reader knows how to read.

    Every check below asks the manifest for a key, and a manifest of
    another version answers by omission: a field that moved is read as
    absent, and an absent field is what turns a certification into a row of
    trivially satisfied checks. The recipient runs profile_checks.py
    standalone (AGENT_GUIDE.md) and build_package.py names it as what an
    override build cannot be certified by, so the gate cannot live only in
    the Docker path (smoke_checks.py had the only one).
    """
    declared = manifest.get(Key.SCHEMA_VERSION)
    ok = declared == MANIFEST_SCHEMA_VERSION
    return ok, (f"manifest {Key.SCHEMA_VERSION}={declared!r}, "
                f"этот проверяльщик читает {MANIFEST_SCHEMA_VERSION}"
                + ("" if ok else " — пакет и проверка из разных версий, "
                                 "остальные проверки не запускались"))


def check_dump_block_is_shaped(manifest: dict, artifact_dir: Path) -> tuple[bool, str]:
    """manifest.dump is a mapping, it names a file, and the file is there.

    The twin of citation_policy_check.check_citation_block_is_shaped, one
    key over and with the same rationale: run_checks() builds the path it
    opens out of this block (`artifact_dir / manifest[DUMP][FILE]`), so a
    `dump` field that is absent, a string or a list raises TypeError or
    KeyError out of the pass before any result exists, and a name that
    points at nothing raises FileNotFoundError from inside gzip.open.

    The version gate does not cover it. A current-version manifest with a
    missing or hand-edited dump block is exactly the tampered case an
    unsigned manifest leaves to the recipient, and it is the shape a
    partial extraction produces as well -- the whole certification then
    ends in a traceback rather than in a verdict about the package.

    Whether the file is the one the manifest describes (bytes, sha256) is a
    different question and lives with the reader that hashes it
    (smoke_test.py, dump_integrity.sha256_file); this gate asks only that
    there IS a file to read.
    """
    dump = manifest.get(Key.DUMP)
    if not isinstance(dump, dict):
        return False, (
            f"manifest {Key.DUMP} — {type(dump).__name__} ({dump!r}), а читается как "
            "словарь: какой файл проверять, из такого манифеста не узнать; "
            "пересоберите артефакт текущим сборщиком (остальные проверки не запускались)"
        )
    name = dump.get(Key.FILE)
    if not isinstance(name, str) or not name:
        return False, (
            f"manifest {Key.DUMP}.{Key.FILE}={name!r} — дамп не назван, проверять нечего "
            "(остальные проверки не запускались)"
        )
    path = artifact_dir / name
    if not path.is_file():
        return False, (
            f"{name}: файла нет в {artifact_dir} — манифест описывает дамп, которого "
            "пакет не несёт (остальные проверки не запускались)"
        )
    return True, f"{name}: файл на месте"
