"""The preconditions of the artifact-side pass: the ones with no
subject-module of their own, and the ORDER every one of them runs in.

Two gates are declared here -- the manifest version this reader is written
against, and the dump block whose path the pass opens -- and run_gates()
below is the ladder itself, which also calls the gates that DO have a
subject (check_profile_is_known next door in manifest_classes.py,
check_citation_block_is_shaped with the module that owns the citation
policy, check_dump_matches_manifest with the module that hashes the file).
The ladder belongs here rather than in profile_checks.py for the same
reason the two gates do: a precondition is this module's subject, while
that one owns the pass and the checks it feeds.

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

import citation_policy_check
from dump_integrity import check_dump_matches_manifest
from manifest_classes import check_legal_vocabulary_is_known, check_profile_is_known
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


def run_gates(manifest: dict, artifact_dir: Path) -> list[tuple[str, bool, str]]:
    """Every precondition the pass has, in the order they must hold, and
    stopping at the first that fails.

    Here rather than in profile_checks.py because these are this module's
    subject: a field the WIRING itself reads, checked before it is read.
    The order is part of the answer -- each gate is what makes the next one
    meaningful -- and the caller's contract is simply "if any row is False,
    that list is the whole verdict".
    """
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
        # citation visitors, so a field that is not a mapping raises out
        # of the pass before a single result exists. A caller that extends
        # its own list with ours (smoke_test.py) then aborts with a traceback
        # and no results at all -- the failure mode the profile gate above
        # exists to prevent, one key over.
        return [version, known, vocabulary, shaped]
    named = ("манифест несёт блок dump с существующим файлом",
             *check_dump_block_is_shaped(manifest, artifact_dir))
    if not named[1]:
        # The same polarity one key over from the citation block, and the
        # same failure it prevents: the pass builds the path it opens out
        # of this block, so a `dump` that is absent, a string or a list
        # raises before a single result exists, and a name pointing at
        # nothing raises from inside gzip.open. A caller
        # that extends its own list with ours aborts with a traceback and
        # no results at all.
        return [version, known, vocabulary, shaped, named]
    intact = ("дамп — тот, что описан манифестом (размер, sha256)",
              *check_dump_matches_manifest(manifest, artifact_dir))
    if not intact[1]:
        # The last gate, and the one that makes every check below a
        # statement about THIS package: they all read the dump's contents,
        # and contents nobody tied to the manifest's own numbers are the
        # contents of whatever file sits at that path. Checked here rather
        # than only in smoke_test.py, because a recipient runs this module
        # standalone (AGENT_GUIDE.md) and Docker is not what makes an
        # artifact's bytes worth certifying.
        return [version, known, vocabulary, shaped, named, intact]
    return [version, known, vocabulary, shaped, named, intact]
