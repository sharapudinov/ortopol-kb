"""Artifact-integrity check: re-hashes every runtime/script file the
artifact claims to carry against what actually sits in an extracted
directory. Pure filesystem, no Postgres/Docker/network -- split out of
smoke_checks.py (module size) as its own concern for exactly that reason:
every other check in that module talks to Postgres or ollama, this one
never does.
"""
from __future__ import annotations

from pathlib import Path

from dump_integrity import sha256_file
from manifest_keys import Key

# Operator/runtime byproducts AGENT_GUIDE.md's own documented in-place flow
# creates next to the artifact -- present by design in that mode, not
# tampering. Currently exactly one file: AGENT_GUIDE.md's "Подключение"
# section has the operator `cp .pgenv.example .pgenv` inside the extracted
# directory, and its very next section ("Самопроверка развёртывания")
# instructs running `python3 smoke_test.py` from that SAME directory --
# so by the time check_bundled_files runs, .pgenv legitimately sits beside
# manifest.json. A tuple (not a config file) on purpose: this is a stable,
# reviewed set of names, not something an operator or a manifest should be
# able to grow at will.
OPERATIONAL_ALLOWLIST = (".pgenv",)


def check_bundled_files(extract_dir: Path, manifest: dict, *, pristine: bool = True) -> tuple[bool, str]:
    """Re-hashes every runtime/script file the artifact claims to carry
    (manifest["files"], built by artifact_bundle.bundle_runtime_files) against
    what actually sits in extract_dir. Needs no Postgres/Docker -- pure
    filesystem, so smoke_test.py can run it before (or even without) bringing
    the stack up.

    Nothing previously consumed manifest["files"] at all: a tampered,
    missing or partially-extracted script/compose file passed unnoticed even
    though the manifest recorded exactly enough information to catch it.

    missing/mismatched are always checked, regardless of mode -- a declared
    file that vanished or was tampered with is exactly as real a problem in
    an already-deployed directory as in a fresh extraction. The unaccounted-
    EXTRA-file scan is the part that must vary: `pristine=True` (the default,
    for a --artifact tar.zst freshly extracted into a temp directory nothing
    else has touched) treats any unaccounted file as tampering. `pristine=
    False` (the extract-and-run-in-place default, and explicit
    --artifact-dir) additionally tolerates OPERATIONAL_ALLOWLIST -- an
    operator following AGENT_GUIDE.md's own documented sequence otherwise
    fails their own first self-check on the flagship standalone path. A file
    outside that allowlist still fails in either mode: this narrows the
    false positive, it does not turn the scan off.
    """
    declared = manifest.get(Key.FILES, {})
    missing = [rel for rel in declared if not (extract_dir / rel).is_file()]
    mismatched = [
        rel for rel in declared
        if rel not in missing and sha256_file(extract_dir / rel) != declared[rel]
    ]

    accounted_for = set(declared) | {"manifest.json", manifest[Key.DUMP][Key.FILE]}
    if not pristine:
        accounted_for |= set(OPERATIONAL_ALLOWLIST)
    # __pycache__ is excluded on purpose, not an oversight: in the
    # "extracted-and-run-in-place" mode (see smoke_test.py's own docstring),
    # extract_dir IS smoke_test.py's own directory, and importing its sibling
    # modules (compose_lifecycle, smoke_checks, ...) has already written
    # bytecode caches there by the time this check runs -- Python's own
    # build byproduct, not artifact content the manifest ever claimed to
    # carry.
    present = {
        str(p.relative_to(extract_dir))
        for p in extract_dir.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }
    extra = sorted(present - accounted_for)

    ok = not missing and not mismatched and not extra
    return ok, (
        f"{len(declared)} declared, missing={missing or 'none'}, "
        f"sha256 mismatch={mismatched or 'none'}, extra={extra or 'none'}"
    )
