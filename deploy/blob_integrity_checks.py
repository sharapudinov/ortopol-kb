"""Source-blob integrity checks against corpus.documents.source_blob /
source_sha256, split out of smoke_checks.py (module size) since these three
functions form one self-contained concern. Same (ok, detail) contract as
smoke_checks.py's other predicates -- see that module's docstring.
"""
from __future__ import annotations

import hashlib

from deploy_pathfix import ensure_corpus_importable

ensure_corpus_importable()

from manifest_keys import Key  # noqa: E402
from pg_common import run_sql, scalar  # noqa: E402


def blob_sha256(env: dict, doc_id: str) -> tuple[int, str] | None:
    """Round-trips corpus.documents.source_blob for doc_id: decodes the hex
    bytea psql prints, hashes it locally, and returns (length, sha256) -- the
    same pair build_package.py records in manifest["blob_probe"].

    Called ONCE per smoke run by smoke_test.py, and the result handed to
    check_blob_roundtrip (which compares it against the manifest). Used only
    as a cheap "does the blob exist at all" gate by
    check_blob_corruption_detected, which recomputes its own hash from a
    second hex round trip inside its own probe transaction rather than
    reusing this value -- see that function's docstring for why the reuse
    would make the check circular.
    """
    hex_blob = scalar(
        env, "SELECT encode(source_blob, 'hex') FROM corpus.documents WHERE id = :'doc';",
        variables={"doc": doc_id},
    )
    if not hex_blob:
        return None
    data = bytes.fromhex(hex_blob)
    return len(data), hashlib.sha256(data).hexdigest()


def check_blob_roundtrip(env: dict, manifest: dict, blob_result: tuple[int, str] | None) -> tuple[bool, str]:
    probe = manifest[Key.BLOB_PROBE]
    if blob_result is None:
        return False, f"{probe[Key.DOCUMENT_ID]}: source_blob is NULL"
    length, sha = blob_result
    ok = length == probe[Key.BYTE_LENGTH] and sha == probe[Key.SHA256]
    return ok, (
        f"{probe[Key.DOCUMENT_ID]}: {length} bytes, sha256={sha} "
        f"(manifest: {probe[Key.BYTE_LENGTH]} bytes, {probe[Key.SHA256]})"
    )


# Runs the whole corruption probe inside one transaction that always
# ROLLBACKs, so the disposable smoke database ends up exactly as it started
# regardless of outcome. The blob is read back ONCE (before the UPDATE --
# it is untouched by it, only source_sha256 changes) and hashed in Python
# inside this same function, so the "does the recomputed hash match the
# stored one" comparison below is the actual integrity predicate AGENT_GUIDE.md
# documents (recompute sha256(source_blob), compare to source_sha256), not a
# comparison against a value some earlier, unrelated call happened to
# compute. `-t -A` output is one line per statement, in order: hex blob,
# before-hash, rows-updated, after-hash.
_BLOB_CORRUPTION_PROBE_SQL = """
BEGIN;
SELECT encode(source_blob, 'hex') FROM corpus.documents WHERE id = :'doc';
SELECT source_sha256 FROM corpus.documents WHERE id = :'doc';
WITH corrupted AS (
    UPDATE corpus.documents SET source_sha256 = 'deadbeef' WHERE id = :'doc' RETURNING id
)
SELECT count(*) FROM corrupted;
SELECT source_sha256 FROM corpus.documents WHERE id = :'doc';
ROLLBACK;
"""


def check_blob_corruption_detected(env: dict, manifest: dict, blob_result: tuple[int, str] | None) -> tuple[bool, str]:
    """Corrupts source_sha256 for the probe document inside a transaction
    that is always rolled back, then asserts the SAME integrity predicate
    the deployment's own proof-reading procedure uses (AGENT_GUIDE.md: pull
    source_blob, hash it, compare to source_sha256) actually catches the
    corruption -- not merely that the UPDATE we issued landed. Recomputing
    the hash from the blob fetched inside THIS probe (rather than reusing
    blob_result, which check_blob_roundtrip already validated separately)
    is what makes "after != recomputed" a real detection, not a
    read-your-own-writes tautology: the earlier, tautological version
    compared the post-UPDATE column to the literal 'deadbeef' it had just
    written, which is guaranteed true regardless of whether anything,
    anywhere, would ever actually notice the corruption.
    """
    doc_id = manifest[Key.BLOB_PROBE][Key.DOCUMENT_ID]
    if blob_result is None:
        return False, f"{doc_id}: source_blob is NULL, cannot verify corruption is detected"

    out = run_sql(
        env, _BLOB_CORRUPTION_PROBE_SQL, variables={"doc": doc_id}, extra_args=["-t", "-A"],
    ).stdout
    records = out.splitlines()
    if len(records) != 4:
        return False, f"expected 4 result lines from the transactional probe, got {len(records)}: {records!r}"
    hex_blob, before_sha, update_count, after_sha = records

    if not hex_blob:
        return False, f"{doc_id}: source_blob read back empty inside the probe transaction"
    computed_sha = hashlib.sha256(bytes.fromhex(hex_blob)).hexdigest()

    if computed_sha != before_sha:
        return False, (
            f"{doc_id}: recomputed blob hash {computed_sha} disagreed with stored "
            f"source_sha256 {before_sha!r} BEFORE corruption -- integrity already broken, "
            "independent of anything this probe does"
        )
    if update_count != "1":
        return False, f"{doc_id}: corruption UPDATE touched {update_count} row(s), expected exactly 1"

    detected = computed_sha != after_sha
    if not detected:
        return False, f"{doc_id}: corruption not detected -- recomputed hash still matches stored value after the UPDATE"

    return True, (
        f"{doc_id}: recomputed hash {computed_sha} matched stored source_sha256 before corruption, "
        f"disagreed with stored value {after_sha!r} after (rolled back)"
    )
