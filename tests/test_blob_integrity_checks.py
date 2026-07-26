"""Unit tests for deploy/blob_integrity_checks.py: no Docker,
no live database. check_blob_corruption_detected's own run_sql call is
stubbed with a canned CompletedProcess; blob_sha256 goes through
pg_common.scalar instead (see BlobSha256Tests) and is stubbed at that
boundary.
"""
from __future__ import annotations

import hashlib
import unittest
from unittest import mock

import _pathfix  # noqa: F401
import _pathfix_deploy  # noqa: F401

import blob_integrity_checks


def _completed(stdout: str):
    return mock.Mock(stdout=stdout, returncode=0)


class BlobSha256Tests(unittest.TestCase):
    """blob_sha256()'s own hex-decode/hash logic (bytes.fromhex + sha256,
    and the empty/NULL guard) -- previously only ever exercised indirectly
    through a hand-built (length, sha) tuple passed to check_blob_roundtrip/
    check_blob_corruption_detected, never through blob_sha256 itself.
    """

    def test_decodes_hex_and_hashes_correctly(self):
        data = b"the quick brown fox"
        hex_blob = data.hex()
        want_sha = hashlib.sha256(data).hexdigest()
        with mock.patch.object(blob_integrity_checks, "scalar", return_value=hex_blob):
            result = blob_integrity_checks.blob_sha256({}, "doc1")
        self.assertEqual(result, (len(data), want_sha))

    def test_empty_hex_returns_none(self):
        with mock.patch.object(blob_integrity_checks, "scalar", return_value=""):
            result = blob_integrity_checks.blob_sha256({}, "doc1")
        self.assertIsNone(result)

    def test_variables_carry_the_document_id(self):
        with mock.patch.object(blob_integrity_checks, "scalar", return_value="ab") as scalar_mock:
            blob_integrity_checks.blob_sha256({}, "1997_sm280")
        _, kwargs = scalar_mock.call_args
        self.assertEqual(kwargs["variables"], {"doc": "1997_sm280"})


class BlobRoundtripTests(unittest.TestCase):
    def test_matches_manifest(self):
        manifest = {"blob_probe": {"document_id": "doc", "byte_length": 100, "sha256": "abc"}}
        ok, _ = blob_integrity_checks.check_blob_roundtrip({}, manifest, (100, "abc"))
        self.assertTrue(ok)

    def test_length_mismatch_fails(self):
        manifest = {"blob_probe": {"document_id": "doc", "byte_length": 100, "sha256": "abc"}}
        ok, _ = blob_integrity_checks.check_blob_roundtrip({}, manifest, (99, "abc"))
        self.assertFalse(ok)

    def test_null_blob_fails(self):
        manifest = {"blob_probe": {"document_id": "doc", "byte_length": 100, "sha256": "abc"}}
        ok, detail = blob_integrity_checks.check_blob_roundtrip({}, manifest, None)
        self.assertFalse(ok)
        self.assertIn("NULL", detail)


class BlobCorruptionDetectedTests(unittest.TestCase):
    """check_blob_corruption_detected's own comparison logic, isolated from
    Docker/run_sql via a stubbed transaction-script result (now 4 lines:
    hex blob, before-hash, rows-updated, after-hash -- the function
    recomputes the hash itself instead of trusting a value computed outside
    this transaction). Covers both the "corruption correctly detected" path
    and every way the negative check could be silently broken.
    """

    BLOB_BYTES = b"pretend this is a pdf/djvu source blob"
    BLOB_HEX = BLOB_BYTES.hex()
    REAL_SHA = hashlib.sha256(BLOB_BYTES).hexdigest()

    def _manifest(self):
        return {"blob_probe": {"document_id": "doc1"}}

    def test_detects_corruption_end_to_end(self):
        # hex blob, before == recomputed hash, exactly 1 row updated,
        # after == the corrupted literal (which necessarily disagrees with
        # the recomputed hash of the untouched blob bytes).
        stub = _completed(f"{self.BLOB_HEX}\n{self.REAL_SHA}\n1\ndeadbeef")
        with mock.patch.object(blob_integrity_checks, "run_sql", return_value=stub):
            ok, detail = blob_integrity_checks.check_blob_corruption_detected(
                {}, self._manifest(), (len(self.BLOB_BYTES), self.REAL_SHA),
            )
        self.assertTrue(ok)
        self.assertIn("rolled back", detail)

    def test_detects_corruption_for_any_mismatched_value_not_just_the_literal_marker(self):
        # The actual integrity predicate is "recomputed hash != stored
        # value", not "stored value == 'deadbeef'" -- any real disagreement
        # must count as detected, since that is what a genuine integrity
        # check would do. This is exactly the case the old, tautological
        # version got backwards (it hard-coded the literal).
        stub = _completed(f"{self.BLOB_HEX}\n{self.REAL_SHA}\n1\n{'b' * 64}")
        with mock.patch.object(blob_integrity_checks, "run_sql", return_value=stub):
            ok, _ = blob_integrity_checks.check_blob_corruption_detected(
                {}, self._manifest(), (len(self.BLOB_BYTES), self.REAL_SHA),
            )
        self.assertTrue(ok)

    def test_fails_if_precondition_hash_already_wrong(self):
        stub = _completed(f"{self.BLOB_HEX}\nsomethingelse\n1\ndeadbeef")
        with mock.patch.object(blob_integrity_checks, "run_sql", return_value=stub):
            ok, detail = blob_integrity_checks.check_blob_corruption_detected(
                {}, self._manifest(), (len(self.BLOB_BYTES), self.REAL_SHA),
            )
        self.assertFalse(ok)
        self.assertIn("BEFORE corruption", detail)

    def test_fails_if_update_touched_wrong_row_count(self):
        stub = _completed(f"{self.BLOB_HEX}\n{self.REAL_SHA}\n0\ndeadbeef")
        with mock.patch.object(blob_integrity_checks, "run_sql", return_value=stub):
            ok, detail = blob_integrity_checks.check_blob_corruption_detected(
                {}, self._manifest(), (len(self.BLOB_BYTES), self.REAL_SHA),
            )
        self.assertFalse(ok)
        self.assertIn("touched 0 row", detail)

    def test_fails_if_update_touched_more_than_one_row(self):
        stub = _completed(f"{self.BLOB_HEX}\n{self.REAL_SHA}\n2\ndeadbeef")
        with mock.patch.object(blob_integrity_checks, "run_sql", return_value=stub):
            ok, detail = blob_integrity_checks.check_blob_corruption_detected(
                {}, self._manifest(), (len(self.BLOB_BYTES), self.REAL_SHA),
            )
        self.assertFalse(ok)

    def test_fails_if_hash_after_corruption_is_unchanged(self):
        # This is exactly the bug the tautological old version could never
        # have caught: corruption silently not applied/visible.
        stub = _completed(f"{self.BLOB_HEX}\n{self.REAL_SHA}\n1\n{self.REAL_SHA}")
        with mock.patch.object(blob_integrity_checks, "run_sql", return_value=stub):
            ok, detail = blob_integrity_checks.check_blob_corruption_detected(
                {}, self._manifest(), (len(self.BLOB_BYTES), self.REAL_SHA),
            )
        self.assertFalse(ok)
        self.assertIn("not detected", detail)

    def test_empty_hex_blob_fails(self):
        stub = _completed(f"\n{self.REAL_SHA}\n1\ndeadbeef")
        with mock.patch.object(blob_integrity_checks, "run_sql", return_value=stub):
            ok, detail = blob_integrity_checks.check_blob_corruption_detected(
                {}, self._manifest(), (len(self.BLOB_BYTES), self.REAL_SHA),
            )
        self.assertFalse(ok)
        self.assertIn("empty", detail)

    def test_null_blob_result_fails_without_touching_db(self):
        with mock.patch.object(blob_integrity_checks, "run_sql") as run_sql_mock:
            ok, detail = blob_integrity_checks.check_blob_corruption_detected({}, self._manifest(), None)
        self.assertFalse(ok)
        run_sql_mock.assert_not_called()

    def test_malformed_transaction_output_fails_cleanly(self):
        stub = _completed("only-one-line")
        with mock.patch.object(blob_integrity_checks, "run_sql", return_value=stub):
            ok, detail = blob_integrity_checks.check_blob_corruption_detected(
                {}, self._manifest(), (len(self.BLOB_BYTES), self.REAL_SHA),
            )
        self.assertFalse(ok)
        self.assertIn("expected 4 result lines", detail)


if __name__ == "__main__":
    unittest.main()
