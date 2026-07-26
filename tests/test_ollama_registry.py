"""Unit tests for deploy/ollama_registry.py: no network, no
Docker. This is the single module manifest_probe.py (build side) and
smoke_checks.py (verify side) both go through for ollama's /api/tags
contract -- see the module docstring for why that used to be duplicated.
"""
from __future__ import annotations

import io
import json
import unittest
from unittest import mock

import _pathfix  # noqa: F401
import _pathfix_deploy  # noqa: F401

import ollama_registry


class TagsUrlTests(unittest.TestCase):
    def test_derives_tags_from_embed_endpoint(self):
        self.assertEqual(ollama_registry.tags_url("http://x/api/embed"), "http://x/api/tags")

    def test_derives_tags_from_a_different_api_path(self):
        self.assertEqual(ollama_registry.tags_url("http://x/api/generate"), "http://x/api/tags")


def _fake_response(payload: dict):
    body = json.dumps(payload).encode()
    cm = mock.MagicMock()
    cm.__enter__.return_value = io.BytesIO(body)
    cm.__exit__.return_value = False
    return cm


def _fake_raw_response(body: bytes):
    cm = mock.MagicMock()
    cm.__enter__.return_value = io.BytesIO(body)
    cm.__exit__.return_value = False
    return cm


class NormalizeModelTagTests(unittest.TestCase):
    """68ea14d3: 'latest' is appended only when the caller's model name has
    no tag of its own -- corpus.embedding_model.model is not schema-
    constrained to be tag-less, and unconditionally appending ':latest'
    would turn a pinned "bge-m3:q8_0" into the never-matching
    "bge-m3:q8_0:latest".
    """

    def test_appends_latest_when_absent(self):
        self.assertEqual(ollama_registry.normalize_model_tag("bge-m3"), "bge-m3:latest")

    def test_leaves_an_explicit_tag_untouched(self):
        self.assertEqual(ollama_registry.normalize_model_tag("bge-m3:q8_0"), "bge-m3:q8_0")


class ServedModelDigestTests(unittest.TestCase):
    def test_returns_digest_and_size_for_matching_model(self):
        payload = {"models": [{"name": "bge-m3:latest", "digest": "abc123", "size": 999}]}
        with mock.patch.object(ollama_registry.urllib.request, "urlopen", return_value=_fake_response(payload)):
            digest, size = ollama_registry.served_model_digest("http://x/api/embed", "bge-m3")
        self.assertEqual(digest, "abc123")
        self.assertEqual(size, 999)

    def test_matches_by_model_field_too(self):
        payload = {"models": [{"model": "bge-m3:latest", "digest": "def456", "size": 1}]}
        with mock.patch.object(ollama_registry.urllib.request, "urlopen", return_value=_fake_response(payload)):
            digest, _ = ollama_registry.served_model_digest("http://x/api/embed", "bge-m3")
        self.assertEqual(digest, "def456")

    def test_pinned_tag_is_not_doubled_with_latest(self):
        # The regression this whole fix closes: a tag-carrying model name
        # must match its OWN tag in /api/tags, not "<name>:<tag>:latest".
        payload = {"models": [{"name": "bge-m3:q8_0", "digest": "abc123", "size": 999}]}
        with mock.patch.object(ollama_registry.urllib.request, "urlopen", return_value=_fake_response(payload)):
            digest, _ = ollama_registry.served_model_digest("http://x/api/embed", "bge-m3:q8_0")
        self.assertEqual(digest, "abc123")

    def test_model_not_found_raises(self):
        payload = {"models": [{"name": "other:latest", "digest": "zzz", "size": 1}]}
        with mock.patch.object(ollama_registry.urllib.request, "urlopen", return_value=_fake_response(payload)):
            with self.assertRaises(RuntimeError):
                ollama_registry.served_model_digest("http://x/api/embed", "bge-m3")

    def test_network_failure_raises_runtime_error(self):
        with mock.patch.object(ollama_registry.urllib.request, "urlopen", side_effect=OSError("boom")):
            with self.assertRaises(RuntimeError):
                ollama_registry.served_model_digest("http://x/api/embed", "bge-m3")

    def test_malformed_json_raises_runtime_error_not_json_decode_error(self):
        # a71ac4d/check_embedding_model_digest's own docstring: every
        # caller's contract is "raises RuntimeError, never anything else" --
        # a transient non-JSON response from ollama used to propagate a bare
        # json.JSONDecodeError straight through smoke_checks.py's (ok,
        # detail) callers instead.
        with mock.patch.object(
            ollama_registry.urllib.request, "urlopen",
            return_value=_fake_raw_response(b"not json at all"),
        ):
            with self.assertRaises(RuntimeError):
                ollama_registry.served_model_digest("http://x/api/embed", "bge-m3")

    def test_non_dict_payload_raises_runtime_error(self):
        # Valid JSON, wrong shape (e.g. a bare array or string) -- .get()
        # on a non-dict would otherwise raise a bare AttributeError.
        with mock.patch.object(
            ollama_registry.urllib.request, "urlopen",
            return_value=_fake_raw_response(b'["not", "a", "dict"]'),
        ):
            with self.assertRaises(RuntimeError):
                ollama_registry.served_model_digest("http://x/api/embed", "bge-m3")

    def test_matching_entry_missing_digest_raises_runtime_error(self):
        # The matching entry exists but lacks the one field this function
        # exists to return -- a bare KeyError used to propagate here.
        payload = {"models": [{"name": "bge-m3:latest", "size": 999}]}
        with mock.patch.object(ollama_registry.urllib.request, "urlopen", return_value=_fake_response(payload)):
            with self.assertRaises(RuntimeError):
                ollama_registry.served_model_digest("http://x/api/embed", "bge-m3")

    def test_matching_entry_with_non_numeric_size_raises_runtime_error(self):
        # int(entry["size"]) used to raise a bare ValueError/TypeError on a
        # malformed size field.
        payload = {"models": [{"name": "bge-m3:latest", "digest": "abc123", "size": "not-a-number"}]}
        with mock.patch.object(ollama_registry.urllib.request, "urlopen", return_value=_fake_response(payload)):
            with self.assertRaises(RuntimeError):
                ollama_registry.served_model_digest("http://x/api/embed", "bge-m3")

    def test_non_dict_entry_in_models_list_is_skipped_not_fatal(self):
        # A stray non-dict item in the "models" list must not crash the
        # scan of the entries that DO match the contract.
        payload = {"models": ["garbage", {"name": "bge-m3:latest", "digest": "abc123", "size": 999}]}
        with mock.patch.object(ollama_registry.urllib.request, "urlopen", return_value=_fake_response(payload)):
            digest, size = ollama_registry.served_model_digest("http://x/api/embed", "bge-m3")
        self.assertEqual(digest, "abc123")
        self.assertEqual(size, 999)


if __name__ == "__main__":
    unittest.main()
