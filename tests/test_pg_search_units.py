"""Unit tests for pg_search.py: no live database, no network.
resolve_model/embed_with are exercised with pg_common.row_or_none and
urllib mocked at the module boundary, the same pattern test_pg_common.py
and test_ollama_registry.py use for their own collaborators. Integration
coverage against a real instance lives in test_pg_semantic.py.
"""
from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import _pathfix  # noqa: F401

import paths
import pg_search


def _fake_response(payload: dict):
    body = json.dumps(payload).encode()
    cm = mock.MagicMock()
    cm.__enter__.return_value = io.BytesIO(body)
    cm.__exit__.return_value = False
    return cm


class ResolveModelTests(unittest.TestCase):
    def test_returns_model_and_int_dims(self):
        with mock.patch.object(pg_search, "row_or_none", return_value=["bge-m3", "1024"]):
            result = pg_search.resolve_model({})
        self.assertEqual(result, ("bge-m3", 1024))

    def test_empty_table_returns_none(self):
        with mock.patch.object(pg_search, "row_or_none", return_value=None):
            self.assertIsNone(pg_search.resolve_model({}))


class EmbedWithTests(unittest.TestCase):
    """embed_with() is the primitive 3860966a split out of embed_query() so
    a caller embedding many queries against the same instance (drift_probe.
    measure_drift) can resolve (model, dims) once and call this n times.
    """

    def test_returns_json_encoded_vector(self):
        payload = {"embeddings": [[0.1, 0.2]]}
        with mock.patch.object(pg_search.urllib.request, "urlopen", return_value=_fake_response(payload)):
            vec = pg_search.embed_with("bge-m3", 2, "запрос")
        self.assertEqual(json.loads(vec), [0.1, 0.2])

    def test_dims_mismatch_raises_value_error(self):
        payload = {"embeddings": [[0.1, 0.2, 0.3]]}
        with mock.patch.object(pg_search.urllib.request, "urlopen", return_value=_fake_response(payload)):
            with self.assertRaises(ValueError):
                pg_search.embed_with("bge-m3", 2, "запрос")

    def test_transport_failure_returns_none(self):
        with mock.patch.object(pg_search.urllib.request, "urlopen", side_effect=OSError("boom")):
            self.assertIsNone(pg_search.embed_with("bge-m3", 2, "запрос"))


class EmbedQueryComposesResolveAndEmbedTests(unittest.TestCase):
    """embed_query() is now a thin composition of resolve_model() +
    embed_with() -- must still work unchanged for one-off callers that
    don't need the split.
    """

    def test_delegates_to_resolve_model_and_embed_with(self):
        with mock.patch.object(pg_search, "resolve_model", return_value=("bge-m3", 3)) as resolve_mock, \
             mock.patch.object(pg_search, "embed_with", return_value="[0.1,0.2,0.3]") as embed_mock:
            vec = pg_search.embed_query("запрос", {}, ollama_url="http://x")
        self.assertEqual(vec, "[0.1,0.2,0.3]")
        resolve_mock.assert_called_once_with({})
        embed_mock.assert_called_once_with("bge-m3", 3, "запрос", "http://x")

    def test_no_model_row_returns_none_without_calling_embed_with(self):
        with mock.patch.object(pg_search, "resolve_model", return_value=None), \
             mock.patch.object(pg_search, "embed_with") as embed_mock:
            self.assertIsNone(pg_search.embed_query("запрос", {}))
        embed_mock.assert_not_called()


class MainCorpusDirDefaultTests(unittest.TestCase):
    """d809a66d: main() resolves --corpus-dir's default with a LOCAL
    try/except (ImportError, RuntimeError) directly against paths.py,
    instead of reaching sideways into deploy/ for
    deploy_pathfix.try_default_corpus_dir -- the general-purpose corpus
    library must not depend on its own deploy/ subpackage (see the module
    docstring's note on the dependency direction).
    """

    def test_no_repo_context_and_no_explicit_args_reports_and_returns_1(self):
        # paths.data_root() (default_corpus_dir()'s own collaborator)
        # raising RuntimeError is exactly the "plain clone, no surrounding
        # theory/iis/ data tree" case -- main() must not crash, it must
        # fall back to requiring an explicit --pgenv/--corpus-dir.
        with mock.patch.object(paths, "data_root", side_effect=RuntimeError("no repo root")):
            with mock.patch("sys.stderr", new_callable=io.StringIO) as stderr:
                exit_code = pg_search.main(["query"])
        self.assertEqual(exit_code, 1)
        self.assertIn("no repository context", stderr.getvalue())

    def test_repo_context_available_resolves_pgenv_under_corpus_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus_dir = Path(tmp)
            with mock.patch.object(paths, "default_corpus_dir", return_value=corpus_dir), \
                 mock.patch.object(pg_search, "load_pgenv") as load_pgenv_mock, \
                 mock.patch("sys.stderr"):
                load_pgenv_mock.side_effect = pg_search.PostgresUnavailable("no pgenv")
                pg_search.main(["query"])
        (seen_path,), _kwargs = load_pgenv_mock.call_args
        self.assertEqual(seen_path, corpus_dir / ".pgenv")


if __name__ == "__main__":
    unittest.main()
