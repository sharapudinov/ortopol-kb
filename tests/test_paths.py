"""Unit tests for paths.py: no live database, no repository
layout dependency (data_root() itself is monkeypatched where its outcome
matters).
"""
from __future__ import annotations

import unittest
from unittest import mock

import _pathfix  # noqa: F401

import paths


class TryDefaultCorpusDirTests(unittest.TestCase):
    """d809a66d: try_default_corpus_dir() is paths.py's own graceful
    fallback over default_corpus_dir() -- the repo-specific "is this a real
    checkout" knowledge belongs here, not duplicated in
    deploy/deploy_pathfix.py (which delegates to this
    function; see test_deploy_pathfix.py for that delegation).
    """

    def test_normal_checkout_returns_the_corpus_dir(self):
        sentinel = object()
        with mock.patch.object(paths, "default_corpus_dir", return_value=sentinel):
            self.assertIs(paths.try_default_corpus_dir(), sentinel)

    def test_data_root_raising_runtime_error_returns_none_not_propagated(self):
        # A plain checkout with no surrounding theory/iis/ data tree --
        # data_root() (default_corpus_dir()'s own collaborator) raises
        # RuntimeError; try_default_corpus_dir() must swallow it into None
        # rather than letting it propagate to callers that only handle None.
        with mock.patch.object(paths, "data_root", side_effect=RuntimeError("no repo root above here")):
            self.assertIsNone(paths.try_default_corpus_dir())


if __name__ == "__main__":
    unittest.main()
