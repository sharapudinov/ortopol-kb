"""Unit tests for paths.py: no live database, no repository
layout dependency (data_root() itself is monkeypatched where its outcome
matters).
"""
from __future__ import annotations

import unittest
from pathlib import Path
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


class CacheDirTests(unittest.TestCase):
    """One parameterized accessor, not one function per channel.

    Four cache channels arrived as four functions whose bodies were
    identical modulo the last path segment; what differed between them was
    the rationale in each docstring -- prose about why THAT channel is worth
    keeping, which belongs where the channel is built (pg_load_citations.py
    builds all four, each through the one read-only rule) and not in the
    module that only knows where the tree keeps a cache.
    """

    ROOT = Path("/data")

    def _cache(self, name: str):
        with mock.patch.object(paths, "data_root", return_value=self.ROOT):
            return paths.cache_dir(name)

    def test_a_channel_lives_under_the_data_tree(self):
        # Inside the data tree, not the checkout: these are third-party
        # bodies and CODE_ONLY keeps every byte of them out of git.
        self.assertEqual(self._cache("openalex"),
                         self.ROOT / "corpus" / "cache" / "openalex")

    def test_every_channel_differs_only_in_its_last_segment(self):
        for name in ("openalex", "mathnet", "embeddings", "zbmath"):
            with self.subTest(name=name):
                self.assertEqual(self._cache(name).parent,
                                 self.ROOT / "corpus" / "cache")
                self.assertEqual(self._cache(name).name, name)

    def test_the_per_channel_accessors_are_gone(self):
        """A new channel is an argument, not another function here."""
        for name in dir(paths):
            with self.subTest(name=name):
                self.assertFalse(name.endswith("_cache_dir"), name)


if __name__ == "__main__":
    unittest.main()
