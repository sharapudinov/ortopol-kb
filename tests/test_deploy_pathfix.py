"""Unit tests for deploy/deploy_pathfix.py: no live database,
no Docker, no real repository layout dependency (paths.data_root() itself
is monkeypatched where its outcome matters).
"""
from __future__ import annotations

import unittest
from unittest import mock

import _pathfix  # noqa: F401
import _pathfix_deploy  # noqa: F401

import deploy_pathfix
import paths


class TryDefaultCorpusDirTests(unittest.TestCase):
    """25b41789: try_default_corpus_dir() is the single chokepoint answering
    "is this a checkout or a bundled artifact" -- it must return None for
    BOTH ways that question resolves to "no repository context", not just
    the module-missing one.
    """

    def test_paths_module_absent_returns_none(self):
        # The ordinary bundled-artifact case: paths.py is deliberately not
        # shipped in the artifact at all (see the module docstring).
        with mock.patch.dict("sys.modules", {"paths": None}):
            self.assertIsNone(deploy_pathfix.try_default_corpus_dir())

    def test_data_root_raising_runtime_error_returns_none_not_propagated(self):
        # paths.py IS importable (a plain checkout has it), but data_root()
        # raises RuntimeError when no ancestor directory has a theory/iis/
        # tree -- e.g. this checkout with no surrounding data tree.
        # Previously only ImportError was caught here, so this would have
        # propagated past smoke_test.artifact_data_dir/pg_search.main,
        # neither of which handles anything but None.
        with mock.patch.object(paths, "data_root", side_effect=RuntimeError("no repo root above here")):
            self.assertIsNone(deploy_pathfix.try_default_corpus_dir())

    def test_normal_checkout_returns_the_corpus_dir(self):
        sentinel = object()
        with mock.patch.object(paths, "default_corpus_dir", return_value=sentinel):
            self.assertIs(deploy_pathfix.try_default_corpus_dir(), sentinel)


if __name__ == "__main__":
    unittest.main()
