"""Shared sys.path shim for deploy/ scripts.

Two layouts import pg_common/pg_search through this module:

- repository checkout: this file lives at deploy/, the flat script layout
  it imports from (pg_common.py, pg_search.py, paths.py) sits one level up.
- self-contained artifact: build_package.py's bundle_runtime_files() copies
  pg_common.py and pg_search.py into a sibling corpus_lib/ directory inside
  the tar (see CORPUS_LIB_FILES in artifact_bundle.py) so the SAME deploy
  scripts run unmodified on a machine with no ortopol checkout at all.
  pg_rank_probe.py is NOT among them -- it is deploy-only (see its own
  module docstring) and ships as a flat sibling in DEPLOY_FILES instead,
  next to smoke_checks.py/manifest_probe.py/drift_probe.py, which import it
  directly without going through ensure_corpus_importable() at all.

ensure_corpus_importable() tries the bundled layout first, then falls back
to the repository layout, so callers do not need to know which one they are
running under.
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def ensure_corpus_importable() -> None:
    for candidate in (_HERE / "corpus_lib", _HERE.parent):
        if (candidate / "pg_common.py").is_file():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return
    raise RuntimeError(
        f"neither {_HERE / 'corpus_lib'} nor {_HERE.parent} contains pg_common.py "
        "-- corrupted checkout or corrupted artifact"
    )


def try_default_corpus_dir() -> Path | None:
    """paths.py's try_default_corpus_dir(), or None outside a checkout
    entirely (the ordinary bundled-artifact case: paths.py is deliberately
    not bundled at all, see the module docstring above).

    Delegates rather than re-implementing: the theory/iis/ ancestor walk
    (and its RuntimeError-on-bare-clone fallback) is repo-specific
    knowledge that belongs in paths.py, not duplicated here. This module's
    own job is narrower and stays here -- tolerating paths.py being ABSENT,
    which paths.py itself cannot express about itself.

    The single place that answers "is this the checkout or the artifact"
    for the one repo-only convenience (auto-discovering the newest
    kb-*.tar.zst, or resolving a bare corpus-relative .pgenv) that has no
    meaning outside a checkout. Callers (smoke_test.artifact_data_dir) only
    ever handle the None case, not a raised exception.
    """
    try:
        from paths import try_default_corpus_dir as _try_default_corpus_dir
    except ImportError:
        return None
    return _try_default_corpus_dir()
