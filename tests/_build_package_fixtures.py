"""The manifest gather_manifest() returns, as build_package.main() relies on
it -- shared by the two test modules that drive main() (test_build_package.py
and test_build_package_citation.py) so the two cannot drift apart.
"""
from __future__ import annotations

import _pathfix  # noqa: F401
import _pathfix_deploy  # noqa: F401

from manifest_keys import Key


def fake_manifest(**overrides) -> dict:
    """Since manifest schema 4 that always includes the profile and the
    schemas the dump carries (main() prints them and profile_checks.py
    verifies them against the dump), and since 9 a citation block: it is
    declared by the probe and STAMPED by main() with what the dump turned
    out to carry, so a manifest without one is not a shape this packager
    ever produces.
    """
    manifest = {
        "schema_version": 5,
        "profile": "full",
        "schemas": ["corpus", "measurements"],
        "documents_count": 70,
        "pages_count": 2462,
        Key.CITATION: {Key.CITATION_MODE: "full-skeleton", Key.TABLE_ROWS: {}},
        Key.CORPUS: {Key.TABLE_ROWS: {}},
    }
    manifest.update(overrides)
    return manifest
