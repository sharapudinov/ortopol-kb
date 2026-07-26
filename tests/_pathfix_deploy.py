"""sys.path shim for tests exercising deploy/*.py modules.

Import _pathfix first (adds the repository root itself, for pg_common/
pg_search/paths), then this (adds deploy/, for build_package/
manifest_probe/artifact_bundle/smoke_checks/smoke_test/compose_lifecycle/
dump_integrity). One shared shim so every deploy-related test file reuses
the same logic instead of open-coding its own sys.path.insert.
"""
import sys
from pathlib import Path

_DEPLOY_DIR = Path(__file__).resolve().parent.parent / "deploy"
if str(_DEPLOY_DIR) not in sys.path:
    sys.path.insert(0, str(_DEPLOY_DIR))
