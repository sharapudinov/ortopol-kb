"""Make the repository-root modules importable from tests/.

The repository root follows a flat-script layout (no __init__.py, every
script run directly with `python3 script.py`); tests need to import those
same modules by name, so this inserts the parent directory into sys.path
once per test process. Import this before importing anything from the
repository root:

    import _pathfix  # noqa: F401
    import encoding
"""
import sys
from pathlib import Path

_CORPUS_DIR = Path(__file__).resolve().parent.parent
if str(_CORPUS_DIR) not in sys.path:
    sys.path.insert(0, str(_CORPUS_DIR))
