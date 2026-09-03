"""Shared path resolution for the knowledge-base scripts.

Two roots, deliberately named apart, because this repository carries CODE
ONLY and the corpus it operates on lives outside it (see CLAUDE.md):

- kb_root(): this repository's own checkout -- where paths.py itself sits.
- data_root(): the directory holding the source tree `theory/iis/` (and,
  beside it, the derived `corpus/` output dir). Located by walking up from
  this file, so no machine-specific absolute path appears anywhere.

data_root()'s marker is `theory/iis/` -- the data itself. It used to be "a
directory containing both theory/ and lib/", which silently tied every
consumer of this module to the layout of a DIFFERENT repository (ortopol's
lib/), the one this code was extracted from; a checkout beside a theory/
tree with no lib/ sibling is a perfectly valid deployment and must resolve.
"""
from __future__ import annotations

from pathlib import Path

# The two source directories under the data tree, named once. Both appear in
# corpus.documents.source_dir, from which source_path is derived, so a typo
# here is a broken path the completeness predicate reports rather than a
# silently wrong string repeated in four loaders.
IIS_SOURCE_DIR = "theory/iis"
EXTERNAL_SOURCE_DIR = "theory/external"


def kb_root() -> Path:
    """This repository's checkout root (paths.py's own directory)."""
    return Path(__file__).resolve().parent


def data_root() -> Path:
    """The directory containing the corpus source tree theory/iis/.

    Typically the parent of this checkout (the ortopol working tree), but
    nothing here assumes that: any ancestor with a theory/iis/ inside it
    wins, so this repository may sit anywhere beneath the data tree.
    """
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "theory" / "iis").is_dir():
            return candidate
    raise RuntimeError(f"could not locate a theory/iis/ data tree above {here}")


def default_pdf_dir() -> Path:
    return data_root() / IIS_SOURCE_DIR


def default_external_dir() -> Path:
    """Where sources by OTHER authors live -- never theory/iis/, which is the
    Sharapudinov corpus itself. Kept apart because the two carry different
    legal regimes and different loaders, not for tidiness.
    """
    return data_root() / EXTERNAL_SOURCE_DIR


def default_corpus_dir() -> Path:
    return data_root() / "corpus"


def try_default_corpus_dir() -> Path | None:
    """default_corpus_dir(), or None when this checkout has no ancestor
    directory containing a theory/iis/ tree (e.g. a bare clone with no
    surrounding data tree) -- the one graceful fallback data_root() itself
    does not offer (it raises).

    The single place that answers "is this a checkout beside real data" for
    callers that need that fallback rather than a hard failure. Also used
    from deploy/deploy_pathfix.py, which delegates here rather than
    re-implementing the same data_root()-raises-RuntimeError handling a
    second time; that module additionally tolerates paths.py being absent
    entirely (the bundled-artifact case, where this whole file isn't
    shipped), which is meaningless to guard against here since a caller
    that can import this module already knows paths.py exists.
    """
    try:
        return default_corpus_dir()
    except RuntimeError:
        return None


def cache_dir(name: str) -> Path:
    """Where the named cache channel lives in the data tree.

    Inside the data tree, not the checkout: cached bodies are third-party
    JSON and HTML, and CODE_ONLY keeps every byte of them out of git. Not
    scratch either -- a wiped cache costs a day of OpenAlex quota to refill
    (measured: one depth-2 attempt spent ~260 of a 1000-request window,
    which then took 23 h to reset), and the other channels are bought just
    as dearly. These are data the tree keeps.

    One accessor with the channel as an argument, because that is all this
    module knows: where a cache goes. WHY each channel is worth keeping
    differs per channel and is prose belonging beside the code that builds
    it -- pg_load_citations.main() builds every one of them, each through
    the one read-only rule, so a new channel cannot arrive without meeting
    that rule. Four functions here whose bodies differed only in the last
    segment put four rationales in the module that shares none of them.
    """
    return data_root() / "corpus" / "cache" / name
