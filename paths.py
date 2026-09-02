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


def default_cache_dir() -> Path:
    """Where the OpenAlex crawl caches raw responses.

    Inside the data tree, not the checkout: these are third-party JSON bodies
    and CODE_ONLY keeps every byte of them out of git. Not scratch either --
    a wiped cache costs a day of OpenAlex quota to refill (measured: one
    depth-2 attempt spent ~260 of a 1000-request window, which then took 23 h
    to reset), so the cache is data the tree keeps, not a temp file.
    """
    return data_root() / "corpus" / "cache" / "openalex"


def default_mathnet_cache_dir() -> Path:
    """Math-Net pages, cached for the same reason as the OpenAlex responses.

    The site starts timing out after a few dozen rapid requests, and the
    titles it serves are the identity anchor the twin rule depends on -- a
    re-seed must not have to earn them again.
    """
    return data_root() / "corpus" / "cache" / "mathnet"


def default_zbmath_cache_dir() -> Path:
    """zbMATH documents, keyed by the zbMATH id, for the same reason again.

    The abstracts the seeds fall back on are static between runs, the API
    already answers 429 under load, and the fallback sits on the startup
    path of every non-offline invocation -- one sequential request per
    matched seed, tens of seconds, before anything else happens.
    """
    return data_root() / "corpus" / "cache" / "zbmath"
