"""manifest.json's citation block: what the package says about the graph.

Beside manifest_probe.py rather than inside it, by responsibility and by
kb/CLAUDE.md FILE_SIZE. That module answers "what does the live database
currently say", once, before anything is written. So does this one, for
the single number in the block a COPY stream cannot produce: the census by
kind. Everything else in the block is a row count of the dump, stamped
afterwards from what was written -- manifest_rows.py, which is where every
such number in the manifest now comes from.
"""
from __future__ import annotations

from deploy_pathfix import ensure_corpus_importable

ensure_corpus_importable()

import citation_profile  # noqa: E402
from manifest_keys import Key  # noqa: E402
from manifest_contract import ships_citation  # noqa: E402


def citation_block(env: dict, mode: str, public: bool, policy_source: str) -> dict:
    """MANIFEST_DESCRIBES_ARTIFACT: the numbers are of the rows THIS package
    carries, not of the live schema. The public profile drops every work row
    (and every edge and journal row that names it) whose document its own
    legal cut removed, so the census is taken with that cut applied --
    citation_content_checks.py compares exactly these numbers against the
    rows the dump turns out to contain.

    The BREAKDOWN is read here because nothing else can produce it: no COPY
    block carries a census. The two totals are not -- they are the dump's
    own row counts, stamped by manifest_rows.stamp_dumped_rows() once the
    file has them.

    `mode` is resolved once per build by citation_profile.
    resolve_citation_mode() and handed in; this module never re-derives it.
    `policy_source` travels the same way and says whose decision that mode
    was -- the owner's row, or the command line's --policy-override.
    """
    # work_count, cites_count and table_rows are declared here and FILLED by
    # build_package.main() from what the dump wrote (manifest_rows.py):
    # this runs before the dump exists, and a live count would describe a
    # package nobody has produced yet. The keys are in every manifest
    # either way -- the recipient's gate refuses an empty table_rows under
    # a shipping mode rather than reading it as silence.
    block = {Key.CITATION_MODE: mode, Key.CITATION_POLICY_SOURCE: policy_source,
             Key.WORK_COUNT: 0, Key.CITES_COUNT: 0, Key.WORK_BY_KIND: {},
             Key.TABLE_ROWS: {}}
    # manifest_contract.ships_citation(), the predicate the dump itself is
    # written by: a mode that carries no citation byte must not have the
    # live schema's census stamped into the block describing it.
    if not ships_citation(mode):
        return block
    block[Key.WORK_BY_KIND] = citation_profile.work_by_kind(env, shipped_only=public)
    return block
