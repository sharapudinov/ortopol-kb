"""manifest.json's citation block: what the package says about the graph.

Beside manifest_probe.py rather than inside it, by responsibility and by
kb/CLAUDE.md FILE_SIZE. That module answers "what does the live database
currently say", once, before anything is written. This block is the one
part of the manifest that cannot be answered that way: its two totals are
the DUMP's own row counts, known only after the dump exists, and its
census is the one number no COPY block can produce. Two moments, two
functions, and build_package.main() calls them in that order.
"""
from __future__ import annotations

from deploy_pathfix import ensure_corpus_importable

ensure_corpus_importable()

import citation_profile  # noqa: E402
from manifest_keys import Key  # noqa: E402
from manifest_contract import ships_citation  # noqa: E402


# Which dumped table each headline count of manifest.citation is. Declared
# once, here, because stamp_dumped_rows() below is the only place the two
# names cross from the dump's vocabulary into the manifest's.
DUMPED_COUNTS = {Key.WORK_COUNT: "work", Key.CITES_COUNT: "cites"}


def stamp_dumped_rows(block: dict, table_rows: dict[str, int]) -> dict:
    """Writes the dump's own row counts into the citation block.

    Called by build_package.main() once the dump exists, because that is
    when the answer exists: MANIFEST_DESCRIBES_ARTIFACT, and the recipient's
    gate holds work_count/cites_count AND table_rows to the very COPY blocks
    the file turns out to contain (citation_cut_checks). One answer stamped
    into all three, so they cannot disagree -- and counting the cut row sets
    against the live database beforehand, only to require equality
    afterwards, was two readings of one cut and the expensive one at that.

    A table the dump did not carry stamps 0: the key is in every manifest,
    and the gate refuses an empty table_rows under a shipping mode rather
    than reading it as silence.
    """
    block[Key.TABLE_ROWS] = table_rows
    for key, table in DUMPED_COUNTS.items():
        block[key] = table_rows.get(table, 0)
    return block


def citation_block(env: dict, mode: str, public: bool, policy_source: str) -> dict:
    """MANIFEST_DESCRIBES_ARTIFACT: the numbers are of the rows THIS package
    carries, not of the live schema. The public profile drops every work row
    (and every edge and journal row that names it) whose document its own
    legal cut removed, so the census is taken with that cut applied --
    citation_content_checks.py compares exactly these numbers against the
    rows the dump turns out to contain.

    The BREAKDOWN is read here because nothing else can produce it: no COPY
    block carries a census. The two totals are not -- they are the dump's
    own row counts, stamped by stamp_dumped_rows() above once it has
    written them.

    `mode` is resolved once per build by citation_profile.
    resolve_citation_mode() and handed in; this module never re-derives it.
    `policy_source` travels the same way and says whose decision that mode
    was -- the owner's row, or the command line's --policy-override.
    """
    # work_count, cites_count and table_rows are declared here and FILLED by
    # build_package.main() from what the dump wrote: this runs before the
    # dump exists, and a live count would describe a package nobody has
    # produced yet. The keys are in every manifest either way -- the
    # recipient's gate refuses an empty table_rows under a shipping mode
    # rather than reading it as silence.
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
