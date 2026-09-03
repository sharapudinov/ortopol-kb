"""manifest.json's citation block: what the package says about the graph.

Beside manifest_probe.py rather than inside it, by responsibility and by
kb/CLAUDE.md FILE_SIZE. That module answers "what does the live database
currently say", once, before anything is written; this one answers what
the block LOOKS like, and every number in it is a fact about the dump,
stamped afterwards from what was written (manifest_rows.py).

The kind census was the exception and is no longer one. It was read live
here -- one psql process, its own connection and its own implicit
transaction, before the dump existed -- while the crawl writes ~100k
journal rows per pass against the same instance, and nothing on the
artifact side ever held it to the shipped bytes. `kind` is TOPOLOGY
(deploy/citation_columns.py), so it ships under every mode that ships the
schema and the census is derivable from the COPY stream like every other
number here.
"""
from __future__ import annotations

from manifest_keys import Key


def citation_block(mode: str, policy_source: str) -> dict:
    """The block's shape, with every number left for the dump to fill.

    MANIFEST_DESCRIBES_ARTIFACT: the numbers are of the rows THIS package
    carries, not of the live schema. The public profile drops every work row
    (and every edge and journal row that names it) whose document its own
    legal cut removed, so counting the live schema here would describe a
    package nobody has produced yet -- work_count, cites_count, table_rows
    and work_by_kind are all FILLED by build_package.main() from what the
    dump wrote (manifest_rows.stamp_dumped_rows), and
    deploy/citation_cut_checks.py holds the shipped rows to each of them.

    The keys are in every manifest either way, under every mode: the
    recipient's gate refuses an empty table_rows under a shipping mode
    rather than reading it as silence.

    `mode` is resolved once per build by citation_profile.
    resolve_citation_mode() and handed in; this module never re-derives it.
    `policy_source` travels the same way and says whose decision that mode
    was -- the owner's row, or the command line's --policy-override.
    """
    return {Key.CITATION_MODE: mode, Key.CITATION_POLICY_SOURCE: policy_source,
            Key.WORK_COUNT: 0, Key.CITES_COUNT: 0, Key.WORK_BY_KIND: {},
            Key.TABLE_ROWS: {}}
