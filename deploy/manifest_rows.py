"""Every manifest number that is a row count of the dump, stamped from the
dump.

MANIFEST_DESCRIBES_ARTIFACT is what this module is: documents_count,
pages_count, the citation block's table_rows and its two headline totals
are all "how many rows does this package carry", and the only place that
answer exists is the package. They were read from the live database instead
-- documents/pages before the dump was written, the citation counts from a
cut-aware query beside the plan, the full profile's from a fresh read after
pg_dump finished -- while the recipient's bundled gate demands exact
equality with the COPY blocks the file turns out to contain. Every one of
those reads is its own psql process, hence its own connection and its own
implicit transaction; nothing tied them to the dump or to each other, and
the crawl writes ~100k journal rows per pass to the same instance. The
equality was a race the build could not see it had lost.

Now the numbers come out of copy_rows.DumpedRows -- rows counted as they
streamed into the file (public), or read back off it (full) -- so the
equality holds by construction and the gate is checking the reader, not
the weather.

Beside manifest_citation.py rather than in it, and beside manifest_probe.py
rather than in that: those two answer "what does the live database say",
once, before anything is written. This one answers only what the finished
file says, and it is the whole of the manifest that has to wait for it.
"""
from __future__ import annotations

from manifest_keys import Key

# Which dumped table each headline count is. Declared once, here, because
# this module is the only place the two names cross from the dump's
# vocabulary into the manifest's.
CITATION_COUNTS = {Key.WORK_COUNT: "work", Key.CITES_COUNT: "cites"}
CORPUS_COUNTS = {Key.DOCUMENTS_COUNT: "documents", Key.PAGES_COUNT: "pages"}


def stamp_dumped_rows(manifest: dict, rows) -> dict:
    """Writes the dump's own row counts into `manifest`, in place.

    Called by build_package.main() once the dump exists, because that is
    when the answer exists. One answer stamped into all of them, so
    table_rows, work_count and cites_count cannot disagree with each other
    or with the file.

    A table the dump did not carry stamps 0: the keys are in every
    manifest, and the recipient's gate refuses an empty table_rows under a
    shipping mode rather than reading it as silence.
    """
    block = manifest[Key.CITATION]
    block[Key.TABLE_ROWS] = dict(rows.citation)
    for key, table in CITATION_COUNTS.items():
        block[key] = rows.citation.get(table, 0)
    for key, table in CORPUS_COUNTS.items():
        manifest[key] = rows.corpus.get(table, 0)
    return manifest
