"""Every sequence the artifact ships arrives where its rows left it.

The one guarantee in the package whose breach shows up only at the
recipient's end, and only later: a BIGSERIAL sequence restored at 1 makes
no noise at restore time -- the COPY blocks carry their ids and every row
lands -- and then their FIRST insert collides with an id the dump already
used. schema_catalog.setval_sql() writes the statement that prevents it,
and until now nothing on this side of the boundary asked whether it was
written. The builder's own unit tests did, which is a claim about the
packager; this is the claim about the FILE, the same polarity every other
check here has (ARTIFACT_SIDE_FAILS_CLOSED).

WHICH columns own a sequence is read from the dump's own DDL (dump_scan.
StatementReader, ALTER SEQUENCE ... OWNED BY), not from a list beside the
classification maps. Those maps say topology or content, which is a
different question, and a serialness map added next to them would be a
second, hand-written copy of what the catalog knows -- silent about exactly
the column added after the copy was written, which is the case this check
exists for. The dump states the ownership itself, in both profiles, so the
question and its subject come off the same bytes.
"""
from __future__ import annotations


def check_sequences_are_repositioned(contents) -> tuple[bool, str]:
    """Every sequence-owning column whose table shipped rows is followed by
    a setval, and the setval comes AFTER the COPY block.

    A table with no COPY block at all is skipped, and that is not a hole:
    nothing was inserted, so the sequence the recipient restores is the one
    a fresh schema starts with. Ordering is checked rather than assumed
    because a setval before the rows is the one shape that reads as done
    and leaves the sequence exactly where it was.
    """
    owned = sorted(contents.sequence_columns)
    problems = []
    checked = []
    for column in owned:
        table = column.rsplit(".", 1)[0]
        scan = contents.tables.get(table)
        if scan is None:
            continue
        checked.append(column)
        at = contents.sequence_resets.get(column)
        if at is None:
            problems.append(f"{column}: {scan.rows} строк приехало, setval нет — "
                            "первая вставка получателя возьмёт занятый id")
        elif at < scan.ended_at:
            problems.append(f"{column}: setval стоит ДО своего COPY-блока "
                            "— последовательность остаётся там, где была")
    return not problems, (
        f"{len(checked)} of {len(owned)} sequence-owning column(s) shipped rows: "
        + ("; ".join(problems) or (", ".join(checked) or "нет таких колонок"))
    )
