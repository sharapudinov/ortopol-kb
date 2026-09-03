"""The one-time journal parse, read without a database.

Everything that checks what pg_schema_citation_backfill.sql PRODUCES lives
in test_journal_columns.py and needs a live instance, because the parse is
SQL. The whole file skips when Postgres is unreachable -- so on a machine
without the corpus container the backfill could be broken with no test
signal at all, which is exactly the reasoning the projection digests were
given an offline guard for.

These read the schema file's own text instead: the regexes it applies are
extracted from it (never restated -- a copy would pass over a parse the
schema no longer performs) and run here with `re` against reason strings of
each shape the crawl ever wrote, plus the SQL-shape checks that say the
parse fills only NULLs and looks only once.

POSIX and Python disagree about plenty; they do not disagree about these,
which are character classes, a bounded quantifier and one capture. When a
future marker needs something they read differently, the live test next
door is what says so -- this file is a floor, not a substitute.
"""
from __future__ import annotations

import re
import unittest

import _pathfix  # noqa: F401

import pg_graph_common
from citations import journal

SCHEMA = pg_graph_common.SCHEMA_BACKFILL.read_text(encoding="utf-8")

# Every `substring(reason from '<regex>')` the schema applies, in file
# order. The column each one fills is read from the same line, so a marker
# renamed on one side of the assignment cannot quietly keep the old name
# here.
_PARSE = re.compile(r"substring\(reason from '([^']+)'\)")

# Journal prose of every shape the crawl wrote before the columns existed:
# a kept candidate, a dropped one, a twin promotion (seed=, no score), a
# hub skip (only a citer count) and a fetch step (nothing to parse). Taken
# from the same rows the live fixture inserts.
REASONS = {
    "keep": "kept; score=0.6123 tau=0.5000 relation=cites node=W_NODE",
    "drop": "below-threshold; score=0.4001 tau=0.5000 relation=referenced",
    "twin": "twin-of=2019_rm9846 seed=W_RU",
    "hub": "cited_by_count=5000 > cap 1000",
    "fetch": "",
}


def _patterns() -> list[str]:
    found = _PARSE.findall(SCHEMA)
    assert found, "в схеме не нашлось ни одного разбора reason"
    return found


def _one(pattern: str, reason: str):
    found = re.search(pattern, reason)
    return found.group(1) if found else None


class ExtractedRegexesTests(unittest.TestCase):
    """The patterns themselves, applied to prose of every shape."""

    def test_the_schema_still_parses_the_six_facts(self):
        self.assertEqual(
            sorted(_patterns()),
            sorted(["node=([^ ]+)", "seed=([^ ]+)", "score=(-?[0-9.]+)",
                    "tau=(-?[0-9.]+)", "relation=([a-z]+)",
                    "cited_by_count=([0-9]+)"]))

    def test_a_kept_candidate_yields_every_column_it_carries(self):
        reason = REASONS["keep"]
        self.assertEqual(_one("node=([^ ]+)", reason), "W_NODE")
        self.assertEqual(float(_one("score=(-?[0-9.]+)", reason)), 0.6123)
        self.assertEqual(float(_one("tau=(-?[0-9.]+)", reason)), 0.5)
        self.assertEqual(_one("relation=([a-z]+)", reason), "cites")
        self.assertIsNone(_one("cited_by_count=([0-9]+)", reason))

    def test_a_dropped_candidate_names_no_node(self):
        reason = REASONS["drop"]
        self.assertIsNone(_one("node=([^ ]+)", reason))
        self.assertIsNone(_one("seed=([^ ]+)", reason))
        self.assertEqual(float(_one("score=(-?[0-9.]+)", reason)), 0.4001)
        self.assertEqual(_one("relation=([a-z]+)", reason), "referenced")

    def test_a_twin_promotion_is_a_seed_and_nothing_else(self):
        reason = REASONS["twin"]
        self.assertIsNone(_one("node=([^ ]+)", reason))
        self.assertEqual(_one("seed=([^ ]+)", reason), "W_RU")
        self.assertIsNone(_one("score=(-?[0-9.]+)", reason))
        self.assertIsNone(_one("tau=(-?[0-9.]+)", reason))

    def test_a_hub_skip_is_a_citer_count_and_nothing_else(self):
        reason = REASONS["hub"]
        self.assertEqual(int(_one("cited_by_count=([0-9]+)", reason)), 5000)
        self.assertIsNone(_one("relation=([a-z]+)", reason))
        self.assertIsNone(_one("score=(-?[0-9.]+)", reason))

    def test_a_step_with_nothing_to_parse_yields_nothing(self):
        for pattern in _patterns():
            self.assertIsNone(_one(pattern, REASONS["fetch"]), pattern)

    def test_the_node_marker_stops_at_the_space_not_at_the_line(self):
        """`[^ ]+` is what keeps `node=W_NODE relation=cites` from putting
        the rest of the sentence into node_key.
        """
        self.assertEqual(_one("node=([^ ]+)", "node=W_A relation=cites"), "W_A")

    def test_a_negative_score_survives_the_number_class(self):
        """A candidate with no embeddable text is scored -1.0, and the
        journal said so in prose before the column existed.
        """
        self.assertEqual(float(_one("score=(-?[0-9.]+)", "no-text; score=-1.0")), -1.0)

    def test_every_pattern_captures_exactly_one_group(self):
        # substring(... from ...) returns the FIRST capture; a pattern that
        # grew a second group would silently return the wrong half.
        for pattern in _patterns():
            self.assertEqual(re.compile(pattern).groups, 1, pattern)


class ParseShapeIsIdempotentTests(unittest.TestCase):
    """The parse is applied on every `pg_graph.py init` and every non-dry-run
    crawl, so "runs again, changes nothing" is a property of its SHAPE, not
    only of the values it happens to produce on the live journal.
    """

    def _update_blocks(self) -> list[str]:
        """The UPDATEs of the PARSE, and only those. The file carries a
        second one-time block (crawl_step_no_text_reason, next class), whose
        statement fills no column out of prose and answers to different
        rules; scoped by the parse's own DO block so neither can be read as
        the other.
        """
        parse = SCHEMA[SCHEMA.index("DO $backfill$"):SCHEMA.index("$backfill$;")]
        blocks, at = [], 0
        while (start := parse.find("UPDATE citation.crawl_step SET", at)) != -1:
            at = parse.index(";", start) + 1
            blocks.append(parse[start:at])
        return blocks

    def test_the_registry_short_circuits_the_whole_block(self):
        """The one-time record is what stops the parse from LOOKING: the
        qualifiers are regex predicates no index can serve, so without it
        every apply was a sequential scan of an append-only table.
        """
        self.assertIn("citation.schema_backfill", SCHEMA)
        guard = SCHEMA[SCHEMA.index("IF EXISTS"):SCHEMA.index("UPDATE citation")]
        self.assertIn("crawl_step_reason_parse", guard)
        self.assertIn("RETURN;", guard)
        self.assertLess(SCHEMA.index("IF EXISTS"),
                        SCHEMA.index("UPDATE citation.crawl_step SET"))
        self.assertLess(SCHEMA.index("UPDATE citation.crawl_step SET"),
                        SCHEMA.index("INSERT INTO citation.schema_backfill"))

    def test_the_registry_row_is_written_inside_the_same_block(self):
        block = SCHEMA[SCHEMA.index("DO $backfill$"):]
        self.assertIn("INSERT INTO citation.schema_backfill (name) VALUES "
                      "('crawl_step_reason_parse');", block)

    def test_every_assignment_fills_a_null_and_never_overwrites(self):
        for block in self._update_blocks():
            pairs = re.findall(r"^\s*(\w+) = coalesce\((\w+)\s*,", block, re.M)
            self.assertTrue(pairs, block)
            for column, first in pairs:
                self.assertEqual(column, first,
                                 f"{column} перезаписывает уже разобранное значение")

    def test_every_filled_column_is_paired_with_its_own_marker(self):
        """`WHERE col IS NULL AND reason ~ 'marker'` per column: a block
        whose qualifier named fewer columns than its SET would keep
        matching rows the parse cannot fill any further, and rewrite them
        on every apply.
        """
        for block in self._update_blocks():
            assigned = set(re.findall(r"^\s*(\w+) = coalesce\(", block, re.M))
            qualified = set(re.findall(r"(\w+) IS NULL AND reason ~", block))
            self.assertEqual(assigned, qualified, block)

    def test_the_blocks_touch_no_column_beyond_the_parsed_ones(self):
        assigned = set()
        for block in self._update_blocks():
            assigned |= set(re.findall(r"^\s*(\w+) = coalesce\(", block, re.M))
        self.assertEqual(assigned, {"node_key", "score", "tau", "relation",
                                    "cited_by_count"})


class NoTextReasonBlockTests(unittest.TestCase):
    """The second one-time block: the prose on drop rows nothing was
    measured on.

    A candidate with no title is scored NO_TEXT_SCORE and never compared
    with tau, but every drop row said "below-threshold" -- a relevance
    verdict on a candidate no relevance was computed for.
    citations/journal.drop_reason() tells them apart now, and this block is
    the rows written before it did.
    """

    NAME = "crawl_step_no_text_reason"

    def _block(self) -> str:
        start = SCHEMA.index("DO $no_text_reason$")
        return SCHEMA[start:SCHEMA.index("$no_text_reason$;", start)]

    def test_the_block_is_guarded_by_its_own_registry_name(self):
        block = self._block()
        self.assertIn(self.NAME, block)
        self.assertLess(block.index("IF EXISTS"), block.index("UPDATE citation"))
        self.assertIn("RETURN;", block[:block.index("UPDATE citation")])

    def test_it_records_itself_under_a_name_of_its_own(self):
        self.assertIn("INSERT INTO citation.schema_backfill (name) VALUES "
                      f"('{self.NAME}');", self._block())
        self.assertNotIn("crawl_step_reason_parse", self._block())

    def test_the_score_column_decides_and_the_prose_only_narrows(self):
        """The distinguishing fact is a COLUMN (JOURNAL_FACTS_ARE_COLUMNS):
        `score <= -1` is the same test journal.drop_reason() makes, and the
        reason is compared by equality only so a row somebody has already
        reworded is left alone. No substring, no regex.
        """
        statement = self._block()[self._block().index("UPDATE citation"):]
        self.assertIn("score <= -1", statement)
        self.assertIn(f"reason = '{journal.BELOW_THRESHOLD_REASON}'", statement)
        self.assertIn(f"SET reason = '{journal.NO_TEXT_REASON}'", statement)
        for forbidden in ("substring(", "split_part(", "strpos(", "regexp_"):
            self.assertNotIn(forbidden, statement)

    def test_a_second_apply_would_find_nothing_even_without_the_guard(self):
        """Value-idempotent as well as work-idempotent: the rows it rewrites
        stop matching its own WHERE the moment it has rewritten them.
        """
        statement = self._block()[self._block().index("UPDATE citation"):]
        self.assertNotEqual(journal.NO_TEXT_REASON, journal.BELOW_THRESHOLD_REASON)
        self.assertIn(f"reason = '{journal.BELOW_THRESHOLD_REASON}'", statement)


if __name__ == "__main__":
    unittest.main()
