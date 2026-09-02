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
        blocks, at = [], 0
        while (start := SCHEMA.find("UPDATE citation.crawl_step SET", at)) != -1:
            at = SCHEMA.index(";", start) + 1
            blocks.append(SCHEMA[start:at])
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


if __name__ == "__main__":
    unittest.main()
