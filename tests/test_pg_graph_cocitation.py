"""Offline tests for the co-citation consumer (pg_graph_cocitation.py) and
the VOSviewer export it feeds: both bounds of the self-join, the
determinism of the answer's order, and the two file formats.

The live fixture that runs the query against the real graph is
test_pg_graph_consumers_live.py.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import _pathfix  # noqa: F401
import pg_graph_cocitation as pgcoci
from pg_common import FIELD_SEP, RECORD_SEP


class CocitationSqlTests(unittest.TestCase):
    """The self-join is bounded on BOTH sides: which citers may generate
    pairs at all, and how many pairs come back.
    """

    SQL = pgcoci._COCITATION_SQL

    def test_out_degree_cap_is_applied_before_the_self_join(self):
        citers = self.SQL[self.SQL.index("WITH citers AS ("):self.SQL.index("pairs AS (")]
        self.assertIn("HAVING count(*) <= :max_out_degree", citers)
        self.assertLess(self.SQL.index(":max_out_degree"),
                        self.SQL.index("JOIN citation.cites c2"))

    def test_the_self_join_only_sees_capped_citers(self):
        self.assertIn("JOIN citers ON citers.citing = c1.citing", self.SQL)

    def test_result_is_limited_and_ordered_deterministically(self):
        self.assertIn("ORDER BY p.n DESC, wa.key, wb.key", self.SQL)
        self.assertIn("LIMIT :limit", self.SQL)
        self.assertLess(self.SQL.index("ORDER BY p.n DESC"), self.SQL.index("LIMIT :limit"))

    def test_defaults_are_named_once_and_reach_the_query(self):
        seen = {}

        def fake_run_sql(env, sql, variables=None, extra_args=None):
            seen.update(variables or {})
            return mock.Mock(stdout="")

        with mock.patch.object(pgcoci, "run_sql", side_effect=fake_run_sql):
            pgcoci.cocitation({})
        self.assertEqual(seen["max_out_degree"], str(pgcoci.MAX_OUT_DEGREE))
        self.assertEqual(seen["limit"], str(pgcoci.COCITATION_LIMIT))


class RowDecodingTests(unittest.TestCase):
    """The other half of the query: turning psql's output back into pairs.

    The SQL tests above read the statement, and the only test that called
    cocitation() stubbed run_sql to return an empty stdout -- so `rec.split(
    FIELD_SEP, 4)` and `int(n)` never ran once. The output is fed here the
    way psql -R -F actually emits it: records separated by RECORD_SEP,
    fields by FIELD_SEP, and a trailing record separator on the last row.
    """

    ROWS = (
        ("W1", "Приближение функций", "W2", "Ортогональные многочлены", "7"),
        ("W3", "", "W4", "Sobolev orthogonality", "2"),
    )

    def _decode(self, stdout: str) -> list[dict]:
        with mock.patch.object(pgcoci, "run_sql", return_value=mock.Mock(stdout=stdout)):
            return pgcoci.cocitation({})

    def _psql_output(self, rows) -> str:
        return "".join(FIELD_SEP.join(row) + RECORD_SEP for row in rows)

    def test_each_record_becomes_one_pair_with_its_count_as_a_number(self):
        pairs = self._decode(self._psql_output(self.ROWS))
        self.assertEqual(pairs, [
            {"a_key": "W1", "a_title": "Приближение функций",
             "b_key": "W2", "b_title": "Ортогональные многочлены", "count": 7},
            {"a_key": "W3", "a_title": "", "b_key": "W4",
             "b_title": "Sobolev orthogonality", "count": 2},
        ])
        self.assertIsInstance(pairs[0]["count"], int)

    def test_a_title_carrying_the_field_separators_own_shape_survives(self):
        """The split is bounded at four, so only the first four separators
        are field boundaries -- a title is the LAST text field of the pair
        and the count is what the bound protects.
        """
        rows = [("W1", "A", "W2", "B: 1, 2, 3", "11")]
        self.assertEqual(self._decode(self._psql_output(rows))[0]["count"], 11)

    def test_no_rows_is_no_pairs_rather_than_one_empty_one(self):
        for stdout in ("", "\n", RECORD_SEP):
            with self.subTest(stdout=repr(stdout)):
                self.assertEqual(self._decode(stdout), [])

    def test_the_decoded_pairs_are_what_the_export_is_written_from(self):
        """The map and the printed table can never describe different sets,
        which is only true while the export reads these very dicts.
        """
        pairs = self._decode(self._psql_output(self.ROWS))
        map_lines, network_lines = pgcoci.build_vosviewer_export(pairs)
        self.assertEqual(len(map_lines), 1 + 4)
        self.assertEqual(network_lines, ["1\t2\t7", "3\t4\t2"])


class VosviewerExportTests(unittest.TestCase):
    """Format: https://app.vosviewer.com/docs/file-types/map-and-network-file-type/
    -- tab-delimited; map file has a header row and one row per distinct
    item; network file has NO header, one 'id1\\tid2\\tweight' row per link.
    """
    PAIRS = [
        {"a_key": "k:a", "a_title": "A", "b_key": "k:b", "b_title": "B", "count": 3},
        {"a_key": "k:a", "a_title": "A", "b_key": "k:c", "b_title": "C", "count": 1},
    ]

    def test_map_has_header_and_one_row_per_distinct_node(self):
        map_lines, _ = pgcoci.build_vosviewer_export(self.PAIRS)
        self.assertEqual(map_lines[0], "id\tlabel")
        self.assertEqual(len(map_lines), 1 + 3, "3 distinct nodes: k:a, k:b, k:c")

    def test_network_has_no_header_and_one_row_per_pair(self):
        _, network_lines = pgcoci.build_vosviewer_export(self.PAIRS)
        self.assertEqual(len(network_lines), len(self.PAIRS))
        for line in network_lines:
            self.assertEqual(len(line.split("\t")), 3)

    def test_node_ids_are_sequential_integers_from_one(self):
        map_lines, network_lines = pgcoci.build_vosviewer_export(self.PAIRS)
        ids = sorted(int(line.split("\t")[0]) for line in map_lines[1:])
        self.assertEqual(ids, [1, 2, 3])
        # every id referenced in the network file must appear in the map
        map_ids = {line.split("\t")[0] for line in map_lines[1:]}
        for line in network_lines:
            a, b, _weight = line.split("\t")
            self.assertIn(a, map_ids)
            self.assertIn(b, map_ids)

    def test_write_creates_both_files_and_reports_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            map_path, network_path, n_nodes, n_edges = pgcoci.write_vosviewer_export(self.PAIRS, Path(tmp))
            self.assertTrue(map_path.is_file())
            self.assertTrue(network_path.is_file())
            self.assertEqual(n_nodes, 3)
            self.assertEqual(n_edges, 2)


if __name__ == "__main__":
    unittest.main()
