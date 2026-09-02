"""The hub measurement as a READER of the journal's columns.

Split from test_journal_columns.py by responsibility (and by kb/CLAUDE.md
FILE_SIZE): that file is about the columns themselves and the one-time
parse that filled them, this one about citations/hub_report.py, the
consumer JOURNAL_FACTS_ARE_COLUMNS names by name -- it used to recover the
depth-1 node set and each node's relation out of `reason` with substring
arithmetic.

Two halves. The shape of POPULATE is read as text and needs no database, so
an offline run catches the parsing coming back; what the statement PRODUCES
is compared against the measurement already recorded, inside a rolled-back
transaction, so the recorded run is never rewritten.
"""
from __future__ import annotations

import unittest

import _pathfix  # noqa: F401
from citations import hub_report
from paths import default_corpus_dir
from pg_common import PostgresUnavailable, check_postgres_available, load_pgenv, run_sql

FIELD_SEP = "\x1f"


def _live_env() -> dict[str, str]:
    try:
        env = load_pgenv(default_corpus_dir() / ".pgenv")
    except PostgresUnavailable as exc:
        raise unittest.SkipTest(f"Postgres not configured: {exc}")
    if not check_postgres_available(env):
        raise unittest.SkipTest("Postgres not reachable")
    return env


class HubReportPopulateShapeTests(unittest.TestCase):
    """hub_report.POPULATE read as text, with no database in sight.

    JOURNAL_FACTS_ARE_COLUMNS names this consumer by name: it used to
    recover the depth-1 node set and each node's relation by picking
    substrings out of `reason`, which takes no index, matches a name
    occurring inside a phrase, and breaks when the phrase is reworded. Its
    siblings -- the public artifact's journal cut and the dump -- carry
    DB-free guards against exactly that, and this one had none: every check
    on it sat behind a live-Postgres skip, so an offline run would not
    notice the parsing coming back.
    """

    def test_the_slice_is_taken_from_columns(self):
        for column in ("node_key", "relation", "action", "depth"):
            self.assertIn(column, hub_report.POPULATE)

    def test_no_word_of_the_prose_is_parsed(self):
        for parser in ("strpos(", "split_part(", "substring(", "j.reason",
                       "reason ~", "reason LIKE"):
            self.assertNotIn(parser, hub_report.POPULATE,
                             "reason снова разбирается на подстроки")

    def test_the_relation_comes_from_the_decision_not_from_the_blob(self):
        """It used to be re-derived from citation.work.evidence with a
        coalesce(..., 'unknown') -- reading a classification off a blob
        shaped by registry.Node.absorb rather than off the row that made
        the decision.
        """
        self.assertIn("j.relation", hub_report.POPULATE)
        self.assertNotIn("'unknown'", hub_report.POPULATE)

    def test_the_evidence_array_is_expanded_once_per_row(self):
        """Two scalar subqueries deserialised the bulkiest column in the
        schema twice for every matched work; one LATERAL does it once.
        """
        self.assertEqual(hub_report.POPULATE.count("jsonb_array_elements"), 1)
        self.assertIn("LEFT JOIN LATERAL", hub_report.POPULATE)


class HubReportPopulateTests(unittest.TestCase):
    """The one consumer that reads the journal's depth-1 slice back.

    The aggregation is checked against the run it already produced: the
    statement runs against the SAME run id inside a rolled-back
    transaction, so "did the rewrite change any number" is answered by the
    data rather than by reading the SQL. Run 93 itself is never rewritten.
    """

    @classmethod
    def setUpClass(cls):
        cls.env = _live_env()
        cls.run_id = run_sql(
            cls.env,
            "SELECT id FROM measurements.run WHERE spike = "
            f"'{hub_report.SPIKE}';",
            extra_args=["-t", "-A"],
        ).stdout.strip()
        if not cls.run_id:
            raise unittest.SkipTest(f"no measurements.run for {hub_report.SPIKE}")
        cls.n_rows = cls._rows_of_the_run()

    def test_it_reproduces_the_recorded_measurement_row_for_row(self):
        columns = "work_key, relation, cited_by_count, n_references"
        script = (
            "BEGIN;\n"
            f"CREATE TEMP TABLE recorded AS SELECT {columns} "
            f"FROM measurements.citation_hub_expansion WHERE run_id = {self.run_id};\n"
            + hub_report.POPULATE.replace(":run", self.run_id) + "\n"
            f"WITH fresh AS (SELECT {columns} FROM measurements.citation_hub_expansion "
            f"               WHERE run_id = {self.run_id})\n"
            "SELECT (SELECT count(*) FROM recorded), (SELECT count(*) FROM fresh),\n"
            "       (SELECT count(*) FROM (SELECT * FROM fresh EXCEPT "
            "                              SELECT * FROM recorded) a),\n"
            "       (SELECT count(*) FROM (SELECT * FROM recorded EXCEPT "
            "                              SELECT * FROM fresh) b);\n"
            "ROLLBACK;\n"
        )
        out = run_sql(self.env, script, extra_args=["-t", "-A", "-F", FIELD_SEP]).stdout
        row = [line for line in out.splitlines() if line.strip()][-1]
        recorded, fresh, added, lost = row.split(FIELD_SEP)
        self.assertEqual(fresh, recorded)
        self.assertEqual((added, lost), ("0", "0"),
                         "перезапись POPULATE изменила записанный замер")

    @classmethod
    def _rows_of_the_run(cls) -> str:
        return run_sql(
            cls.env,
            "SELECT count(*) FROM measurements.citation_hub_expansion "
            f"WHERE run_id = {cls.run_id};",
            extra_args=["-t", "-A"],
        ).stdout.strip()

    def test_the_recorded_run_is_untouched_afterwards(self):
        """Against the count taken before the probe ran, never a literal.

        How many rows the recorded run has is data: the documented workflow
        rewrites it, and a number frozen here turns that into a red suite
        with no defect behind it. What the probe owes is that it changed
        nothing -- which is a comparison, not a constant.
        """
        self.assertEqual(self._rows_of_the_run(), self.n_rows)


class HubReportRelationTests(unittest.TestCase):
    """The hub measurement classifies every depth-1 node by how it entered.

    It used to take that from citation.work.evidence -- a blob
    registry.Node.absorb builds out of the source's own records -- with a
    coalesce(..., 'unknown') for the rows where the key was missing. The
    decision that classified the node is the journal's, so the journal is
    where the measurement reads it: this test makes the two disagree and
    checks which one the report believes.
    """

    KEY = "test:hubrel:W1"

    @classmethod
    def setUpClass(cls):
        cls.env = _live_env()
        run = run_sql(
            cls.env,
            "SELECT id FROM measurements.run WHERE spike = "
            f"'{hub_report.SPIKE}' ORDER BY id DESC LIMIT 1;",
            extra_args=["-t", "-A"],
        ).stdout.strip()
        if not run:
            raise unittest.SkipTest("замера hub-expansion в базе ещё нет")
        # An EXISTING run id, never a fresh one: measurements.run is a
        # BIGSERIAL, a rollback does not give the number back, and the
        # versioned dump next door carries its setval (kb/CLAUDE.md).
        cls.run_id = run

    def test_the_journal_outranks_the_evidence_blob(self):
        out = run_sql(
            self.env,
            "BEGIN;\n"
            "INSERT INTO citation.work (key, source, kind, evidence) VALUES "
            f"('{self.KEY}', 'test', 'external-skeleton', "
            """'{"relation": "referenced", "records": [{"cited_by_count": 7,
                  "referenced_works_count": 3}]}'::jsonb);\n"""
            "INSERT INTO citation.crawl_step (crawl_id, depth, action, node_key, relation) "
            f"VALUES ('test:hubrel', 1, 'keep', '{self.KEY}', 'cites');\n"
            + hub_report.POPULATE +
            "\nSELECT relation, cited_by_count FROM measurements.citation_hub_expansion "
            f"WHERE run_id = :run AND work_key = '{self.KEY}';\nROLLBACK;\n",
            variables={"run": self.run_id},
            extra_args=["-t", "-A", "-F", FIELD_SEP],
        ).stdout
        rows = [line for line in out.splitlines() if line.strip()]
        self.assertEqual(rows, [FIELD_SEP.join(["cites", "7"])])


if __name__ == "__main__":
    unittest.main()
