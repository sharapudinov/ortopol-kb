"""The crawl journal's machine-readable columns, on the live instance.

Split from test_pg_graph.py (kb/CLAUDE.md FILE_SIZE) along a real seam:
that file is about the plumbing pg_graph.py drives and the AGE projection,
this one is about citation.crawl_step's own columns -- the schema's
one-time parse of the facts that used to live in `reason`. The consumer
that reads them back is next door, in test_hub_report_columns.py.

JOURNAL_FACTS_ARE_COLUMNS is the invariant under test: what the pipeline
reads is a column, `reason` is prose for a human. Everything here needs a
live database (the parse is SQL, not Python) and skips when Postgres is
unreachable; every write happens inside a rolled-back transaction, so the
real journal is never touched.
"""
from __future__ import annotations

import unittest

import _pathfix  # noqa: F401
import pg_graph_common
from _citation_fixtures import work
from citations import candidates, journal, store
from citations.registry import WorkRegistry
from citations.scoring import NO_TEXT_SCORE
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


def _backfill_statements() -> list[str]:
    """Every one-time journal backfill, taken from the schema file itself.

    Read out of pg_schema_citation_backfill.sql rather than restated here: a
    copy would let the test keep passing over statements the schema no
    longer applies -- or miss one it has since grown.
    """
    schema = pg_graph_common.SCHEMA_BACKFILL.read_text(encoding="utf-8")
    found, at = [], 0
    while (start := schema.find("UPDATE citation.crawl_step SET", at)) != -1:
        at = schema.index(";", start) + 1
        found.append(schema[start:at])
    return found


class JournalBackfillTests(unittest.TestCase):
    """604 journal rows were written before node_key/score/tau -- and, later,
    relation/cited_by_count -- existed, with those facts inside `reason`. The
    schema parses them back into the columns: once, only into NULLs, and
    inside a rolled-back transaction here so the live journal is untouched.
    """

    @classmethod
    def setUpClass(cls):
        cls.env = _live_env()

    FIXTURE = """
    INSERT INTO citation.crawl_step (crawl_id, depth, frontier_key, candidate_key,
                                     action, reason)
    VALUES ('test:backfill', 1, 'W_F', 'W_C', 'keep',
            'kept; score=0.6123 tau=0.5000 relation=cites node=W_NODE'),
           ('test:backfill', 1, 'W_F', 'W_D', 'drop',
            'below-threshold; score=0.4001 tau=0.5000 relation=referenced'),
           ('test:backfill', 0, '2019_rm9846', 'W_EN', 'keep',
            'twin-of=2019_rm9846 seed=W_RU'),
           ('test:backfill', 2, 'W_H', NULL, 'hub-skip',
            'cited_by_count=5000 > cap 1000'),
           ('test:backfill', 1, 'W_F', NULL, 'fetch', NULL);
    """

    def _backfilled(self, repeats: int = 1) -> list[str]:
        out = run_sql(
            self.env,
            "BEGIN;\n" + self.FIXTURE + "".join(_backfill_statements()) * repeats + "\n"
            "SELECT action, coalesce(node_key, '-'), coalesce(score::text, '-'), "
            "coalesce(tau::text, '-'), coalesce(relation, '-'), "
            "coalesce(cited_by_count::text, '-') FROM citation.crawl_step "
            "WHERE crawl_id = 'test:backfill' ORDER BY id;\nROLLBACK;\n",
            extra_args=["-t", "-A", "-F", FIELD_SEP],
        ).stdout
        return [line for line in out.splitlines() if line.strip()]

    def test_prose_becomes_columns_row_by_row(self):
        self.assertEqual(self._backfilled(), [
            FIELD_SEP.join(["keep", "W_NODE", "0.6123", "0.5", "cites", "-"]),
            FIELD_SEP.join(["drop", "-", "0.4001", "0.5", "referenced", "-"]),
            FIELD_SEP.join(["keep", "W_RU", "-", "-", "-", "-"]),
            FIELD_SEP.join(["hub-skip", "-", "-", "-", "-", "5000"]),
            FIELD_SEP.join(["fetch", "-", "-", "-", "-", "-"]),
        ])

    def test_running_it_again_changes_nothing(self):
        self.assertEqual(self._backfilled(repeats=2), self._backfilled())

    def _rows_written(self, passes: int) -> list[int]:
        """How many rows each successive pass of the backfill REWRITES.

        Value-idempotent is not enough: the schema file is applied in full
        on every `pg_graph.py init` and every non-dry-run crawl, so a guard
        that keeps matching a row the parse cannot fill any further rewrites
        it forever -- a new tuple version, WAL, and index maintenance on the
        three key indexes, on a table that is otherwise append-only.
        """
        statements = _backfill_statements()
        counted = "".join("WITH touched AS (" + statement.rstrip(";\n ")
                          + " RETURNING 1) SELECT count(*) FROM touched;\n"
                          for statement in statements)
        out = run_sql(
            self.env,
            "BEGIN;\n" + self.FIXTURE + counted * passes + "ROLLBACK;\n",
            extra_args=["-t", "-A"],
        ).stdout
        each = [int(line) for line in out.splitlines() if line.strip()]
        step = len(statements)
        return [sum(each[i:i + step]) for i in range(0, len(each), step)]

    def test_the_second_pass_writes_no_row_at_all(self):
        written = self._rows_written(3)
        # The fixture's parseable rows, plus whatever the live journal still
        # has to fill -- the first pass is the one allowed to write.
        self.assertGreaterEqual(written[0], 6)
        self.assertEqual(written[1:], [0, 0], "backfill переписывает те же строки заново")

    def test_the_live_journal_has_no_unparsed_relation_left(self):
        """Every keep/drop row names how its candidate reached the frontier.

        A hub-skip row has no relation (nothing was kept), and neither has a
        twin promotion, which is a statement about the corpus rather than
        about a traversal edge: the record IS one of our own works, it did
        not cite or get cited into the frontier. Those are the rows
        legitimately without one -- measured on the live journal, they are
        the only ones.
        """
        left = run_sql(
            self.env,
            "SELECT count(*) FROM citation.crawl_step WHERE relation IS NULL "
            "AND action IN ('keep', 'drop') "
            "AND reason NOT IN ('двойник нашей работы') AND reason !~ 'twin-of=';",
            extra_args=["-t", "-A"],
        ).stdout.strip()
        self.assertEqual(left, "0")

    def test_the_live_journal_has_no_unparsed_score_left(self):
        """The columns are filled wherever the prose ever carried them --
        the state the migration is supposed to leave the base in."""
        left = run_sql(
            self.env,
            "SELECT count(*) FROM citation.crawl_step "
            "WHERE reason ~ 'score=' AND score IS NULL;",
            extra_args=["-t", "-A"],
        ).stdout.strip()
        self.assertEqual(left, "0")


class ActionVocabularyGuardTests(unittest.TestCase):
    """The action vocabulary is a named CHECK constraint, and re-adding one
    is not free: Postgres validates it against every existing row under an
    ACCESS EXCLUSIVE lock. The journal grows by ~100k rows per depth-2 crawl
    and the schema is applied on every `pg_graph.py init` AND every
    non-dry-run crawl, so an unconditional DROP+ADD is the same unbounded
    scan the reason-parse backfill was guarded against one section below.

    Applied inside a rolled-back transaction; the live constraint is read
    but never replaced.
    """

    @classmethod
    def setUpClass(cls):
        cls.env = _live_env()
        cls.schema = pg_graph_common.SCHEMA_CONSTRAINTS.read_text(encoding="utf-8")

    DEFINITION = (
        "SELECT c.oid::text, c.xmin::text, pg_get_constraintdef(c.oid) "
        "FROM pg_constraint c WHERE c.conname = 'crawl_step_action_check';"
    )

    def _probe(self, script: str) -> list[str]:
        out = run_sql(self.env, "BEGIN;\n" + script + "\nROLLBACK;\n",
                      extra_args=["-t", "-A", "-F", FIELD_SEP]).stdout
        return [line for line in out.splitlines() if line.strip()]

    def test_a_repeat_apply_leaves_the_same_constraint_in_place(self):
        rows = self._probe(self.schema + self.DEFINITION
                           + self.schema + self.DEFINITION)
        self.assertEqual(rows[0], rows[1],
                         "constraint пересоздан: oid/xmin изменились")

    def test_the_live_constraint_already_matches_the_file(self):
        """Not a property of the file but of THIS instance: if the live
        vocabulary differed, every apply here would legitimately rewrite it
        and the test above would be measuring a no-op it created itself.
        """
        before = run_sql(self.env, self.DEFINITION,
                         extra_args=["-t", "-A", "-F", FIELD_SEP]).stdout.strip()
        after = self._probe(self.schema + self.DEFINITION)[0]
        self.assertEqual(before, after)

    def test_a_stale_vocabulary_is_replaced(self):
        """The complement: the guard must not be "never touch it". A
        constraint created before hub-skip existed is exactly the case
        DROP+ADD was written for. NOT VALID only because the live journal
        already carries the values the narrow fixture forbids.
        """
        stale = (
            "ALTER TABLE citation.crawl_step DROP CONSTRAINT "
            "crawl_step_action_check;\n"
            "ALTER TABLE citation.crawl_step ADD CONSTRAINT "
            "crawl_step_action_check CHECK (action IN ('seed', 'fetch')) "
            "NOT VALID;\n"
        )
        rows = self._probe(stale + self.DEFINITION + self.schema + self.DEFINITION)
        self.assertNotIn("hub-skip", rows[0])
        self.assertIn("hub-skip", rows[1])
        self.assertNotEqual(rows[0].split(FIELD_SEP)[0], rows[1].split(FIELD_SEP)[0])


class BackfillRegistryTests(unittest.TestCase):
    """The parse runs once per database, and citation.schema_backfill is how
    the schema knows it already did.

    Value-idempotence was never the gap: each UPDATE fills only a NULL. The
    gap was that the WHERE clauses are regex predicates no index can serve,
    so every apply -- every `pg_graph.py init` and every non-dry-run crawl
    -- re-scanned the whole journal to discover there was nothing to fill.
    Everything below runs inside a rolled-back transaction; the live
    registry row and the live journal survive untouched.
    """

    @classmethod
    def setUpClass(cls):
        cls.env = _live_env()
        cls.schema = pg_graph_common.SCHEMA_BACKFILL.read_text(encoding="utf-8")

    BACKFILL_NAME = "crawl_step_reason_parse"
    # One row the parse WOULD fill: prose carrying score/tau/relation/node
    # with every column still NULL.
    FIXTURE = """
    INSERT INTO citation.crawl_step (crawl_id, depth, frontier_key, candidate_key,
                                     action, reason)
    VALUES ('test:backfill-registry', 1, 'W_F', 'W_C', 'keep',
            'kept; score=0.6123 tau=0.5000 relation=cites node=W_NODE');
    """
    PROBE = (
        "SELECT coalesce(node_key, '-'), coalesce(score::text, '-'), "
        "coalesce(relation, '-') FROM citation.crawl_step "
        "WHERE crawl_id = 'test:backfill-registry';"
    )
    FORGET = f"DELETE FROM citation.schema_backfill WHERE name = '{BACKFILL_NAME}';"

    def _probe(self, script: str) -> list[str]:
        out = run_sql(self.env, "BEGIN;\n" + script + "\nROLLBACK;\n",
                      extra_args=["-t", "-A", "-F", FIELD_SEP]).stdout
        return [line for line in out.splitlines() if line.strip()]

    def test_an_unrecorded_backfill_runs_and_records_itself(self):
        rows = self._probe(
            self.FORGET + self.FIXTURE + self.schema
            + f"SELECT name FROM citation.schema_backfill WHERE name = '{self.BACKFILL_NAME}';"
            + self.PROBE
        )
        self.assertEqual(rows[0], self.BACKFILL_NAME)
        self.assertEqual(rows[1], FIELD_SEP.join(["W_NODE", "0.6123", "cites"]))

    def test_a_recorded_backfill_does_no_work_at_all(self):
        """A row the parse WOULD fill, inserted after the registry entry
        exists, comes out untouched -- which is only possible if the UPDATEs
        never ran.
        """
        rows = self._probe(
            self.FORGET + self.schema      # first apply: records the parse
            + self.FIXTURE + self.schema   # second apply: must skip it
            + self.PROBE
        )
        self.assertEqual(rows, [FIELD_SEP.join(["-", "-", "-"])])

    def test_the_registry_row_is_not_rewritten_by_a_repeat_apply(self):
        rows = self._probe(
            self.FORGET + self.schema
            + f"SELECT xmin::text FROM citation.schema_backfill "
              f"WHERE name = '{self.BACKFILL_NAME}';"
            + self.schema
            + f"SELECT xmin::text FROM citation.schema_backfill "
              f"WHERE name = '{self.BACKFILL_NAME}';"
        )
        self.assertEqual(rows[0], rows[1])

    def test_the_live_database_carries_the_record(self):
        """Not a property of the file but of THIS instance: the migration
        has to have been applied here, or the guard is untested in practice.
        """
        out = run_sql(
            self.env,
            "SELECT count(*) FROM citation.schema_backfill "
            f"WHERE name = '{self.BACKFILL_NAME}';",
            extra_args=["-t", "-A"],
        ).stdout.strip()
        self.assertEqual(out, "1")


class DropReasonSaysWhichDropItWasTests(unittest.TestCase):
    """Offline: the one journal builder whose prose is not a constant.

    `action` and `score` are the columns; the sentence beside them is what a
    human reads, and it was wrong on one whole class of row -- a candidate
    with no title is never measured against tau (citations/candidates.py
    scores it NO_TEXT_SCORE), so "below-threshold" claimed a comparison the
    crawl never made. Everything machine-readable stays where it was: the
    two cases are told apart by the score column, here and in the schema's
    one-time rewrite alike.
    """

    def test_a_candidate_that_failed_tau_says_so(self):
        step = journal.drop("c", 1, "W1", 0.4001, 0.5, "cites")
        self.assertEqual(step["reason"], journal.BELOW_THRESHOLD_REASON)
        self.assertEqual(step["score"], 0.4001)

    def test_a_candidate_with_nothing_to_embed_says_that_instead(self):
        step = journal.drop("c", 1, "W1", NO_TEXT_SCORE, 0.5, "cites")
        self.assertEqual(step["reason"], journal.NO_TEXT_REASON)
        self.assertEqual(step["score"], NO_TEXT_SCORE)
        self.assertEqual(step["action"], journal.CrawlAction.DROP)

    def test_the_untitled_candidate_the_crawl_produces_is_that_case(self):
        """Not a hand-built number: the score comes from the scoring pass
        itself, for a record with no title at all.
        """
        registry = WorkRegistry()
        untitled = work("W_NO_TEXT", title="")
        scored = candidates.score(registry, lambda holders: {}, [(untitled, "cites", ())])
        self.assertEqual([item["score"] for item in scored], [NO_TEXT_SCORE])
        self.assertEqual(journal.drop_reason(scored[0]["score"]), journal.NO_TEXT_REASON)


class NoTextReasonBackfillTests(unittest.TestCase):
    """The live half of the same distinction: the rows written before the
    builder made it.

    Applied as the schema applies it -- the whole file, guarded by
    citation.schema_backfill -- inside a rolled-back transaction, so neither
    the live journal nor the live registry is touched.
    """

    NAME = "crawl_step_no_text_reason"

    @classmethod
    def setUpClass(cls):
        cls.env = _live_env()
        cls.schema = pg_graph_common.SCHEMA_BACKFILL.read_text(encoding="utf-8")

    FIXTURE = f"""
    INSERT INTO citation.crawl_step (crawl_id, depth, candidate_key, action,
                                     score, tau, relation, reason)
    VALUES ('test:no-text', 2, 'W_A', 'drop', -1, 0.5, 'cites', 'below-threshold'),
           ('test:no-text', 2, 'W_B', 'drop', 0.4001, 0.5, 'cites', 'below-threshold'),
           ('test:no-text', 2, 'W_C', 'drop', -1, 0.5, 'cites', 'что-то своё');
    """
    PROBE = ("SELECT candidate_key, reason FROM citation.crawl_step "
             "WHERE crawl_id = 'test:no-text' ORDER BY candidate_key;")

    def _probe(self, script: str) -> list[str]:
        out = run_sql(self.env, "BEGIN;\n" + script + "\nROLLBACK;\n",
                      extra_args=["-t", "-A", "-F", FIELD_SEP]).stdout
        return [line for line in out.splitlines() if line.strip()]

    def _forget(self) -> str:
        return f"DELETE FROM citation.schema_backfill WHERE name = '{self.NAME}';"

    def test_only_the_unmeasured_row_is_reworded(self):
        rows = self._probe(self._forget() + self.FIXTURE + self.schema + self.PROBE)
        self.assertEqual(rows, [
            FIELD_SEP.join(["W_A", journal.NO_TEXT_REASON]),
            FIELD_SEP.join(["W_B", journal.BELOW_THRESHOLD_REASON]),
            FIELD_SEP.join(["W_C", "что-то своё"]),
        ])

    def test_a_recorded_rewrite_does_no_work_at_all(self):
        rows = self._probe(self._forget() + self.schema
                           + self.FIXTURE + self.schema + self.PROBE)
        self.assertEqual(rows[0], FIELD_SEP.join(["W_A", journal.BELOW_THRESHOLD_REASON]))

    def test_the_live_database_carries_the_record(self):
        """A property of THIS instance, like the parse's own: applied here,
        or the 14 rows it exists for still misname themselves.
        """
        out = run_sql(
            self.env,
            "SELECT count(*) FROM citation.schema_backfill "
            f"WHERE name = '{self.NAME}';",
            extra_args=["-t", "-A"],
        ).stdout.strip()
        self.assertEqual(out, "1")

    def test_no_drop_row_in_the_live_journal_still_misnames_itself(self):
        out = run_sql(
            self.env,
            "SELECT count(*) FROM citation.crawl_step WHERE action = 'drop' "
            f"AND score <= -1 AND reason = '{journal.BELOW_THRESHOLD_REASON}';",
            extra_args=["-t", "-A"],
        ).stdout.strip()
        self.assertEqual(out, "0")


class StepColumnsTests(unittest.TestCase):
    """The write seam between journal.py and the table, pinned at both ends.

    store.STEP_COLUMNS is the column list the COPY names, and a step field
    missing from it is dropped in silence -- the value is journalled by the
    code and absent from the database, which is worse than not recording it
    at all. The other end is the catalog: a column list naming something the
    table does not have fails the whole batch, on a crawl, at the end.
    """

    STEPS = (
        journal.seed("c", "doc", "W1"),
        journal.seed_missing("c", "doc"),
        journal.seed_error("c", "doc", "W1"),
        journal.zbmath_error("c", "doc", "Z1", "HTTP 429"),
        journal.keep("c", 2, "W1", "W_NODE", 0.61, 0.5, "cites", "W_F"),
        journal.drop("c", 2, "W2", 0.40, 0.5, "referenced", "W_F"),
        journal.fetch("c", 2, "W_F", 10, 3),
        journal.hub_skip("c", 2, "W_H", 5000, 1000),
        journal.twin("c", "W_EN", "2019_rm9846", "W_RU"),
    )

    def test_no_journal_field_is_dropped_by_the_writer(self):
        fields = set().union(*(set(step) for step in self.STEPS))
        self.assertEqual(fields - set(store.STEP_COLUMNS), set())

    def test_the_writer_names_only_columns_the_table_has(self):
        env = _live_env()
        out = run_sql(
            env,
            "SELECT a.attname FROM pg_attribute a JOIN pg_class c ON c.oid = a.attrelid "
            "JOIN pg_namespace n ON n.oid = c.relnamespace "
            "WHERE n.nspname = 'citation' AND c.relname = 'crawl_step' "
            "AND a.attnum > 0 AND NOT a.attisdropped;",
            extra_args=["-t", "-A"],
        ).stdout
        columns = {line.strip() for line in out.splitlines() if line.strip()}
        self.assertEqual(set(store.STEP_COLUMNS) - columns, set())


if __name__ == "__main__":
    unittest.main()
