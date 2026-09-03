"""Tests for pg_graph_common.py (the plumbing pg_graph.py's CLI drives) and
the citation.cypher_literal / citation.project_graph SQL functions it wraps
(pg_schema_citation.sql).

Nothing here needs a database: compare_counts is pure Python, the CLI layer
is read as source, and the schema's shape is read as TEXT. What the live
AGE-backed graph actually does is test_pg_graph_live.py's, split off by
that seam (and by kb/CLAUDE.md FILE_SIZE).
"""
from __future__ import annotations

import ast
import unittest
from pathlib import Path
from unittest import mock

import _pathfix  # noqa: F401
import _pathfix_deploy  # noqa: F401

import citation_checks
import citation_profile
import pg_graph
import pg_graph_candidates
import pg_graph_cocitation
import pg_graph_common
import pg_graph_cypher


class CompareCountsTests(unittest.TestCase):
    """Pure function, no database: the arithmetic project() --check reports on."""

    def test_matching_counts_are_zero_diff(self):
        self.assertEqual(pg_graph_common.compare_counts(3, 2, 3, 2), (0, 0))

    def test_check_reports_count_mismatch(self):
        # Fewer vertices than work rows -- the graph is missing something.
        self.assertEqual(pg_graph_common.compare_counts(5, 2, 3, 2), (-2, 0))
        # More edges than cites rows -- the graph has something the relational
        # tables no longer do (e.g. a stale projection after a DELETE).
        self.assertEqual(pg_graph_common.compare_counts(3, 2, 3, 5), (0, 3))


class SchemaFileSplitTests(unittest.TestCase):
    """pg_schema_citation.sql grew past kb/CLAUDE.md's FILE_SIZE cap (code
    <= 300 lines) and was split by responsibility into four files: data
    definition, idempotent constraint migrations, AGE projection, journal
    backfill. What matters structurally is that each stays under the cap and
    that init_schema() applies all of them, in that fixed order (the
    constraints ALTER tables the first file declares; the backfill's UPDATEs
    read columns it adds; the projection functions are independent of both,
    but documented and bundled between them -- SCHEMA_PATHS order).
    """

    def test_every_schema_file_is_within_the_line_cap(self):
        for path in pg_graph_common.SCHEMA_PATHS:
            lines = path.read_text(encoding="utf-8").count("\n")
            self.assertLessEqual(lines, 300, f"{path.name}: {lines} lines")

    def test_init_schema_applies_every_file_in_order(self):
        applied = []
        with mock.patch.object(
            pg_graph_common, "run_sql_file",
            side_effect=lambda env, path: applied.append(path),
        ):
            pg_graph_common.init_schema({})
        self.assertEqual(applied, list(pg_graph_common.SCHEMA_PATHS))


class CliLayerTests(unittest.TestCase):
    """pg_graph.py parses arguments, dispatches and prints; the plumbing its
    consumers need lives in pg_graph_common.py. The one function-level
    import it is allowed is the optional paths.py shim (pg_search.py's main()
    carries the same one for the same reason: paths.py is not bundled into
    the deploy artifact). A deferred import of a graph module would mean the
    cycle that forced one is back.
    """

    SOURCE = Path(pg_graph.__file__).read_text(encoding="utf-8")
    TREE = ast.parse(SOURCE)

    def _function_level_imports(self) -> set[str]:
        names = set()
        for node in ast.walk(self.TREE):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for inner in ast.walk(node):
                if isinstance(inner, ast.Import):
                    names |= {alias.name for alias in inner.names}
                elif isinstance(inner, ast.ImportFrom):
                    names.add(inner.module or "")
        return names

    def test_no_import_lives_inside_a_function(self):
        self.assertEqual(self._function_level_imports(), set())

    def test_the_query_layer_is_imported_at_module_level(self):
        imported = {
            alias.name
            for node in self.TREE.body if isinstance(node, ast.Import)
            for alias in node.names
        }
        self.assertIn("pg_graph_candidates", imported)
        self.assertIn("pg_graph_cocitation", imported)
        self.assertIn("pg_graph_cypher", imported)
        self.assertIn("pg_graph_common", imported)

    def test_no_query_module_re_exports_another(self):
        """pg_graph_cypher owns citers/hybrid, pg_graph_candidates and
        pg_graph_cocitation own one relational consumer each, and a caller
        imports the module that owns the name. A facade re-exporting a
        neighbour's surface -- underscore-private names included -- makes
        the files one unit again, which is the coupling the split was made
        to remove.
        """
        modules = (pg_graph_candidates, pg_graph_cocitation, pg_graph_cypher)
        names = {module.__name__ for module in modules}
        for module in modules:
            tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
            imported = {node.module for node in ast.walk(tree)
                        if isinstance(node, ast.ImportFrom)}
            imported |= {alias.name for node in ast.walk(tree)
                         if isinstance(node, ast.Import) for alias in node.names}
            self.assertEqual(imported & names, set(),
                             f"{module.__name__} imports a sibling query module")

    def test_the_plumbing_is_not_defined_here(self):
        defined = {node.name for node in self.TREE.body
                   if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        for name in ("graph_sql", "graph_exists",
                     "projection_reading", "compare_counts", "project",
                     "init_schema"):
            self.assertNotIn(name, defined, f"{name}() belongs to pg_graph_common.py")
            self.assertTrue(hasattr(pg_graph_common, name))

    def test_the_printed_verdict_is_defined_here_and_not_in_the_plumbing(self):
        """check() is the one thing that moved the other way. It prints
        Russian status text and returns an exit code -- presentation, which
        the four consumers importing the plumbing as a library do not want,
        and which pushed pg_graph_common.py past the FILE_SIZE cap.
        """
        defined = {node.name for node in self.TREE.body
                   if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        self.assertIn("check", defined)
        self.assertFalse(hasattr(pg_graph_common, "check"))


class SharedCitationPlumbingTests(unittest.TestCase):
    """citation_schema_exists() and the kind census answer questions three
    consumers ask, so they are read from one place. Written verbatim in two
    modules once, together with FIELD_SEP, they are exactly the drift
    pg_graph_common.py exists to prevent -- hence the guard below, in the
    shape CliLayerTests uses on pg_graph.py.
    """

    CONSUMERS = (Path(citation_checks.__file__), Path(citation_profile.__file__))

    def test_schema_exists_reads_to_regclass(self):
        with mock.patch.object(pg_graph_common, "scalar", return_value="t") as scalar_mock:
            self.assertTrue(pg_graph_common.citation_schema_exists({}))
        self.assertIn("to_regclass('citation.work')", scalar_mock.call_args[0][1])
        with mock.patch.object(pg_graph_common, "scalar", return_value="f"):
            self.assertFalse(pg_graph_common.citation_schema_exists({}))

    def test_kind_census_parses_the_json_object(self):
        text = '{"external-skeleton": 382, "our-document": 56}'
        with mock.patch.object(pg_graph_common, "scalar", return_value=text):
            self.assertEqual(pg_graph_common.kind_counts({}),
                             {"external-skeleton": 382, "our-document": 56})

    def test_an_empty_table_censuses_as_an_empty_object(self):
        with mock.patch.object(pg_graph_common, "scalar", return_value="{}"):
            self.assertEqual(pg_graph_common.kind_counts({}), {})

    def test_kind_census_takes_the_callers_narrowing_clause(self):
        with mock.patch.object(pg_graph_common, "scalar",
                                return_value="{}") as scalar_mock:
            self.assertEqual(pg_graph_common.kind_counts({}, " WHERE w.year > 2000"), {})
        self.assertIn(" WHERE w.year > 2000", scalar_mock.call_args[0][1])

    def test_the_census_expression_is_what_the_statement_is_built_from(self):
        """A third caller reads the census inside a script of its own
        (citation_checks.py), so the expression is the shared thing and the
        one-statement form is built out of it.
        """
        with mock.patch.object(pg_graph_common, "scalar",
                                return_value="{}") as scalar_mock:
            pg_graph_common.kind_counts({})
        self.assertIn(pg_graph_common.kind_counts_expression(),
                      scalar_mock.call_args[0][1])

    def _definitions(self, path: Path) -> tuple[set[str], set[str]]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        functions = {node.name for node in tree.body
                     if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        assigned = {target.id for node in tree.body if isinstance(node, ast.Assign)
                    for target in node.targets if isinstance(target, ast.Name)}
        return functions, assigned

    def test_neither_consumer_redeclares_the_shared_primitives(self):
        for path in self.CONSUMERS:
            functions, assigned = self._definitions(path)
            self.assertNotIn("citation_schema_exists", functions, path.name)
            self.assertNotIn("kind_counts", functions, path.name)
            self.assertNotIn("counts_by_kind", functions, path.name)
            self.assertNotIn("FIELD_SEP", assigned, f"{path.name}: import it from pg_common")

    def test_neither_consumer_carries_its_own_copy_of_the_queries(self):
        for path in self.CONSUMERS:
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("to_regclass('citation.work')", text, path.name)
            self.assertNotIn("count(*) AS n FROM citation.work w", text, path.name)

    def _from_shared_layer(self, path: Path) -> set[str]:
        return {
            alias.name
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
            if isinstance(node, ast.ImportFrom) and node.module == "pg_graph_common"
            for alias in node.names
        }

    def test_both_consumers_reach_the_shared_layer(self):
        for path in self.CONSUMERS:
            self.assertIn("citation_schema_exists", self._from_shared_layer(path), path.name)

    def test_the_census_reader_reads_it_from_there_too(self):
        """One consumer left: the completeness check. The manifest's census
        was the other, and it is the dump's own answer now -- tallied off
        the COPY stream the packager wrote (deploy/copy_rows.FieldTally) and
        held to the shipped bytes by deploy/citation_cut_checks.py.
        """
        imported = self._from_shared_layer(Path(citation_checks.__file__))
        self.assertTrue({"kind_counts", "kind_counts_expression"} & imported, imported)
        self.assertNotIn("kind_counts",
                         self._from_shared_layer(Path(citation_profile.__file__)))


class ProjectionDiffTests(unittest.TestCase):
    """projection_diff() is the ONE reading behind "is the projection
    faithful"; check(), citation_checks._projection_stale() and
    deploy/smoke_checks.check_citation_projection() are three renderings of
    its result and own no reads of their own.
    """

    # Every module that could plausibly ask the question, checked against
    # the re-implementation signature rather than against a list of known
    # offenders: a NEW consumer assembling the sequence by hand is exactly
    # what this guard is for.
    SEARCHED = sorted(
        set(Path(pg_graph_common.__file__).resolve().parent.glob("*.py"))
        | set((Path(pg_graph_common.__file__).resolve().parent / "deploy").glob("*.py"))
        | set((Path(pg_graph_common.__file__).resolve().parent / "citations").glob("*.py"))
    )

    def test_returns_none_when_the_graph_was_never_projected(self):
        with mock.patch.object(pg_graph_common, "graph_exists", return_value=False), \
             mock.patch.object(pg_graph_common, "projection_reading") as read_mock:
            self.assertIsNone(pg_graph_common.projection_diff({}))
        read_mock.assert_not_called()

    def test_returns_the_four_counts_in_relational_then_graph_order(self):
        row = "438\x1f2425\x1f438\x1f2424\x1fa\x1fa\x1fb\x1fb"
        with mock.patch.object(pg_graph_common, "graph_exists", return_value=True), \
             mock.patch.object(pg_graph_common, "graph_sql",
                               return_value=mock.Mock(stdout=row)):
            seen = pg_graph_common.projection_diff({})
        self.assertEqual(seen[:4], (438, 2425, 438, 2424))

    def test_the_whole_reading_is_two_psql_invocations(self):
        """One guard ("is there a graph at all", which the reading itself
        cannot ask -- naming citation_graph."Work" fails outright when the
        graph was never projected) and one statement for everything else.
        Five of them used to answer the same question.
        """
        row = "438\x1f2425\x1f438\x1f2425\x1fa\x1fa\x1fb\x1fb"
        exists = mock.Mock(stdout="1")
        with mock.patch.object(pg_graph_common, "run_sql",
                               side_effect=[exists, mock.Mock(stdout=row)]) as run_mock, \
             mock.patch.object(pg_graph_common, "scalar") as scalar_mock:
            seen = pg_graph_common.projection_diff({})
        self.assertEqual(run_mock.call_count, 2)
        scalar_mock.assert_not_called()
        self.assertEqual(pg_graph_common.projection_faults(seen), [])

    def test_check_renders_the_diff_as_an_exit_code(self):
        # The content half of the reading lives in test_pg_graph_projection.py.
        # The renderer is the CLI's (pg_graph.check); what it renders is the
        # plumbing's.
        faithful = pg_graph_common.Projection(5, 3, 5, 3, "w", "w", "c", "c")
        with mock.patch.object(pg_graph_common, "projection_diff",
                               return_value=faithful):
            self.assertEqual(pg_graph.check({}), 0)
        with mock.patch.object(pg_graph_common, "projection_diff",
                               return_value=faithful._replace(vertex_n=4)):
            self.assertEqual(pg_graph.check({}), 1)
        with mock.patch.object(pg_graph_common, "projection_diff", return_value=None):
            self.assertEqual(pg_graph.check({}), 1)

    def _functions_assembling_the_sequence(self, path: Path) -> list[str]:
        """Function names in `path` that name BOTH graph_exists and the
        reading -- the signature of a hand-assembled projection read.
        """
        tree = ast.parse(path.read_text(encoding="utf-8"))
        offenders = []
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            names = {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)}
            names |= {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
            if {"graph_exists", "projection_reading"} <= names:
                offenders.append(node.name)
        return offenders

    def test_nobody_else_assembles_graph_exists_plus_the_reading(self):
        for path in self.SEARCHED:
            offenders = self._functions_assembling_the_sequence(path)
            if path.name == "pg_graph_common.py":
                self.assertEqual(offenders, ["projection_diff"], path.name)
                continue
            self.assertEqual(
                offenders, [],
                f"{path.name}: {offenders} re-derive the projection read -- "
                "call pg_graph_common.projection_diff() instead",
            )


class CrawlStepIndexTests(unittest.TestCase):
    """crawl_step is the largest table in the schema and the public cut
    matches two of its columns by equality, once per name it removes.
    """

    SCHEMA = pg_graph_common.SCHEMA_DEFINITION.read_text(encoding="utf-8")

    def test_the_cut_columns_are_indexed_idempotently(self):
        for column in ("frontier_key", "candidate_key"):
            self.assertIn(
                f"CREATE INDEX IF NOT EXISTS crawl_step_{column}_idx "
                f"ON citation.crawl_step ({column});",
                self.SCHEMA,
            )


class ProjectionShapeTests(unittest.TestCase):
    """The projection's COST is a property of its shape, and the shape is
    readable without a database: two bulk INSERT ... SELECT statements into
    the AGE label tables, not one Cypher command per row. The row-by-row
    form this replaced re-planned a command per row AND resolved each edge's
    endpoints with `MATCH (a:Work {key: ...})`, a sequential scan of the
    whole vertex label per edge (AGE indexes no property by itself).
    """

    SCHEMA = pg_graph_common.SCHEMA_GRAPH.read_text(encoding="utf-8")

    def test_labels_are_filled_by_bulk_insert(self):
        self.assertIn('INSERT INTO citation_graph."Work" (id, properties)', self.SCHEMA)
        self.assertIn(
            'INSERT INTO citation_graph."CITES" (id, start_id, end_id, properties)',
            self.SCHEMA,
        )

    def test_no_per_row_cypher_command_is_assembled(self):
        body = self.SCHEMA[self.SCHEMA.index("CREATE OR REPLACE FUNCTION citation.project_graph()"):]
        self.assertNotIn("FOR w IN", body)
        self.assertNotIn("FOR c IN", body)
        self.assertNotIn("MATCH (a:Work", body)
        self.assertNotIn("CREATE (:Work", body)

    def test_edge_endpoints_are_computed_from_the_relational_id(self):
        # _graphid(<Work label>, citation.work.id) -- so an edge's endpoints
        # are arithmetic, not a lookup by key.
        self.assertIn("ag_catalog._graphid($1, ci.citing)", self.SCHEMA)
        self.assertIn("ag_catalog._graphid($1, ci.cited)", self.SCHEMA)


if __name__ == "__main__":
    unittest.main()
