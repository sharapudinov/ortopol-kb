"""Where the crawl's pieces are allowed to live, checked instead of agreed.

The citations package declares a split in its own __init__ (clients do HTTP,
inputs.py does the reads that establish a run, store.py is the write seam,
crawl.py is the traversal) and paths.py declares that 'theory/iis' is spelled
once. Both are the kind of rule a new call site breaks silently: a duplicated
source_dir literal returns zero rows rather than failing, and pipeline code
in the argparse module simply cannot be called without argparse.
"""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

import _pathfix  # noqa: F401
import paths
import pg_load_citations
from citations import inputs

CITATIONS_DIR = Path(inputs.__file__).resolve().parent
LOADER = Path(pg_load_citations.__file__).resolve()
SOURCES = sorted(CITATIONS_DIR.glob("*.py")) + [LOADER]


class SourceDirLiteralTests(unittest.TestCase):
    """paths.IIS_SOURCE_DIR exists so the string appears once. A second
    spelling of it queries corpus.documents.source_dir for a directory
    nothing was loaded under: zero rows, no error, a seed set silently
    emptied and a twin index silently unanchored.
    """

    def test_no_module_spells_the_source_directory_itself(self):
        for path in SOURCES:
            text = path.read_text(encoding="utf-8")
            self.assertNotIn(
                paths.IIS_SOURCE_DIR, text,
                f"{path.name}: import paths.IIS_SOURCE_DIR instead of the literal",
            )

    def test_the_default_is_the_constant_itself(self):
        tree = ast.parse((CITATIONS_DIR / "inputs.py").read_text(encoding="utf-8"))
        defaults = {
            node.name: [d for d in node.args.defaults]
            for node in tree.body if isinstance(node, ast.FunctionDef)
        }
        for name in ("corpus_seed_documents", "corpus_document_ids"):
            self.assertTrue(
                any(isinstance(d, ast.Name) and d.id == "IIS_SOURCE_DIR"
                    for d in defaults[name]),
                f"{name}: source_dir must default to paths.IIS_SOURCE_DIR",
            )


class SeedPredicateTests(unittest.TestCase):
    """"What counts as a seed document" is one predicate. Written twice, a
    change applied to one copy produces seeds with no Math-Net title anchor
    -- which is precisely what the twin rule depends on.
    """

    PREDICATE = "extraction_state <> 'metadata'"

    def test_the_predicate_appears_in_exactly_one_module(self):
        carriers = [p.name for p in SOURCES
                    if self.PREDICATE in p.read_text(encoding="utf-8")]
        self.assertEqual(carriers, ["inputs.py"])

    def test_both_readings_go_through_that_one_query(self):
        seen = []
        rows = "doc_a\x1fhttps://mathnet.ru/eng/sm123\ndoc_b\x1f\n"

        class _Result:
            stdout = rows

        def fake_run_sql(env, sql, **kwargs):
            seen.append(sql)
            return _Result()

        original = inputs.run_sql
        inputs.run_sql = fake_run_sql
        try:
            self.assertEqual(inputs.corpus_document_ids({}), ["doc_a", "doc_b"])
            self.assertEqual(
                inputs.corpus_seed_documents({}),
                [("doc_a", "https://mathnet.ru/eng/sm123"), ("doc_b", "")],
            )
        finally:
            inputs.run_sql = original
        self.assertEqual(seen, [inputs._SEED_DOCUMENTS_SQL] * 2)


class LoaderIsADispatcherTests(unittest.TestCase):
    """pg_load_citations.py parses flags, constructs and dispatches. The
    run-establishing pipeline (a SQL read, a client, journal writes and the
    reporting around them) belongs beside the rest of the package, where it
    is callable without a command line.
    """

    TREE = ast.parse(LOADER.read_text(encoding="utf-8"))

    def test_the_seed_metadata_pipeline_is_not_defined_here(self):
        defined = {node.name for node in self.TREE.body
                   if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        for name in ("zbmath_abstracts", "mathnet_names"):
            self.assertNotIn(name, defined, f"{name}() belongs to citations/seed_metadata.py")

    def test_the_loader_issues_no_sql_of_its_own(self):
        text = LOADER.read_text(encoding="utf-8")
        self.assertNotIn("run_sql", text)
        self.assertNotIn("SELECT ", text)

    def test_it_still_dispatches_to_both(self):
        imported = {alias.name for node in ast.walk(self.TREE)
                    if isinstance(node, ast.ImportFrom)
                    and node.module == "citations.seed_metadata"
                    for alias in node.names}
        self.assertEqual(imported, {"mathnet_names", "zbmath_abstracts"})


if __name__ == "__main__":
    unittest.main()
