"""deploy/copy_writer.py: ONE writer of a resolved COPY block, for both
schemas.

Split off test_citation_dump.py / test_public_dump.py (kb/CLAUDE.md
FILE_SIZE) along the seam the code now has: those two test what each half
decides -- which rows travel and how each column is projected -- and this
tests what happens to a block once decided, which is the same thing for
either schema.

The two used to be one writer per schema, identical but for the schema
literal in the header and the optional tally. This file therefore asserts
the property that keeps them one: the schema comes off the BLOCK, and
deploy/ holds exactly one definition of write_copy_block.
"""
from __future__ import annotations

import ast
import io
import pathlib
import unittest
from unittest import mock

import _pathfix  # noqa: F401
import _pathfix_deploy  # noqa: F401

import citation_dump
import copy_writer
import corpus_cut
from copy_plan import CopyBlock
from manifest_contract import CitationMode

DEPLOY_DIR = pathlib.Path(copy_writer.__file__).resolve().parent


def _write(block: CopyBlock, rows: bytes = b"", tally=None) -> tuple[str, int]:
    buffer = io.BytesIO()
    with mock.patch.object(copy_writer, "stream_stdout",
                           side_effect=lambda argv, env, dst: dst.write(rows)):
        written = copy_writer.write_copy_block({}, buffer, block, tally)
    return buffer.getvalue().decode(), written


class OneWriterForBothSchemasTests(unittest.TestCase):
    def test_the_header_names_the_blocks_own_schema(self):
        """Neither schema is a literal in the writer: the same call writes
        `corpus.` or `citation.` because the block says which.
        """
        for schema, table in ((corpus_cut.SCHEMA, "documents"),
                              (citation_dump.SCHEMA, "work")):
            with self.subTest(schema=schema):
                text, _rows = _write(CopyBlock(schema, table, ["id", "key"], (), "COPY ..."))
                self.assertTrue(text.startswith(f"COPY {schema}.{table} (id, key) FROM stdin;\n"),
                                text)

    def test_the_setval_names_it_too(self):
        text, _rows = _write(CopyBlock(corpus_cut.SCHEMA, "pages", ["id"], ("id",), "COPY ..."))
        self.assertIn("pg_get_serial_sequence('corpus.pages', 'id')", text)
        text, _rows = _write(CopyBlock(citation_dump.SCHEMA, "work", ["id"], ("id",), "COPY ..."))
        self.assertIn("pg_get_serial_sequence('citation.work', 'id')", text)

    def test_deploy_defines_the_writer_exactly_once(self):
        """The property, not today's arrangement: two hand-kept twins agree
        only by accident, and this is the one seam of the dump that was not
        parameterised by schema.
        """
        defined = []
        for path in sorted(DEPLOY_DIR.glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            defined += [f"{path.name}:{node.name}" for node in ast.walk(tree)
                        if isinstance(node, ast.FunctionDef) and node.name == "write_copy_block"]
        self.assertEqual(defined, ["copy_writer.py:write_copy_block"])


class BlockShapeTests(unittest.TestCase):
    """pg_dump's own shape, so the result restores through plain psql."""

    def test_serial_table_writes_setval_after_the_terminator(self):
        text, _rows = _write(
            CopyBlock(citation_dump.SCHEMA, "work", ["id", "key"], ("id",),
                      citation_dump.copy_select("work", ["id", "key"],
                                                CitationMode.FULL_SKELETON)),
            rows=b"1\tk1\n")
        self.assertIn("COPY citation.work (id, key) FROM stdin;\n1\tk1\n\\.\n", text)
        self.assertLess(text.index("\\.\n"), text.index("setval"))

    def test_non_serial_table_writes_no_setval(self):
        columns = ["citing", "cited", "source"]
        text, _rows = _write(
            CopyBlock(citation_dump.SCHEMA, "cites", columns, (),
                      citation_dump.copy_select("cites", columns, CitationMode.FULL_SKELETON)),
            rows=b"1\t2\tmanual\n")
        self.assertNotIn("setval", text)

    def test_the_server_side_copy_is_what_streams(self):
        with mock.patch.object(copy_writer, "stream_stdout") as stream_mock:
            copy_writer.write_copy_block(
                {}, io.BytesIO(),
                CopyBlock(corpus_cut.SCHEMA, "pages", ["document_id"], (),
                          corpus_cut.copy_select("pages", ["document_id"])))
        (argv, _env, _dst), _kwargs = stream_mock.call_args
        self.assertEqual(argv[0], "psql")
        self.assertIn("COPY (SELECT", argv[-1])
        self.assertIn("TO STDOUT", argv[-1])


class RowsAreCountedAsTheyPassTests(unittest.TestCase):
    """The manifest's numbers and the file's COPY blocks are one fact: the
    count is what streamed, never a second reading of the live database
    (MANIFEST_DESCRIBES_ARTIFACT, copy_rows.py).
    """

    def test_a_block_reports_the_rows_it_streamed(self):
        text, rows = _write(
            CopyBlock(citation_dump.SCHEMA, "work", ["id", "key"], (), "COPY ..."),
            rows=b"a\nb\nc\n")
        self.assertEqual(rows, 3)
        self.assertIn("COPY citation.work (id, key) FROM stdin;\na\nb\nc\n\\.\n", text)

    def test_an_empty_block_reports_zero(self):
        _text, rows = _write(CopyBlock(citation_dump.SCHEMA, "work", ["id"], (), "COPY ..."))
        self.assertEqual(rows, 0)


if __name__ == "__main__":
    unittest.main()
