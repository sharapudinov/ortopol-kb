"""deploy/public_dump.py's handshake with the citation half of the dump.

Split from test_public_dump.py for module size (kb/CLAUDE.md FILE_SIZE)
along the seam the modules themselves keep: dump_public() decides WHETHER
and under which mode citation_dump.dump_citation() writes, and that is what
is asserted here. HOW one mode's worth of citation.* is projected --
which columns are blanked, which rows are cut -- is citation_dump.py's own
and belongs to test_citation_dump.py; the corpus half's shape is next door.
"""
from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import _pathfix  # noqa: F401
import _pathfix_deploy  # noqa: F401
from _dump_fixtures import CORPUS_COLUMNS, CORPUS_SERIALS, DUMPED_CITATION_TABLES

import citation_columns
import citation_dump
import citation_profile
import dump_scan
import public_dump
import schema_catalog
from manifest_contract import CitationMode, Profile, schemas_for


class CitationDumpIntegrationTests(unittest.TestCase):
    """dump_public()'s citation-schema gate and dispatch -- the column-level
    blanking behaviour under CitationMode.TOPOLOGY_ONLY belongs to
    citation_dump.py's own tests (test_citation_dump.py); this covers the
    handshake between the two modules.
    """

    CITATION_COLUMNS = {
        "work": ["id", "key", "title", "abstract", "evidence"],
        "cites": ["citing", "cited", "evidence"],
        "crawl_step": ["id", "crawl_id"],
        "public_policy": ["id", "mode", "note"],
        "schema_backfill": ["name", "applied_at"],
    }

    def _fake_stream(self, argv, env, dst):
        dst.write(b"-- DDL\n" if "pg_dump" in argv[0] else b"row\n")

    def _run(self, tmp, citation_mode):
        gz_path = Path(tmp) / "01_dump.sql.gz"
        with mock.patch.object(public_dump, "require_classified"), \
             mock.patch.object(public_dump, "corpus_tables",
                                return_value=list(CORPUS_COLUMNS)), \
             mock.patch.object(public_dump.schema_catalog, "schema_columns",
                                return_value=dict(CORPUS_COLUMNS)), \
             mock.patch.object(public_dump.schema_catalog, "schema_serial_columns",
                                return_value=CORPUS_SERIALS), \
             mock.patch.object(public_dump, "stream_stdout", side_effect=self._fake_stream), \
             mock.patch.object(citation_dump, "citation_tables",
                                return_value=list(DUMPED_CITATION_TABLES)), \
             mock.patch.object(citation_dump, "schema_columns",
                                return_value=dict(self.CITATION_COLUMNS)), \
             mock.patch.object(citation_dump, "schema_serial_columns",
                                    return_value={}), \
             mock.patch.object(citation_dump, "stream_stdout", side_effect=self._fake_stream):
            public_dump.dump_public({}, gz_path, citation_mode=citation_mode)
        return gz_path

    def test_none_mode_ships_no_citation_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            gz_path = self._run(tmp, CitationMode.NONE)
            self.assertEqual(dump_scan.schema_names(gz_path), {"corpus"})

    def test_full_skeleton_mode_ships_citation_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            gz_path = self._run(tmp, CitationMode.FULL_SKELETON)
            self.assertEqual(dump_scan.schema_names(gz_path), {"corpus", "citation"})

    def test_every_mode_dumps_exactly_what_the_manifest_will_declare(self):
        """MANIFEST_DESCRIBES_ARTIFACT, checked against the dump's own bytes:
        manifest.json's schemas[] is schemas_for(), and so is this.
        """
        for mode in CitationMode.ALL:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                gz_path = self._run(tmp, mode)
                self.assertEqual(dump_scan.schema_names(gz_path),
                                 set(schemas_for(Profile.PUBLIC, mode)))

    def test_topology_only_blanks_abstracts_and_evidence(self):
        # End-to-end through dump_public() -- the column-level SQL itself is
        # test_citation_dump.py's CopySelectTests; this confirms the actual
        # bytes public_dump.dump_public() produces carry no abstract/evidence.
        gz_path = None
        seen_selects = []

        def capturing_stream(argv, env, dst):
            if argv[0] == "psql":
                seen_selects.append(argv[-1])
            self._fake_stream(argv, env, dst)

        with tempfile.TemporaryDirectory() as tmp:
            gz_path = Path(tmp) / "01_dump.sql.gz"
            with mock.patch.object(public_dump, "require_classified"), \
                 mock.patch.object(public_dump, "corpus_tables",
                                    return_value=list(CORPUS_COLUMNS)), \
                 mock.patch.object(public_dump.schema_catalog, "schema_columns",
                                    return_value=dict(CORPUS_COLUMNS)), \
                 mock.patch.object(public_dump.schema_catalog, "schema_serial_columns",
                                    return_value=CORPUS_SERIALS), \
                 mock.patch.object(public_dump, "stream_stdout", side_effect=capturing_stream), \
                 mock.patch.object(citation_dump, "citation_tables",
                                    return_value=list(DUMPED_CITATION_TABLES)), \
                 mock.patch.object(citation_dump, "schema_columns",
                                    return_value=dict(self.CITATION_COLUMNS)), \
                 mock.patch.object(citation_dump, "schema_serial_columns",
                                    return_value={}), \
                 mock.patch.object(citation_dump, "stream_stdout", side_effect=capturing_stream):
                public_dump.dump_public({}, gz_path, citation_mode=CitationMode.TOPOLOGY_ONLY)
        work_select = next(s for s in seen_selects if "citation.work" in s)
        cites_select = next(s for s in seen_selects if "citation.cites" in s)
        self.assertIn("NULL::text AS abstract", work_select)
        self.assertIn("NULL::jsonb AS evidence", work_select)
        self.assertIn("NULL::jsonb AS evidence", cites_select)

    def test_the_dump_never_reads_the_policy_itself(self):
        """The mode arrives as a value from build_package.main(), which
        resolved it once for the manifest and the dump alike. If this module
        asked the database again, the two could disagree -- so the policy
        reader is made to explode here and the dump must not notice.
        """
        with mock.patch.object(citation_profile, "require_citation_mode",
                                side_effect=AssertionError("policy re-read by the dump")), \
             mock.patch.object(citation_profile, "citation_public_policy",
                                side_effect=AssertionError("policy re-read by the dump")):
            with tempfile.TemporaryDirectory() as tmp:
                gz_path = self._run(tmp, CitationMode.FULL_SKELETON)
                self.assertEqual(dump_scan.schema_names(gz_path), {"corpus", "citation"})

    def test_dumps_never_carry_age_catalog(self):
        # Defense in depth on top of --schema already whitelisting corpus:
        # both DDL invocations must explicitly exclude the AGE-owned schemas
        # (apache/age issue #2503, see pg_schema_citation.sql's header).
        seen_argv = []

        def capture(argv, env, dst):
            seen_argv.append(argv)
            self._fake_stream(argv, env, dst)

        with tempfile.TemporaryDirectory() as tmp:
            gz_path = Path(tmp) / "01_dump.sql.gz"
            with mock.patch.object(public_dump, "require_classified"), \
                 mock.patch.object(public_dump, "corpus_tables",
                                    return_value=list(CORPUS_COLUMNS)), \
                 mock.patch.object(public_dump.schema_catalog, "schema_columns",
                                    return_value=dict(CORPUS_COLUMNS)), \
                 mock.patch.object(public_dump.schema_catalog, "schema_serial_columns",
                                    return_value=CORPUS_SERIALS), \
                 mock.patch.object(public_dump, "stream_stdout", side_effect=capture), \
                 mock.patch.object(citation_dump, "citation_tables",
                                    return_value=list(DUMPED_CITATION_TABLES)), \
                 mock.patch.object(citation_dump, "schema_columns",
                                    return_value=dict(self.CITATION_COLUMNS)), \
                 mock.patch.object(citation_dump, "schema_serial_columns",
                                    return_value={}), \
                 mock.patch.object(citation_dump, "stream_stdout", side_effect=capture):
                public_dump.dump_public({}, gz_path, citation_mode=CitationMode.FULL_SKELETON)
        pg_dump_argvs = [argv for argv in seen_argv if argv[0] == "pg_dump"]
        self.assertEqual(len(pg_dump_argvs), 2)  # corpus DDL, citation DDL
        for argv in pg_dump_argvs:
            self.assertIn("--exclude-schema=citation_graph", argv)
            self.assertIn("--exclude-schema=ag_catalog", argv)


class RefusalLeavesNoFileTests(unittest.TestCase):
    """dump_public()'s docstring promises it "refuses to write anything at
    all", and the citation half is written LAST -- after the corpus DDL and
    every corpus COPY block.

    Its two refusals, TableUnclassified and ColumnUnclassified, are neither
    CommandFailed, so raised from inside the gzip context they went straight
    past the handler that unlinks, leaving a truncated 01_dump.sql.gz on
    disk. The classification is resolved before the file is opened now, so
    the refusal costs no bytes and no cleanup.
    """

    CITATION_COLUMNS = dict(CitationDumpIntegrationTests.CITATION_COLUMNS)

    def _refuse(self, tmp, **patches):
        gz_path = Path(tmp) / "01_dump.sql.gz"
        written = []

        def record(argv, env, dst):
            written.append(argv)
            dst.write(b"row\n")

        with mock.patch.object(public_dump, "require_classified"), \
             mock.patch.object(public_dump, "corpus_tables",
                               return_value=list(CORPUS_COLUMNS)), \
             mock.patch.object(public_dump.schema_catalog, "schema_columns",
                               return_value=dict(CORPUS_COLUMNS)), \
             mock.patch.object(public_dump.schema_catalog, "schema_serial_columns",
                               return_value=CORPUS_SERIALS), \
             mock.patch.object(public_dump, "stream_stdout", side_effect=record), \
             mock.patch.object(citation_dump, "schema_serial_columns", return_value={}), \
             mock.patch.object(citation_dump, "stream_stdout", side_effect=record):
            with contextlib.ExitStack() as stack:
                for target, patch in patches.items():
                    stack.enter_context(mock.patch.object(citation_dump, target, **patch))
                with self.assertRaises(self.EXPECTED) as ctx:
                    public_dump.dump_public({}, gz_path,
                                            citation_mode=CitationMode.FULL_SKELETON)
        return gz_path, written, ctx.exception

    EXPECTED = RuntimeError  # both refusals subclass it

    def test_an_unclassified_citation_table_leaves_no_partial_dump(self):
        with tempfile.TemporaryDirectory() as tmp:
            gz_path, written, exc = self._refuse(
                tmp,
                schema_columns={"return_value": dict(self.CITATION_COLUMNS)},
                citation_tables={"side_effect": schema_catalog.TableUnclassified(
                    "citation.brand_new не классифицирована")},
            )
        self.assertIsInstance(exc, schema_catalog.TableUnclassified)
        self.assertFalse(gz_path.exists())
        self.assertEqual(written, [])

    def test_an_unclassified_citation_column_leaves_no_partial_dump(self):
        """The later of the two: the column class is asked per column while
        the COPY statement is built, which used to happen inside the loop
        that writes into the open file.
        """
        columns = dict(self.CITATION_COLUMNS, work=["id", "key", "brand_new"])
        with tempfile.TemporaryDirectory() as tmp:
            gz_path, written, exc = self._refuse(
                tmp,
                citation_tables={"return_value": list(DUMPED_CITATION_TABLES)},
                schema_columns={"return_value": columns},
            )
        self.assertIsInstance(exc, citation_columns.ColumnUnclassified)
        self.assertFalse(gz_path.exists())
        self.assertEqual(written, [])


if __name__ == "__main__":
    unittest.main()
