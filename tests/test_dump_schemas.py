"""The FULL profile's dump: pg_dump streamed straight through gzip, and the
numbers that come off those bytes.

Split out of test_artifact_bundle.py (kb/CLAUDE.md FILE_SIZE, and by
responsibility): that module is about the runtime files bundled beside the
dump and the tar that holds them; this one is about the one call that
writes the dump. pg_dump is stubbed and nothing else is -- the counts are
the stream's own, so a stubbed output carrying real COPY blocks is the
whole input they need, and there is no second database read left to stub.

The seam those counts are taken at is tests/test_copy_rows.py; here it is
what dump_schemas() does with it.
"""
from __future__ import annotations

import gzip
import io
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import _pathfix  # noqa: F401
import _pathfix_deploy  # noqa: F401

import artifact_bundle
import classification_gate
import copy_rows
import dump_scan
from _artifact_fixtures import citation_copy_block, citation_scan
from citation_columns import CENSUS_COLUMN, CENSUS_TABLE
from citation_vocab import WorkKind
from column_class_checks import CLASSIFIED_SCHEMAS
from column_classes import ColumnUnclassified
from schema_catalog import TableUnclassified
from copy_rows import DumpedRows
from manifest_contract import CitationMode, Profile, schemas_for


class FakeProc:
    def __init__(self, stdout: bytes, stderr: bytes = b"", returncode: int = 0):
        self.stdout = io.BytesIO(stdout)
        self._stderr = io.BytesIO(stderr)
        self._returncode = returncode

    @property
    def stderr(self):
        return self._stderr

    def wait(self):
        return self._returncode


class DumpSchemasTests(unittest.TestCase):
    """pg_dump is stubbed here; nothing else is. The full profile's row
    counts are READ BACK OFF the file it wrote, so a stubbed pg_dump output
    carrying real COPY blocks is the whole input the count needs -- there is
    no second database read left to stub.
    """

    def setUp(self):
        """The classification gate reads the live catalog before pg_dump is
        spawned, and pg_dump is the only child these tests have. It has its
        own class below; here it stands aside.
        """
        patch = mock.patch.object(artifact_bundle, "require_classified_schemas")
        patch.start()
        self.addCleanup(patch.stop)

    # A miniature pg_dump output: two COPY blocks, one per schema the
    # manifest is stamped from, plus a table of neither.
    DUMP = (b"CREATE SCHEMA corpus;\n"
            b"COPY corpus.documents (id) FROM stdin;\n2009_isu34\n1997_sm280\n\\.\n"
            b"COPY citation.work (id) FROM stdin;\n1\n2\n3\n\\.\n"
            b"COPY measurements.run (id) FROM stdin;\n7\n\\.\n")

    def test_the_dump_reports_what_the_manifest_will_declare(self):
        """Both writers answer with {table: rows} per schema: the packager
        stamps it into the manifest, and the recipient's gate holds the
        shipped bytes to it. Here the answer IS the shipped bytes.
        """
        with tempfile.TemporaryDirectory() as tmp:
            gz_path = Path(tmp) / "dump.sql.gz"
            with mock.patch.object(artifact_bundle.subprocess, "Popen",
                                    return_value=FakeProc(self.DUMP)):
                carried = artifact_bundle.dump_schemas(
                    {}, gz_path, CitationMode.FULL_SKELETON)
        self.assertEqual(carried.citation, {"work": 3})
        self.assertEqual(carried.corpus, {"documents": 2})

    def test_a_dump_carrying_no_citation_block_reports_no_citation_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            gz_path = Path(tmp) / "dump.sql.gz"
            with mock.patch.object(artifact_bundle.subprocess, "Popen",
                                    return_value=FakeProc(b"-- fake\n")):
                carried = artifact_bundle.dump_schemas({}, gz_path, CitationMode.NONE)
        self.assertEqual(carried.citation, {})
        self.assertEqual(carried.corpus, {})

    def test_the_stream_is_counted_once_and_the_file_never_reopened(self):
        """The counts are the bytes' own, and reading them costs no second
        inflate: gzip.open is called exactly once for the whole build -- the
        write. Re-read afterwards, a full-profile dump that carries every
        source PDF as hex is decompressed and line-split a second time for
        numbers the stream had already gone past.
        """
        opened = []
        real_open = gzip.open

        def counting_open(*args, **kwargs):
            opened.append(args[0])
            return real_open(*args, **kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            gz_path = Path(tmp) / "dump.sql.gz"
            with mock.patch.object(artifact_bundle.subprocess, "Popen",
                                    return_value=FakeProc(self.DUMP)), \
                 mock.patch.object(gzip, "open", side_effect=counting_open):
                artifact_bundle.dump_schemas({}, gz_path, CitationMode.FULL_SKELETON)
        self.assertEqual(opened, [gz_path])

    def test_the_streamed_count_is_the_one_the_finished_file_reads_back(self):
        """Two readings of one dump: the state machine in the stream, and
        the artifact-side reader over the bytes it wrote. They are the same
        recognition (dump_scan's COPY_HEADER and terminator), so the
        manifest and the recipient's gate cannot drift apart.
        """
        with tempfile.TemporaryDirectory() as tmp:
            gz_path = Path(tmp) / "dump.sql.gz"
            with mock.patch.object(artifact_bundle.subprocess, "Popen",
                                    return_value=FakeProc(self.DUMP)):
                streamed = artifact_bundle.dump_schemas(
                    {}, gz_path, CitationMode.FULL_SKELETON)
            read_back = DumpedRows.from_contents(dump_scan.scan(gz_path))
        self.assertEqual(streamed, read_back)

    # The counters themselves -- block recognition, the kept prefix, the
    # census -- are tests/test_copy_rows.py: this module is about what
    # dump_schemas() assembles, not about the seam it streams through.

    def test_no_count_is_asked_of_the_live_database_at_all(self):
        """The read that used to follow pg_dump is gone, not merely
        unused: a count from a fresh connection describes whatever the
        database held at that moment, and the gate demands it equal the
        file.
        """
        self.assertFalse(hasattr(artifact_bundle, "live_row_counts"))

    def test_the_dump_asks_for_exactly_what_the_manifest_declares(self):
        """One source of truth for "which schemas does this profile ship":
        manifest.json's schemas[] and pg_dump's --schema arguments are the
        same list from schemas_for(), not two lists kept equal by hand.
        """
        for mode in CitationMode.ALL:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                gz_path = Path(tmp) / "dump.sql.gz"
                with mock.patch.object(artifact_bundle.subprocess, "Popen",
                                        return_value=FakeProc(b"-- fake\n")) as popen_mock:
                    artifact_bundle.dump_schemas({}, gz_path, citation_mode=mode)
                (argv,), _kwargs = popen_mock.call_args
                asked = [a.split("=", 1)[1] for a in argv if a.startswith("--schema=")]
                self.assertEqual(asked, schemas_for(Profile.FULL, mode))

    def test_the_owners_own_backup_carries_the_citation_schema(self):
        # The full profile applies no legal or policy cut: whenever the
        # schema exists at all (i.e. the mode is not NONE) it ships whole.
        self.assertIn("citation", schemas_for(Profile.FULL, CitationMode.FULL_SKELETON))
        self.assertNotIn("citation", schemas_for(Profile.FULL, CitationMode.NONE))

    def test_dumps_never_carry_age_catalog(self):
        with tempfile.TemporaryDirectory() as tmp:
            gz_path = Path(tmp) / "dump.sql.gz"
            with mock.patch.object(artifact_bundle.subprocess, "Popen",
                                    return_value=FakeProc(b"-- fake\n")) as popen_mock:
                artifact_bundle.dump_schemas({}, gz_path, CitationMode.FULL_SKELETON)
            (argv,), _kwargs = popen_mock.call_args
        self.assertIn("--exclude-schema=citation_graph", argv)
        self.assertIn("--exclude-schema=ag_catalog", argv)
        self.assertIn("--schema=citation", argv)

    def test_streams_pg_dump_stdout_through_gzip(self):
        payload = b"-- fake pg_dump output\n" * 1000
        with tempfile.TemporaryDirectory() as tmp:
            gz_path = Path(tmp) / "dump.sql.gz"
            with mock.patch.object(artifact_bundle.subprocess, "Popen",
                                    return_value=FakeProc(payload)):
                artifact_bundle.dump_schemas({}, gz_path, CitationMode.FULL_SKELETON)
            self.assertTrue(gz_path.is_file())
            with gzip.open(gz_path, "rb") as f:
                self.assertEqual(f.read(), payload)

    def test_pg_dump_failure_raises_and_removes_partial_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            gz_path = Path(tmp) / "dump.sql.gz"
            with mock.patch.object(
                artifact_bundle.subprocess, "Popen",
                return_value=FakeProc(b"partial", stderr=b"connection refused", returncode=1),
            ):
                with self.assertRaises(RuntimeError) as ctx:
                    artifact_bundle.dump_schemas({}, gz_path, CitationMode.FULL_SKELETON)
            self.assertIn("connection refused", str(ctx.exception))
            self.assertFalse(gz_path.exists())

    def test_large_stderr_output_does_not_deadlock(self):
        # 16bd7012: with both stdout and stderr as pipes and nothing
        # draining stderr while the main thread blocks copying stdout, a
        # child that writes enough to stderr (NOTICE/WARNING lines are
        # routine on a large multi-schema pg_dump) to fill the OS pipe
        # buffer (~64KB on Linux) would deadlock -- the child blocks
        # writing to stderr, the parent blocks reading stdout, forever. A
        # real subprocess is required to reproduce that: FakeProc's
        # in-memory BytesIO has no OS pipe buffer to fill. Popen is
        # redirected to a real python3 -c script instead of the real
        # "pg_dump" argv (not installed, and irrelevant to the pipe
        # mechanics under test); the whole call is run on a background
        # thread with a bounded join() so an actual deadlock fails this
        # test instead of hanging the suite forever.
        script = (
            "import sys\n"
            "sys.stderr.write('E' * 200000)\n"
            "sys.stderr.flush()\n"
            "sys.stdout.buffer.write(b'ok')\n"
        )
        real_argv = [sys.executable, "-c", script]
        real_popen = subprocess.Popen  # captured before patching -- see below

        def fake_popen(_argv, **kwargs):
            # NOT subprocess.Popen: mock.patch.object below patches the
            # Popen attribute on the actual subprocess module (the same
            # object artifact_bundle.subprocess IS), so calling
            # subprocess.Popen from inside this function would recurse into
            # itself instead of spawning the real interpreter.
            return real_popen(real_argv, **kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            gz_path = Path(tmp) / "dump.sql.gz"
            done = threading.Event()
            with mock.patch.object(artifact_bundle.subprocess, "Popen", side_effect=fake_popen):
                worker = threading.Thread(
                    target=lambda: (artifact_bundle.dump_schemas(
                        {}, gz_path, CitationMode.FULL_SKELETON), done.set()),
                )
                worker.start()
                worker.join(timeout=10)
            self.assertTrue(done.is_set(), "dump_schemas deadlocked on large stderr output")
            with gzip.open(gz_path, "rb") as f:
                self.assertEqual(f.read(), b"ok")


class KindCensusTests(unittest.TestCase):
    """manifest.citation.work_by_kind for the FULL profile, taken off the
    bytes pg_dump wrote.

    The public profile counts per block, one psql child at a time
    (citation_dump.dump_citation); the full profile has no such seam, so the
    census is read out of the single stream by name -- and that is a second
    implementation of one manifest number, on the profile whose rows are the
    long ones. A number the artifact side re-derives
    (citation_cut_checks.check_kind_census_matches_manifest) and the build
    itself never asserts is a promise rather than a fact -- and the promise
    is kept only until the column list shifts.
    """

    def setUp(self):
        """The classification gate reads the live catalog before pg_dump is
        spawned, and pg_dump is the only child these tests have. It has its
        own class below; here it stands aside.
        """
        patch = mock.patch.object(artifact_bundle, "require_classified_schemas")
        patch.start()
        self.addCleanup(patch.stop)

    def _dumped(self, payload: str, mode: str = CitationMode.FULL_SKELETON):
        with tempfile.TemporaryDirectory() as tmp:
            gz_path = Path(tmp) / "dump.sql.gz"
            with mock.patch.object(artifact_bundle.subprocess, "Popen",
                                    return_value=FakeProc(payload.encode())):
                return artifact_bundle.dump_schemas({}, gz_path, mode)

    def _work_block(self, columns: list[str], rows: list[list[str]]) -> str:
        return citation_copy_block(f"citation.{CENSUS_TABLE}", columns, rows)

    def test_every_kind_reaches_the_manifest_and_the_total_still_holds(self):
        carried = self._dumped(self._work_block(
            ["id", "key", CENSUS_COLUMN],
            [["1", "W1", WorkKind.OUR_DOCUMENT],
             ["2", "W2", WorkKind.EXTERNAL_SKELETON],
             ["3", "W3", WorkKind.EXTERNAL_SKELETON],
             ["4", "W4", WorkKind.INDEXED]]))
        self.assertEqual(carried.work_by_kind,
                         {WorkKind.OUR_DOCUMENT: 1, WorkKind.EXTERNAL_SKELETON: 2,
                          WorkKind.INDEXED: 1})
        # The census counts the same rows table_rows does, so the two
        # manifest numbers cannot describe different reads of one block.
        self.assertEqual(sum(carried.work_by_kind.values()),
                         carried.citation[CENSUS_TABLE])

    def test_the_column_is_taken_by_position_from_the_block_header(self):
        """A column added to citation.work shifts every index after it, so
        a hand-kept offset would count the neighbouring field instead --
        plausibly, since most of them are short strings too.
        """
        carried = self._dumped(self._work_block(
            [CENSUS_COLUMN, "id", "key"],
            [[WorkKind.EXTERNAL_SKELETON, "1", "W1"]]))
        self.assertEqual(carried.work_by_kind, {WorkKind.EXTERNAL_SKELETON: 1})

    def test_a_row_longer_than_the_kept_prefix_is_still_counted_by_kind(self):
        """The census block is the one whose rows are unbounded: `abstract`
        and `evidence` are third-party text and the vector renders as 1024
        floats. Truncated at LINE_PREFIX the row loses the tab the census
        field is read from, and the tally comes out silently short by
        however many rows were long -- stamped into the manifest as fact.
        """
        long_abstract = "x" * (copy_rows.LINE_PREFIX * 3)
        carried = self._dumped(self._work_block(
            ["id", "abstract", CENSUS_COLUMN],
            [["1", long_abstract, WorkKind.EXTERNAL_SKELETON],
             ["2", "short", WorkKind.OUR_DOCUMENT]]))
        self.assertEqual(carried.citation, {CENSUS_TABLE: 2})
        self.assertEqual(carried.work_by_kind,
                         {WorkKind.EXTERNAL_SKELETON: 1, WorkKind.OUR_DOCUMENT: 1})

    def test_the_streamed_census_is_what_the_artifact_side_reader_finds(self):
        """The two answers the recipient's gate compares: the producer's
        tally off the write, and the bundled checker's tally off the file.
        Different code, same bytes -- and the gate demands they be equal
        element by element.
        """
        dump = self._work_block(
            ["id", "key", CENSUS_COLUMN],
            [["1", "W1", WorkKind.OUR_DOCUMENT],
             ["2", "W2", WorkKind.EXTERNAL_SKELETON],
             ["3", "W3", WorkKind.OUR_DOCUMENT]])
        carried = self._dumped(dump)
        _scans, facts = citation_scan(dump, CitationMode.FULL_SKELETON)
        self.assertEqual(carried.work_by_kind, facts.citation.work_by_kind)

    def test_a_work_block_without_the_census_column_tallies_nothing(self):
        """An empty census is one no manifest number can equal, which is
        the direction this has to fail in: a census that quietly agreed by
        being empty is the [OK] about what nobody looked at.
        """
        carried = self._dumped(self._work_block(["id", "key"], [["1", "W1"]]))
        self.assertEqual(carried.citation, {CENSUS_TABLE: 1})
        self.assertEqual(carried.work_by_kind, {})

    def test_a_dump_carrying_no_work_block_has_an_empty_census(self):
        carried = self._dumped(
            "COPY corpus.documents (id) FROM stdin;\n1997_sm280\n\\.\n")
        self.assertEqual(carried.citation, {})
        self.assertEqual(carried.work_by_kind, {})

    def test_no_other_block_contributes_to_the_census(self):
        """The block is recognised by its qualified name, not by carrying a
        column that happens to be spelled the same.
        """
        carried = self._dumped(
            citation_copy_block("citation.cites", ["citing", CENSUS_COLUMN],
                                [["1", WorkKind.EXTERNAL_SKELETON]])
            + self._work_block(["id", CENSUS_COLUMN],
                               [["1", WorkKind.OUR_DOCUMENT]]))
        self.assertEqual(carried.work_by_kind, {WorkKind.OUR_DOCUMENT: 1})


class ClassificationGateTests(unittest.TestCase):
    """The full profile refuses an unclassified table or column too.

    It applies no cut and ships whatever the schemas hold, so the verdict
    changes nothing about the contents -- but the checker bundled INTO the
    artifact holds every COPY block of schemas corpus and citation to the
    same maps whatever profile wrote them
    (column_class_checks.check_columns_are_classified takes no profile).
    Without a gate here, a table or column added to either schema built
    cleanly, reported success, and failed the finished package's own
    certification: the failure MANIFEST_DESCRIBES_ARTIFACT names, on the
    profile that skipped the producer-side half.

    The catalog is a mock: what is under test is which questions are asked
    of it and what the answers are held to.
    """

    CATALOG = {
        "corpus": {"documents": ["id", "source_blob"],
                   "pages": ["document_id", "body"],
                   "embedding_model": ["id", "model"]},
        "citation": {CENSUS_TABLE: ["id", CENSUS_COLUMN], "cites": ["citing", "cited"]},
    }

    def _gate(self, catalog: dict) -> None:
        with mock.patch.object(classification_gate, "present_tables",
                                side_effect=lambda _env, schema: list(catalog[schema])), \
             mock.patch.object(classification_gate, "schema_columns",
                                side_effect=lambda _env, schema: dict(catalog[schema])):
            classification_gate.require_classified_schemas(
                {}, ["corpus", "citation", "measurements"])

    def test_a_catalog_the_maps_know_passes(self):
        self._gate(self.CATALOG)

    def test_a_table_outside_the_map_is_refused(self):
        catalog = {schema: dict(tables) for schema, tables in self.CATALOG.items()}
        catalog["citation"]["annotations"] = ["id"]
        with self.assertRaises(TableUnclassified) as caught:
            self._gate(catalog)
        self.assertIn("citation.annotations", str(caught.exception))

    def test_a_column_outside_the_map_is_refused(self):
        catalog = {schema: dict(tables) for schema, tables in self.CATALOG.items()}
        catalog["corpus"]["documents"] = ["id", "source_blob", "annotation"]
        with self.assertRaises(ColumnUnclassified) as caught:
            self._gate(catalog)
        self.assertIn("corpus.documents.annotation", str(caught.exception))

    def test_an_unclassified_schema_is_not_this_gates_business(self):
        """measurements travels in the full artifact and no map names it --
        the same schemas the bundled checker passes over, and for the same
        reason. Asked about it at all, the gate would refuse every correct
        build of this profile.
        """
        self.assertNotIn("measurements", CLASSIFIED_SCHEMAS)
        with mock.patch.object(classification_gate, "present_tables") as tables_mock, \
             mock.patch.object(classification_gate, "schema_columns", return_value={}):
            classification_gate.require_classified_schemas({}, ["measurements"])
        tables_mock.assert_not_called()

    def test_the_gate_and_the_bundled_checker_read_one_declaration(self):
        self.assertIs(classification_gate.CLASSIFIED_SCHEMAS, CLASSIFIED_SCHEMAS)

    def test_the_dump_refuses_before_pg_dump_is_spawned(self):
        """The refusal costs nothing and leaves nothing: no child, no file.
        A full dump embeds every source PDF as hex, so finding out after the
        stream is minutes of work and a partial artifact on disk.
        """
        with tempfile.TemporaryDirectory() as tmp:
            gz_path = Path(tmp) / "dump.sql.gz"
            with mock.patch.object(artifact_bundle, "require_classified_schemas",
                                    side_effect=ColumnUnclassified("corpus.pages.note")), \
                 mock.patch.object(artifact_bundle.subprocess, "Popen") as popen_mock:
                with self.assertRaises(ColumnUnclassified):
                    artifact_bundle.dump_schemas({}, gz_path, CitationMode.FULL_SKELETON)
            popen_mock.assert_not_called()
            self.assertFalse(gz_path.exists())

    def test_the_gate_is_asked_about_exactly_the_schemas_dumped(self):
        """One list: what pg_dump is asked for and what the classification
        is held to are the same schemas_for() answer, so a mode that adds a
        schema cannot add it past the gate.
        """
        for mode in CitationMode.ALL:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                gz_path = Path(tmp) / "dump.sql.gz"
                with mock.patch.object(artifact_bundle, "require_classified_schemas") as gate, \
                     mock.patch.object(artifact_bundle.subprocess, "Popen",
                                        return_value=FakeProc(b"-- fake\n")) as popen_mock:
                    artifact_bundle.dump_schemas({}, gz_path, citation_mode=mode)
                (argv,), _kwargs = popen_mock.call_args
                asked = [a.split("=", 1)[1] for a in argv if a.startswith("--schema=")]
                (_env, gated), _kwargs = gate.call_args
                self.assertEqual(list(gated), asked)


if __name__ == "__main__":
    unittest.main()
