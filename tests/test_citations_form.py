"""The shape of the crawl at depth >= 2, and the corpus-twin rule.

Both are answers to things that went wrong on live data, so both are pinned
offline rather than left to the next crawl to rediscover:

- the first depth-2 attempt expanded every depth-1 node upward, asked "who
  cites Higher Transcendental Functions", and burned a 1000-request window
  without writing a node;
- `pg_graph.py candidates` recommended the English translation of one of our
  own papers as something to go read, because original and translation carry
  different DOIs and the id union cannot join them.
"""
from __future__ import annotations

import inspect
import unittest
from unittest import mock

import _pathfix  # noqa: F401
from _citation_fixtures import FakeClient, PlannedEmbedder, unit, work
from citations import journal, twin_pass, twins
from citations.crawl import HUB_CAP, Snowball
from citations import store
from citations.dry_store import DryRunWriter
from pg_copy import CopyResult


def snowball_over(records, citers, *, tau=0.5, hub_cap=HUB_CAP, seed_key="W_SEED"):
    writer = DryRunWriter()
    client = FakeClient(records, citers)
    embedder = PlannedEmbedder({"Seed": unit(0), "Near": unit(0), "Ref": unit(0),
                                "Far": unit(500)})
    snow = Snowball(client, embedder, writer, tau=tau, crawl_id="c",
                    log=lambda *_: None, hub_cap=hub_cap)
    snow.seed(["doc_a"], {"doc_a": seed_key})
    return writer, client, snow


class ExpansionFormTests(unittest.TestCase):
    def _two_kinds(self):
        seed = work("W_SEED", title="Seed Chebyshev", refs=["W_REF"])
        citer = work("W_CITER", title="Near Chebyshev citer", refs=["W_SEED"])
        reference = work("W_REF", title="Ref classic handbook")
        return snowball_over([seed, citer, reference], {"W_SEED": [citer]})

    def test_depth_one_expands_the_seeds_whatever_their_relation(self):
        _writer, _client, snow = self._two_kinds()
        self.assertEqual(snow.expandable(["W_SEED"], 1), ["W_SEED"])

    def test_referenced_node_is_a_leaf_at_depth_two(self):
        writer, client, snow = self._two_kinds()
        kept = snow.expand(["W_SEED"], 1)
        self.assertEqual(sorted(kept), ["W_CITER", "W_REF"])
        # Both are written -- being a leaf is about expansion, not storage.
        # (Seeds are written by run(), not by a bare expand().)
        self.assertEqual(sorted(n.key for n in writer.works_seen),
                         ["W_CITER", "W_REF"])
        self.assertEqual(snow.expandable(kept, 2), ["W_CITER"])

    def test_cites_node_expands_in_both_directions(self):
        seed = work("W_SEED", title="Seed Chebyshev")
        citer = work("W_CITER", title="Near Chebyshev citer",
                     refs=["W_SEED", "W_DOWN"])
        down = work("W_DOWN", title="Near Chebyshev reference")
        up = work("W_UP", title="Near Chebyshev second order",
                  refs=["W_CITER"])
        _writer, client, snow = snowball_over(
            [seed, citer, down, up],
            {"W_SEED": [citer], "W_CITER": [up]})
        snow.expand(["W_SEED"], 1)
        snow.expand(snow.expandable(["W_CITER"], 2), 2)
        self.assertIn("W_UP", snow.registry.nodes, "не пошли вверх от cites-узла")
        self.assertIn("W_DOWN", snow.registry.nodes, "не пошли вниз от cites-узла")

    def test_run_stops_expanding_referenced_nodes(self):
        writer, client, snow = self._two_kinds()
        snow.run(2)
        # W_REF must never appear in a cites: batch.
        self.assertNotIn("W_REF", [i for b in client.cites_batches for i in b])
        self.assertIn("W_CITER", [i for b in client.cites_batches for i in b])


class HubCapTests(unittest.TestCase):
    def _with_hub(self, cap):
        seed = work("W_SEED", title="Seed Chebyshev", refs=["W_HUB"])
        seed["cited_by_count"] = 3
        hub = work("W_HUB", title="Ref classic handbook")
        hub["cited_by_count"] = 5000
        citer = work("W_C", title="Near Chebyshev", refs=["W_HUB"])
        return snowball_over([seed, hub, citer],
                             {"W_SEED": [], "W_HUB": [citer]}, hub_cap=cap)

    def test_hub_is_not_asked_upward_and_says_so_in_the_journal(self):
        writer, client, snow = self._with_hub(1000)
        snow.expand(["W_SEED"], 1)          # brings W_HUB in as a reference
        writer.steps_seen.clear()
        client.cites_batches.clear()
        snow.expand(["W_HUB"], 2)
        self.assertEqual(client.cites_batches, [],
                         "хаб всё-таки спросили вверх")
        skips = [s for s in writer.steps_seen if s["action"] == "hub-skip"]
        self.assertEqual([s["frontier_key"] for s in skips], ["W_HUB"])
        self.assertEqual(skips[0]["cited_by_count"], 5000)

    def test_a_hub_still_expands_downward(self):
        writer, client, snow = self._with_hub(1000)
        snow.expand(["W_SEED"], 1)
        before = client.id_batches.copy()
        snow.expand(["W_HUB"], 2)
        self.assertGreaterEqual(len(client.id_batches), len(before))

    def test_below_the_cap_the_node_is_asked_upward(self):
        writer, client, snow = self._with_hub(10000)
        snow.expand(["W_SEED"], 1)
        client.cites_batches.clear()
        snow.expand(["W_HUB"], 2)
        self.assertEqual(client.cites_batches, [["W_HUB"]])

    def test_at_the_cap_exactly_the_node_is_still_asked_upward(self):
        """The cap is the last value that expands, not the first that does
        not: `>` and `>=` differ by exactly this node.
        """
        writer, client, snow = self._with_hub(5000)
        snow.expand(["W_SEED"], 1)
        client.cites_batches.clear()
        writer.steps_seen.clear()
        snow.expand(["W_HUB"], 2)
        self.assertEqual(client.cites_batches, [["W_HUB"]])
        self.assertEqual([s for s in writer.steps_seen if s["action"] == "hub-skip"], [])

    def test_cited_by_count_sums_over_twins(self):
        _writer, _client, snow = snowball_over(
            [work("W_SEED", title="Seed Chebyshev")], {})
        node = snow.registry.nodes["W_SEED"]
        node.absorb(dict(work("W_TWIN", doi="10.1/x"), cited_by_count=7))
        node.absorb(dict(work("W_TWIN2", doi="10.1/y"), cited_by_count=5))
        self.assertEqual(node.cited_by_count, 12)


class JournalFormatTests(unittest.TestCase):
    """Which column carries what is a contract with SQL written elsewhere:
    the score distribution query, the hub report's depth-1 node set and the
    public artifact's journal cut all read columns, never prose.
    """

    def test_keep_and_drop_carry_score_and_tau_as_values(self):
        kept = journal.keep("c", 2, "W1", "W_NODE", 0.6123, 0.5, "cites")
        dropped = journal.drop("c", 2, "W2", 0.4001, 0.5, "referenced")
        self.assertEqual((kept["score"], kept["tau"]), (0.6123, 0.5))
        self.assertEqual((dropped["score"], dropped["tau"]), (0.4001, 0.5))
        self.assertEqual(kept["node_key"], "W_NODE")
        self.assertNotIn("node_key", dropped, "у отброшенного кандидата узла нет")

    def test_keep_and_drop_carry_the_relation_as_a_value(self):
        """How a node entered the graph decides whether it expands at all
        (SNOWBALL_FRONTIER: only relation='cites' opens depth >= 2), and the
        hub measurement groups by it. A fact the pipeline acts on is a
        column -- the alternative was re-deriving it from citation.work.
        evidence with an 'unknown' fallback, one altitude down.
        """
        kept = journal.keep("c", 2, "W1", "W_NODE", 0.6123, 0.5, "cites")
        dropped = journal.drop("c", 2, "W2", 0.4001, 0.5, "referenced")
        self.assertEqual(kept["relation"], "cites")
        self.assertEqual(dropped["relation"], "referenced")

    def test_no_step_hides_a_number_or_a_name_in_its_prose(self):
        steps = [journal.keep("c", 2, "W1", "W_NODE", 0.6123, 0.5, "cites"),
                 journal.drop("c", 2, "W2", 0.4001, 0.5, "referenced"),
                 journal.hub_skip("c", 2, "W1", 5000, 1000),
                 journal.twin("c", "W_EN", "2019_rm9846", "W_RU")]
        for step in steps:
            for marker in ("score=", "tau=", "node=", "twin-of=", "seed=",
                           "relation=", "cited_by_count="):
                self.assertNotIn(marker, step["reason"], step)

    def test_hub_skip_names_the_node_it_declined_to_open(self):
        row = journal.hub_skip("c", 2, "W1", 5000, 1000)
        self.assertEqual(row["action"], "hub-skip")
        self.assertEqual(row["node_key"], "W1")
        self.assertEqual(row["cited_by_count"], 5000)

    def test_a_twin_row_names_the_document_the_record_and_the_seed(self):
        row = journal.twin("c", "W_EN", "2019_rm9846", "W_RU")
        self.assertEqual(row["action"], "keep")
        self.assertEqual(row["depth"], 0)
        self.assertEqual(row["frontier_key"], "2019_rm9846")
        self.assertEqual(row["candidate_key"], "W_EN")
        self.assertEqual(row["node_key"], "W_RU")


class CorpusTwinTests(unittest.TestCase):
    """The live case: 2019_rm9846 and its English translation."""

    RU = "Соболевская ортогональность и её приложения"
    EN = "Sobolev-orthogonal systems of functions and some of their applications"

    SEED = {"key": "W2966149037", "document_id": "2019_rm9846",
            "mathnet_id": "rm9846"}

    def index(self):
        return twins.build_index([dict(self.SEED, titles=[self.RU, self.EN])])

    def mathnet(self):
        return twins.build_mathnet_index([dict(self.SEED, titles=[])])

    def find(self, title, year, doi="", *, titles=True):
        return twins.find_twin(
            title, year, doi,
            self.index() if titles else {}, self.mathnet(), {"2019_rm9846": [2019]})

    def test_the_live_case_matches_on_the_mathnet_doi_alone(self):
        """/rus/rm9846 prints the translation's journal but NOT its English
        title, so the title rule cannot see this pair -- the DOI suffix can."""
        hit = self.find(self.EN, 2019, "10.1070/rm9846", titles=False)
        self.assertEqual(hit, ("2019_rm9846", "W2966149037", "mathnet-doi"))

    def test_the_originals_own_doi_matches_too(self):
        hit = self.find("что угодно", None, "https://doi.org/10.4213/RM9846", titles=False)
        self.assertEqual(hit[:2], ("2019_rm9846", "W2966149037"))

    def test_a_foreign_doi_is_not_our_work(self):
        self.assertIsNone(self.find("Нечто", 2019, "10.1016/j.jat.2019.01.001",
                                    titles=False))

    def test_translation_of_our_work_is_recognised(self):
        self.assertEqual(self.find(self.EN, 2019),
                         ("2019_rm9846", "W2966149037", "title+year"))

    def test_translation_a_year_later_still_matches(self):
        self.assertEqual(self.find(self.EN, 2020),
                         ("2019_rm9846", "W2966149037", "title+year"))

    def test_a_different_year_is_not_our_work(self):
        self.assertIsNone(self.find(self.EN, 2005))

    def test_the_year_just_outside_the_span_is_not_our_work(self):
        """The excluded side of the +-span boundary, asserted on the
        predicate itself: it decides whether a candidate is promoted to
        our-document, which rewrites what the corpus claims about itself,
        and only the included side (diff=1) and a far-outside one were
        covered.
        """
        self.assertTrue(twins.year_matches(2020, [2019]))
        self.assertFalse(twins.year_matches(2021, [2019]))
        self.assertTrue(twins.year_matches(2021, [2019], span=2))
        self.assertFalse(twins.year_matches(2022, [2019], span=2))

    def test_punctuation_and_case_do_not_matter(self):
        noisy = "SOBOLEV–ORTHOGONAL Systems of Functions, and Some of Their Applications!"
        self.assertEqual(self.find(noisy, 2019),
                         ("2019_rm9846", "W2966149037", "title+year"))

    def test_a_merely_similar_title_is_refused(self):
        # No fuzzy tier here on purpose: promoting to our-document rewrites
        # what the corpus claims about itself.
        near = "Sobolev orthogonal systems of functions and their applications"
        self.assertIsNone(self.find(near, 2019))

    def test_a_title_two_documents_claim_is_dropped_from_the_index(self):
        index = twins.build_index([
            {"key": "W1", "document_id": "doc_a", "titles": ["Same Name"]},
            {"key": "W2", "document_id": "doc_b", "titles": ["Same Name"]},
        ])
        self.assertEqual(index, {})

    def test_a_monograph_without_a_year_matches_on_title_alone(self):
        index = twins.build_index([
            {"key": "W9", "document_id": "mono", "titles": ["Многочлены на сетках"]},
        ])
        self.assertEqual(
            twins.find_twin("Многочлены на сетках", None, "", index, {}, {"mono": []}),
            ("mono", "W9", "title+year"))

    def test_doi_suffix_is_registrant_agnostic(self):
        self.assertEqual(twins.doi_suffix("https://doi.org/10.1070/RM9846."), "rm9846")
        self.assertEqual(twins.doi_suffix("10.4213/rm9846"), "rm9846")
        self.assertEqual(twins.doi_suffix(None), "")

    def test_normalization_folds_yo_and_tex(self):
        self.assertEqual(twins.normalize_title(r"Чебышёв $\\alpha$-ряды"),
                         twins.normalize_title("чебышев ряды"))


class MathnetParseTests(unittest.TestCase):
    def test_both_citations_come_off_the_page_head(self):
        from citations.mathnet import mathnet_id, parse_titles
        head = ("<title>И. И. Шарапудинов, “Асимптотические свойства”, "
                "Матем. сб., 180:9 (1989), 1;  I. I. Sharapudinov, "
                "“Asymptotic properties”, Math. USSR-Sb., 68:1 (1991), 2"
                "</title>")
        titles, years = parse_titles(head)
        self.assertEqual(titles, ["Асимптотические свойства", "Asymptotic properties"])
        self.assertEqual(years, [1989, 1991])
        self.assertEqual(mathnet_id("https://www.mathnet.ru/rus/sm1659"), "sm1659")
        self.assertIsNone(mathnet_id(""))
        self.assertIsNone(mathnet_id("https://doi.org/10.1/x"))


class TwinPromotionBatchTests(unittest.TestCase):
    """The promotion goes through the same Writer seam the crawl uses, and
    is staged and applied in ONE statement, not one psql process per node:
    the N+1 write pattern the rest of the package already avoids, and a
    failure mid-loop would otherwise leave a half-promoted set with no
    journal rows written at all.
    """

    MERGED = [
        {"key": "W_EN", "document_id": "2019_rm9846", "seed_key": "W_RU", "rule": "title"},
        {"key": "W_FR", "document_id": "2015_demr1", "seed_key": "W_RU2", "rule": "doi"},
    ]

    SEEDS = [{"key": "W_RU", "document_id": "2019_rm9846",
              "titles": ["Некоторая работа"], "mathnet_id": ""}]

    def _found(self, writer):
        with mock.patch.object(twin_pass, "seed_titles", return_value=self.SEEDS), \
             mock.patch.object(twin_pass, "corpus_years", return_value={"2019_rm9846": [2019]}), \
             mock.patch.object(twin_pass, "skeleton_nodes", return_value=[
                 ("W_EN", "Некоторая работа", 2019, "")]):
            return twin_pass.merge_twins({}, "crawl-1", writer)

    def test_one_staged_update_for_the_whole_batch(self):
        calls = []

        def capture(env, target, rows, **kwargs):
            calls.append((target, list(rows), kwargs))
            return CopyResult(2, "2\n")

        with mock.patch.object(store, "copy_csv_rows", side_effect=capture):
            promoted = store.PostgresWriter({}).promote(self.MERGED)
        self.assertEqual(promoted, 2)
        self.assertEqual(len(calls), 1, "staging DDL, COPY и UPDATE — один скрипт")
        target, rows, kwargs = calls[0]
        self.assertTrue(target.startswith("stage_twin ("), target)
        self.assertEqual([row[0] for row in rows], ["W_EN", "W_FR"])
        self.assertIn("CREATE TEMP TABLE stage_twin", kwargs["preamble"])
        update = kwargs["epilogue"]
        self.assertIn("UPDATE citation.work w", update)
        self.assertIn("FROM stage_twin s", update)
        self.assertIn("WHERE w.key = s.key", update)
        self.assertIn("'twin_of', s.seed_key", update)

    def test_the_promoted_count_is_the_one_the_update_reported(self):
        """A key the pass merged but citation.work has no row for updates
        nothing, and the twin count must not claim it.
        """
        with mock.patch.object(store, "copy_csv_rows",
                                return_value=CopyResult(2, "1\n")):
            writer = store.PostgresWriter({})
            self.assertEqual(writer.promote(self.MERGED), 1)
        self.assertEqual(writer.counts["twin"], 1)

    def test_empty_batch_touches_the_database_at_all_never(self):
        with mock.patch.object(store, "copy_csv_rows") as copy_mock:
            self.assertEqual(store.PostgresWriter({}).promote([]), 0)
        copy_mock.assert_not_called()

    def test_merge_twins_promotes_and_journals_every_match(self):
        writer = DryRunWriter()
        merged = self._found(writer)
        self.assertEqual(len(merged), 1)
        self.assertEqual([m["key"] for m in writer.promoted_seen], ["W_EN"])
        self.assertEqual([s["action"] for s in writer.steps_seen], ["keep"])
        self.assertEqual(writer.steps_seen[0]["node_key"], "W_RU")
        self.assertEqual(writer.counts["twin"], 1)

    def test_a_dry_run_of_the_twin_pass_issues_no_statement_at_all(self):
        """Not "the flag was checked at both call sites" -- the mode has no
        object that can write, which is the DRY_RUN_WRITES_NOTHING form the
        other three modes already have.
        """
        with mock.patch.object(store, "copy_csv_rows") as copy_mock:
            merged = self._found(DryRunWriter())
        self.assertEqual(len(merged), 1)
        copy_mock.assert_not_called()

    def test_merge_twins_has_no_dry_run_flag_left_to_get_wrong(self):
        self.assertNotIn("dry_run", inspect.signature(twin_pass.merge_twins).parameters)

    def _matched_nothing(self, writer):
        """The same pass over a skeleton whose only node resembles no seed:
        a different title, a different year and no DOI, so neither rule can
        fire.
        """
        with mock.patch.object(twin_pass, "seed_titles", return_value=self.SEEDS), \
             mock.patch.object(twin_pass, "corpus_years", return_value={"2019_rm9846": [2019]}), \
             mock.patch.object(twin_pass, "skeleton_nodes", return_value=[
                 ("W_XX", "Совершенно другая работа", 1975, "")]):
            return twin_pass.merge_twins({}, "crawl-1", writer)

    def test_a_round_that_matches_nothing_asks_the_writer_for_nothing(self):
        """The empty batch never reaches the writer at all.

        promote([]) and journal([]) are no-ops in the live writer TODAY, so
        moving these calls out of the `if merged:` guard would break no
        assertion about the database -- it would just issue two writes per
        no-op round, and DRY_RUN_WRITES_NOTHING is a property of what gets
        CALLED, not of what the callee then decides to skip.
        """
        writer = DryRunWriter()
        promote = mock.patch.object(writer, "promote", side_effect=AssertionError(
            "promote() вызван на раунде без совпадений"))
        journalled = mock.patch.object(writer, "journal", side_effect=AssertionError(
            "journal() вызван на раунде без совпадений"))
        with promote, journalled:
            merged = self._matched_nothing(writer)
        self.assertEqual(merged, [])
        self.assertEqual(writer.counts["twin"], 0)
        self.assertEqual(writer.steps_seen, [])

    def test_a_corpus_seeded_before_the_twin_rule_stops_the_pass(self):
        """No seed carries titles or a mathnet id -> both indexes would be
        empty, and every skeleton node would be tested against nothing and
        pass as "no twin". The guard says so instead.
        """
        with mock.patch.object(twin_pass, "seed_titles", return_value=[]), \
             mock.patch.object(twin_pass, "skeleton_nodes") as nodes:
            with self.assertRaises(RuntimeError) as raised:
                twin_pass.merge_twins({}, "crawl-1", DryRunWriter())
        nodes.assert_not_called()
        self.assertIn("перезапустите посев", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
