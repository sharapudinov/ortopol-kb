"""citations/registry.py's pure helpers, on the shapes OpenAlex actually
sends -- including the ones it sometimes omits.

These functions decide node IDENTITY: which records are the same work, what
it is called, who wrote it. They were exercised only through crawl-level
tests, where a KeyError on a record missing 'authorships' surfaces as an
obscure failure four layers up, and where a branch no fixture happens to
hit is not exercised at all. Nothing here needs a network, a cache or a
database.
"""
from __future__ import annotations

import unittest

import _pathfix  # noqa: F401

from citations.registry import (
    Node,
    WorkRegistry,
    normalize_doi,
    record_authors,
    record_ids,
    record_title,
    scoring_fields,
)


class NormalizeDoiTests(unittest.TestCase):
    """One work, one DOI string: the same doi under two spellings would
    make two nodes out of one work, which is the failure the whole module
    exists to prevent.
    """

    def test_none_and_empty_are_the_empty_string(self):
        self.assertEqual(normalize_doi(None), "")
        self.assertEqual(normalize_doi(""), "")

    def test_the_https_resolver_prefix_is_stripped(self):
        self.assertEqual(normalize_doi("https://doi.org/10.1070/SM1989V064N01ABEH003295"),
                         "10.1070/sm1989v064n01abeh003295")

    def test_the_old_dx_prefix_is_stripped_too(self):
        self.assertEqual(normalize_doi("http://dx.doi.org/10.4213/sm123"),
                         "10.4213/sm123")

    def test_case_and_surrounding_space_do_not_make_a_second_doi(self):
        self.assertEqual(normalize_doi("  10.4213/SM123  "), "10.4213/sm123")

    def test_a_trailing_sentence_mark_is_not_part_of_the_doi(self):
        self.assertEqual(normalize_doi("10.4213/sm123.,"), "10.4213/sm123")


class RecordIdsTests(unittest.TestCase):
    def test_a_record_with_no_ids_block_still_yields_its_own_id(self):
        self.assertEqual(record_ids({"id": "https://openalex.org/W7"}), {"openalex:W7"})

    def test_a_record_with_nothing_at_all_claims_nothing(self):
        self.assertEqual(record_ids({}), set())
        self.assertEqual(record_ids({"ids": None}), set())

    def test_every_namespace_is_prefixed_so_two_kinds_cannot_collide(self):
        found = record_ids({"ids": {"openalex": "https://openalex.org/W1",
                                    "doi": "https://doi.org/10.1/X",
                                    "mag": "12345", "pmid": "999", "pmcid": "PMC7"}})
        self.assertEqual(found, {"openalex:W1", "doi:10.1/x", "mag:12345",
                                 "pmid:999", "pmcid:PMC7"})

    def test_two_namespaces_carrying_the_same_digits_stay_two_ids(self):
        found = record_ids({"ids": {"mag": "12345", "pmid": "12345"}})
        self.assertEqual(found, {"mag:12345", "pmid:12345"})
        self.assertEqual(len(found), 2)

    def test_a_url_shaped_id_is_reduced_to_its_tail(self):
        """OpenAlex sends pmid/pmcid as resolver urls and mag as a bare
        number; the same work seen under both spellings must be one id.
        """
        self.assertEqual(record_ids({"ids": {"pmid": "https://pubmed.ncbi.nlm.nih.gov/999"}}),
                         {"pmid:999"})

    def test_a_null_or_empty_value_is_not_an_id(self):
        self.assertEqual(record_ids({"ids": {"openalex": None, "doi": "", "mag": 0}}),
                         set())

    def test_the_top_level_doi_and_the_ids_block_agree_on_one_entry(self):
        found = record_ids({"id": "https://openalex.org/W1",
                            "doi": "https://doi.org/10.1/X",
                            "ids": {"doi": "10.1/x"}})
        self.assertEqual(found, {"openalex:W1", "doi:10.1/x"})

    def test_a_namespace_left_with_no_value_is_dropped_rather_than_shipped(self):
        """A bare "doi:" would match every other record whose doi also
        normalised away -- one namespace prefix uniting unrelated works.
        """
        self.assertEqual(record_ids({"doi": "https://doi.org/"}), set())


class RecordTitleTests(unittest.TestCase):
    def test_title_wins_over_display_name(self):
        self.assertEqual(record_title({"title": "Т", "display_name": "D"}), "Т")

    def test_display_name_is_the_fallback(self):
        self.assertEqual(record_title({"title": None, "display_name": "D"}), "D")

    def test_a_blank_title_is_no_title(self):
        self.assertIsNone(record_title({"title": "   ", "display_name": None}))
        self.assertIsNone(record_title({}))

    def test_a_blank_title_falls_through_to_the_display_name(self):
        self.assertEqual(record_title({"title": "  ", "display_name": "D"}), "D")

    def test_the_answer_is_trimmed(self):
        self.assertEqual(record_title({"title": "  Т  "}), "Т")


class RecordAuthorsTests(unittest.TestCase):
    def test_a_record_without_authorships_has_no_authors(self):
        self.assertEqual(record_authors({}), [])
        self.assertEqual(record_authors({"authorships": None}), [])

    def test_a_display_name_is_taken_from_the_author_object(self):
        self.assertEqual(
            record_authors({"authorships": [{"author": {"display_name": "И. И. Ш."}}]}),
            ["И. И. Ш."])

    def test_an_authorship_with_no_author_object_falls_back_to_the_raw_name(self):
        self.assertEqual(
            record_authors({"authorships": [{"raw_author_name": "Sharapudinov I I"}]}),
            ["Sharapudinov I I"])

    def test_a_null_author_object_is_not_an_attribute_error(self):
        self.assertEqual(
            record_authors({"authorships": [{"author": None,
                                             "raw_author_name": "Raw"}]}),
            ["Raw"])

    def test_an_authorship_naming_nobody_is_skipped_not_recorded_as_blank(self):
        names = record_authors({"authorships": [
            {"author": {"display_name": None}, "raw_author_name": None},
            {"author": {"display_name": "Второй"}},
        ]})
        self.assertEqual(names, ["Второй"])

    def test_the_order_the_source_gave_is_kept(self):
        names = record_authors({"authorships": [
            {"author": {"display_name": "Первый"}},
            {"author": {"display_name": "Второй"}},
        ]})
        self.assertEqual(names, ["Первый", "Второй"])


class ScoringFieldsTests(unittest.TestCase):
    """What the tau filter reads off a candidate, and no more: building a
    whole Node for every one of thousands of dropped candidates was the
    cost this shortcut exists to avoid.
    """

    def test_a_bare_record_scores_on_its_key_alone(self):
        fields = scoring_fields({"id": "https://openalex.org/W1"})
        self.assertEqual(fields, ("W1", None, None))

    def test_the_abstract_is_reassembled_from_the_inverted_index(self):
        fields = scoring_fields({"id": "W1", "title": "Т",
                                 "abstract_inverted_index": {"чебышёв": [1], "многочлен": [0]}})
        self.assertEqual(fields.abstract, "многочлен чебышёв")

    def test_an_empty_inverted_index_is_no_abstract_rather_than_an_empty_one(self):
        self.assertIsNone(scoring_fields({"id": "W1",
                                          "abstract_inverted_index": {}}).abstract)


class NodeIdentityFieldsTests(unittest.TestCase):
    def _node(self, **kwargs) -> Node:
        return Node(key="W1", kind="external-skeleton", depth=1, **kwargs)

    def test_openalex_ids_are_the_openalex_aliases_unprefixed_and_sorted(self):
        node = self._node(aliases={"openalex:W9", "openalex:W2", "doi:10.1/x",
                                   "mag:5"})
        self.assertEqual(node.openalex_ids(), ["W2", "W9"])

    def test_a_node_with_no_openalex_alias_offers_none(self):
        self.assertEqual(self._node(aliases={"doi:10.1/x"}).openalex_ids(), [])

    def test_a_doi_containing_a_colon_keeps_its_tail(self):
        """partition/split on the FIRST colon only: a doi is allowed to
        carry one, and a namespace is not the whole of the alias.
        """
        node = self._node(aliases={"doi:10.1/x:y"})
        self.assertEqual(node.external_ids()["doi"], ["10.1/x:y"])

    def test_external_ids_groups_by_namespace_and_keeps_the_flat_list(self):
        node = self._node(aliases={"openalex:W1", "openalex:W2", "doi:10.1/x"})
        grouped = node.external_ids()
        self.assertEqual(grouped["openalex"], ["W1", "W2"])
        self.assertEqual(grouped["doi"], ["10.1/x"])
        self.assertEqual(grouped["aliases"], ["doi:10.1/x", "openalex:W1", "openalex:W2"])

    def test_titles_and_years_appear_only_when_there_are_any(self):
        bare = self._node(aliases={"openalex:W1"}).external_ids()
        self.assertNotIn("titles", bare)
        self.assertNotIn("years", bare)
        rich = self._node(aliases={"openalex:W1"}, titles=["Рус", "Eng"],
                          years=[1991, 1989]).external_ids()
        self.assertEqual(rich["titles"], ["Рус", "Eng"])
        self.assertEqual(rich["years"], [1989, 1991])


class AbsorbTests(unittest.TestCase):
    """Union, not "pick the best record": which record carries the abstract
    and which carries the reference list is not predictable in advance.
    """

    def _node(self) -> Node:
        return Node(key="W1", kind="external-skeleton", depth=1)

    def test_the_first_non_empty_value_of_each_field_wins(self):
        node = self._node()
        node.absorb({"id": "W1", "title": "Первое", "publication_year": 1989})
        node.absorb({"id": "W1", "title": "Второе", "publication_year": 1991})
        self.assertEqual(node.title, "Первое")
        self.assertEqual(node.year, 1989)

    def test_the_reference_list_is_kept_out_of_the_evidence_records(self):
        node = self._node()
        node.absorb({"id": "W1", "referenced_works": ["https://openalex.org/W2"],
                     "referenced_works_count": 1})
        self.assertEqual(node.referenced_works, {"W2"})
        self.assertNotIn("referenced_works", node.records[0])
        self.assertEqual(node.records[0]["referenced_works_count"], 1)

    def test_cited_by_counts_are_summed_across_the_twins(self):
        node = self._node()
        node.absorb({"id": "W1", "cited_by_count": 40})
        node.absorb({"id": "W2", "cited_by_count": 31})
        self.assertEqual(node.cited_by_count, 71)

    def test_a_missing_cited_by_count_contributes_nothing(self):
        node = self._node()
        node.absorb({"id": "W1"})
        node.absorb({"id": "W1", "cited_by_count": None})
        self.assertEqual(node.cited_by_count, 0)

    def test_both_languages_of_a_title_are_remembered_once_each(self):
        node = self._node()
        node.absorb({"id": "W1", "title": "Рус", "display_name": "Рус"})
        node.absorb({"id": "W2", "display_name": "Eng"})
        self.assertEqual(node.titles, ["Рус", "Eng"])

    def test_an_abstract_names_the_source_it_came_from(self):
        node = self._node()
        node.absorb({"id": "W1", "abstract_inverted_index": {"текст": [0]}})
        self.assertEqual((node.abstract, node.abstract_source), ("текст", "openalex"))

    def test_a_record_with_no_abstract_leaves_the_source_unclaimed(self):
        node = self._node()
        node.absorb({"id": "W1", "abstract_inverted_index": None})
        self.assertIsNone(node.abstract)
        self.assertIsNone(node.abstract_source)


class RegistryIdentityTests(unittest.TestCase):
    def test_a_record_sharing_any_id_lands_on_the_same_node(self):
        registry = WorkRegistry()
        first, is_new = registry.add({"id": "https://openalex.org/W1",
                                      "doi": "10.1/x"},
                                     kind="external-skeleton", depth=1)
        self.assertTrue(is_new)
        second, is_new = registry.add({"id": "https://openalex.org/W2",
                                       "doi": "https://doi.org/10.1/X"},
                                      kind="external-skeleton", depth=2)
        self.assertFalse(is_new)
        self.assertIs(second, first)
        self.assertEqual(len(registry), 1)
        self.assertEqual(registry.resolve_openalex("https://openalex.org/W2"), "W1")

    def test_our_document_is_never_lost_to_a_later_sighting(self):
        registry = WorkRegistry()
        registry.add({"id": "W1"}, kind="our-document", depth=0, document_id="1989_sm1")
        node, _is_new = registry.add({"id": "W1"}, kind="external-skeleton", depth=2)
        self.assertEqual(node.kind, "our-document")
        self.assertEqual(node.document_id, "1989_sm1")

    def test_a_record_with_no_identifier_at_all_is_refused(self):
        with self.assertRaises(ValueError):
            WorkRegistry().add({"title": "безымянная"}, kind="external-skeleton", depth=1)

    def test_a_record_with_only_a_doi_is_keyed_by_it(self):
        registry = WorkRegistry()
        node, _is_new = registry.add({"doi": "10.1/x"}, kind="external-skeleton", depth=1)
        self.assertEqual(node.key, "doi:10.1/x")
        self.assertEqual(registry.key_for("doi:10.1/x"), "doi:10.1/x")


class ReleaseWrittenTests(unittest.TestCase):
    """The mechanism the crawl's documented peak-memory bound rests on.

    A node outlives its row -- the next level reads its ids, its relation
    and its reference list off the registry -- but two of its fields cannot
    be read again: the vector and the raw source records, taken once by
    store's works(). Held to the end of the crawl they are 1024 floats plus
    a record list that grows on every re-sighting, multiplied by every node
    ever kept, which is exactly the peak level-at-a-time scoring bounds.
    Nothing asserted that they were actually dropped.
    """

    def _registry(self):
        registry = WorkRegistry()
        for key in ("W1", "W2"):
            node, _is_new = registry.add({"id": f"https://openalex.org/{key}",
                                          "title": f"Title {key}"},
                                         kind="external-skeleton", depth=1)
            node.embedding = [0.5] * 8
        return registry

    def test_the_payload_the_write_consumed_is_dropped(self):
        registry = self._registry()
        registry.release_written(["W1"])
        self.assertIsNone(registry.nodes["W1"].embedding)
        self.assertEqual(registry.nodes["W1"].records, [])

    def test_a_node_nobody_wrote_keeps_its_payload(self):
        registry = self._registry()
        registry.release_written(["W1"])
        self.assertEqual(registry.nodes["W2"].embedding, [0.5] * 8)
        self.assertEqual(len(registry.nodes["W2"].records), 1)

    def test_the_node_itself_survives_the_release(self):
        """Released, not forgotten: the next level resolves edges through
        this registry, so a key that vanished here is an edge endpoint the
        crawl can no longer place.
        """
        registry = self._registry()
        registry.release_written(["W1"])
        self.assertEqual(registry.resolve_openalex("W1"), "W1")

    def test_a_key_with_no_node_is_ignored(self):
        """The caller names what it WROTE; which of those the registry
        still holds is the registry's own question.
        """
        registry = self._registry()
        registry.release_written(["W1", "W_NEVER_ADDED"])
        self.assertIsNone(registry.nodes["W1"].embedding)


if __name__ == "__main__":
    unittest.main()
