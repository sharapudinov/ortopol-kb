#!/usr/bin/env python3
"""Node identity: a work is a work, not a source record.

The rule comes from run 85 (survey.md §7 and the verdict): OpenAlex holds
the Russian original and its English translation as two separate `works`
with separate counters, and treating them as two nodes understated the
Jaccard overlap of citer sets threefold (SOBOLEV2019: 0.09 -> 0.46). So a
node is the union of every record that shares any identifier with it --
openalex / doi / mag / pmid / pmcid -- and the canonical key is the
OpenAlex id of the record seen first; the rest survive in
external_ids.aliases so that a later record arriving under any of them
lands on the same node instead of creating a twin.

Union, not "pick the best record": the fields are merged field by field,
first non-empty wins, because which record carries the abstract and which
carries the reference list is not predictable in advance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

from citation_vocab import WorkKind
from .openalex_client import restore_abstract, short_id

ID_FIELDS = ("openalex", "doi", "mag", "pmid", "pmcid")


def normalize_doi(value: str | None) -> str:
    if not value:
        return ""
    return (
        str(value).strip().lower()
        .replace("https://doi.org/", "")
        .replace("http://dx.doi.org/", "")
        .rstrip(".,")
    )


def record_ids(record: dict) -> set[str]:
    """Every identifier a record claims, namespaced so that a DOI can never
    collide with a MAG id that happens to read the same."""
    found: set[str] = set()
    raw = record.get("ids") or {}
    for field_name in ID_FIELDS:
        value = raw.get(field_name)
        if not value:
            continue
        if field_name == "doi":
            found.add("doi:" + normalize_doi(value))
        elif field_name == "openalex":
            found.add("openalex:" + short_id(value))
        else:
            found.add(f"{field_name}:{short_id(value)}")
    if record.get("id"):
        found.add("openalex:" + short_id(record["id"]))
    if record.get("doi"):
        found.add("doi:" + normalize_doi(record["doi"]))
    return {i for i in found if not i.endswith(":")}


def record_title(record: dict) -> str | None:
    for candidate in (record.get("title"), record.get("display_name")):
        if candidate and str(candidate).strip():
            return str(candidate).strip()
    return None


def record_authors(record: dict) -> list[str]:
    names = []
    for authorship in record.get("authorships") or []:
        author = authorship.get("author") or {}
        name = author.get("display_name") or authorship.get("raw_author_name")
        if name:
            names.append(str(name))
    return names


class ScoringFields(NamedTuple):
    """Everything the tau filter reads off a candidate record, and no more.

    A candidate is not a Node until it has passed tau: absorb() namespaces
    every id the record claims, lists its authors, normalises its DOI and
    keeps a copy of the record itself, and at depth-2 candidate counts
    (thousands per level, nine in ten dropped) that was built for every
    candidate and then built AGAIN by registry.add() for the ones kept.
    Scoring needs the title and the abstract; the key names the row it
    belongs to.

    The abstract stays INVERTED here and is restored on read. OpenAlex
    ships it as {word: [positions]}, and turning that back into text is a
    sort and a join per candidate -- work frontier.vectors_for() asks for
    only where the vector is not already in citation.work, which on every
    --resume re-crawl is a small minority of the level. (The text is what
    the embedder's memo is keyed by, so a candidate the memo answers for is
    restored and then not embedded; what the memo saves is the inference,
    not the restore.) Node.absorb() keeps its own eager restore: the rows
    actually written need the text.
    """
    key: str
    title: str | None
    abstract_index: dict | None

    @property
    def abstract(self) -> str | None:
        return restore_abstract(self.abstract_index) or None


def scoring_fields(record: dict) -> ScoringFields:
    return ScoringFields(
        short_id(record.get("id")),
        record_title(record),
        record.get("abstract_inverted_index"),
    )


@dataclass
class Node:
    key: str
    kind: str
    depth: int
    document_id: str | None = None
    relation: str | None = None
    discovered_from: str | None = None
    title: str | None = None
    abstract: str | None = None
    abstract_source: str | None = None
    year: int | None = None
    doi: str | None = None
    authors: list[str] = field(default_factory=list)
    aliases: set[str] = field(default_factory=set)
    referenced_works: set[str] = field(default_factory=set)
    records: list[dict] = field(default_factory=list)
    score: float | None = None
    embedding: list[float] | None = None
    zbmath_id: str | None = None
    # Summed over the node's records, not maxed: twins carry separate
    # counters (survey §7), and `cites:W1|W2` asks for both citer sets, so
    # the sum is what expanding this node upward would actually cost.
    cited_by_count: int = 0
    # Every name the work is known by. For a seed these are the Russian and
    # English citations off its Math-Net page, cached here so the twin rule
    # (twins.py) has both languages without going back to the network.
    titles: list[str] = field(default_factory=list)
    years: list[int] = field(default_factory=list)
    twin_of: str | None = None

    def openalex_ids(self) -> list[str]:
        return sorted(a.split(":", 1)[1] for a in self.aliases if a.startswith("openalex:"))

    def external_ids(self) -> dict:
        grouped: dict[str, list[str]] = {}
        for alias in sorted(self.aliases):
            namespace, _, value = alias.partition(":")
            grouped.setdefault(namespace, []).append(value)
        grouped["aliases"] = sorted(self.aliases)
        if self.titles:
            grouped["titles"] = list(self.titles)
        if self.years:
            grouped["years"] = sorted(self.years)
        return grouped

    def absorb(self, record: dict) -> None:
        # The ids are extracted below; what is KEPT is the record without
        # referenced_works, because self.records becomes
        # citation.work.evidence (store.PostgresWriter.evidence_of) and that
        # one field is bulkier than everything else the record carries put
        # together -- tens of thousands of ids per level, held for the whole
        # crawl and then written into JSONB. referenced_works_count, which
        # OpenAlex sends alongside, stays: it is what the reference volume
        # is measured by (citations/hub_report.py). A copy, not a pop: the
        # caller's own record still feeds edges.among_known.
        self.referenced_works |= {short_id(r) for r in (record.get("referenced_works") or [])}
        self.records.append({k: v for k, v in record.items() if k != "referenced_works"})
        self.aliases |= record_ids(record)
        self.title = self.title or record_title(record)
        if not self.abstract:
            recovered = restore_abstract(record.get("abstract_inverted_index"))
            if recovered:
                self.abstract, self.abstract_source = recovered, "openalex"
        self.year = self.year or record.get("publication_year")
        self.doi = self.doi or (normalize_doi(record.get("doi")) or None)
        if not self.authors:
            self.authors = record_authors(record)
        self.cited_by_count += int(record.get("cited_by_count") or 0)
        if record.get("publication_year") and record["publication_year"] not in self.years:
            self.years.append(int(record["publication_year"]))
        for name in (record.get("title"), record.get("display_name")):
            if name and str(name).strip() and str(name).strip() not in self.titles:
                self.titles.append(str(name).strip())


class WorkRegistry:
    """The set of nodes this crawl knows, indexed by every id each claims."""

    def __init__(self):
        self.nodes: dict[str, Node] = {}
        self._by_id: dict[str, str] = {}

    def __len__(self) -> int:
        return len(self.nodes)

    def key_for(self, identifier: str) -> str | None:
        return self._by_id.get(identifier)

    def find(self, record: dict) -> str | None:
        """Key of the node this record belongs to, if any id already known."""
        for identifier in record_ids(record):
            key = self._by_id.get(identifier)
            if key:
                return key
        return None

    def add(
        self,
        record: dict,
        *,
        kind: str,
        depth: int,
        document_id: str | None = None,
        relation: str | None = None,
        discovered_from: str | None = None,
    ) -> tuple[Node, bool]:
        """(node, is_new). An existing node absorbs the record instead of
        being duplicated, and never loses `our-document` to a later
        `external-skeleton` sighting of the same work."""
        existing_key = self.find(record)
        if existing_key is not None:
            node = self.nodes[existing_key]
            node.absorb(record)
            if kind == WorkKind.OUR_DOCUMENT:
                node.kind, node.document_id = kind, document_id or node.document_id
            self._reindex(node)
            return node, False

        key = short_id(record.get("id")) or next(iter(sorted(record_ids(record))), None)
        if not key:
            raise ValueError(f"запись без единого идентификатора: {str(record)[:200]}")
        node = Node(
            key=key,
            kind=kind,
            depth=depth,
            document_id=document_id,
            relation=relation,
            discovered_from=discovered_from,
        )
        node.absorb(record)
        self.nodes[key] = node
        self._reindex(node)
        return node, True

    def release_written(self, keys) -> None:
        """Drops the payload the WRITE consumed, on nodes just written.

        A node outlives its row: the next level reads its ids, its relation
        and its reference list off this registry, so it cannot be forgotten
        when it is written. Two of its fields cannot be read again, though
        -- store's works() takes the vector and the raw source records once,
        and nothing else ever asks for either. Held to the end of the crawl
        they are 1024 floats plus a record list that grows on every
        re-sighting, multiplied by every node ever kept, which is exactly
        the peak the level-at-a-time scoring was built to bound.

        A key with no node is ignored: the caller names what it wrote, and
        the registry is the one that knows what is still there.
        """
        for key in keys:
            node = self.nodes.get(key)
            if node is not None:
                node.embedding, node.records = None, []

    def _reindex(self, node: Node) -> None:
        for identifier in node.aliases:
            self._by_id[identifier] = node.key

    def resolve_openalex(self, openalex_id: str) -> str | None:
        return self._by_id.get("openalex:" + short_id(openalex_id))
