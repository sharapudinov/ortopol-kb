#!/usr/bin/env python3
"""The BFS snowball itself: seeds, depth 1..N, filter, journal.

Shape of one level, for the frontier F already in the registry:

  up   -- `filter=cites:` over F in batches of 50. The response does not say
          which member of the batch a citer cites, so the edge is recovered
          from the citer's own `referenced_works` intersected with F.
  down -- F's own `referenced_works`, whose metadata is fetched in batches of
          50 by `openalex_id:`. The edge is known before the metadata
          arrives; the request buys the title and abstract the filter needs.

Only nodes KEPT at level d expand at level d+1 -- that is what makes depth 2
affordable (survey.md §8 priced the unfiltered version at ~305 requests).

Edges are derived AFTER the level's nodes are in the registry, in one pass
over every record seen, so that an edge is written whenever both of its
endpoints are known -- seed to seed included. A dropped candidate is not a
node, so nothing points at it: the `drop` journal row carrying tau and the
score is the entire record of it.

The journal follows the shape of crawl_step's columns: one `fetch` row per
expanded frontier node, one `keep`/`drop` row per candidate,
`seed`/`seed-missing` at depth 0.

Discovery is many-to-many -- a citer can cite several members of the
frontier, a work can be referenced by several of them -- so a candidate
carries the SET of frontier nodes it was reached from. The fetch row's
n_found and n_kept are counted over that set: n_found is how many of this
level's candidates this node contributed to, n_kept how many of THOSE
passed tau, so the pair is two counts of one population and every node gets
credit for every candidate it brought. Where only one name fits -- the
keep/drop row's frontier_key, the node's own evidence, an edge's provenance
-- it is the smallest key of the set (gathering.principal_hit), a
representative and not a claim of sole discovery.

Every keep and every drop carries its score and the tau it was measured
against in columns of their own, so the score distribution at depths the tau
calibration never saw is a query, not a re-crawl:

    SELECT depth, action, score
    FROM citation.crawl_step WHERE crawl_id = ...
"""
from __future__ import annotations

from citation_vocab import CrawlAction, Relation, WorkKind
from . import candidates as candidates_mod
from . import edges as edges_mod
from . import gathering, journal, seeding
from .frontier import vectors_for as frontier_vectors
from .registry import WorkRegistry
from .scoring import keeps

# A node cited more than this is not asked "who cites you": the answer is
# tens of thousands of works about the field, not about the node. Default
# measured, not guessed -- see expandable()'s docstring.
HUB_CAP = 1000


class Snowball:
    def __init__(self, client, embed, writer, *, tau, crawl_id, log=print,
                 skip_keys=frozenset(), hub_cap=HUB_CAP, known_vectors=None):
        """`embed` is a callable list[str] -> list[list[float]]; the model is
        bound by the caller from corpus.embedding_model, never chosen here.

        `known_vectors` is the read side of the same seam: a callable
        list[str] -> {key: vector} answering which of those keys the
        database already carries an embedding for (citations/inputs.
        known_embeddings). Default: nothing is known, which is what a unit
        test with no database sees.
        """
        self.client = client
        self.embed = embed
        self.known_vectors = known_vectors or (lambda keys: {})
        self.writer = writer
        self.tau = tau
        self.hub_cap = hub_cap
        self.crawl_id = crawl_id
        self.log = log
        self.skip_keys = set(skip_keys)
        self.registry = WorkRegistry()
        self.centroid: list[float] | None = None
        self.seed_keys: list[str] = []
        self.per_depth: dict[int, dict[str, int]] = {}
        self.candidate_refs: dict[str, set[str]] = {}

    # -- seeds -----------------------------------------------------------
    def seed(self, documents, matches, abstracts=None, names=None) -> list[str]:
        """The seed set, established by seeding.py and assigned here.

        Kept as a method so the crawl reads as one object; implemented there
        so this file stays about traversal. Every attribute this touches is
        set from a returned value -- seeding.py holds no Snowball.
        """
        steps, n_matched = seeding.collect_seeds(
            self.registry, self.client, self.crawl_id, documents, matches,
            abstracts, names)
        # Journalled BEFORE the centroid is computed, not after: a run where
        # OpenAlex returned nothing has no centroid and cannot continue, and
        # the error rows explaining why must survive that exit.
        self.writer.journal(steps)
        self.seed_keys, self.centroid, self.per_depth[0] = seeding.rank_seeds(
            self.registry, self.embed_nodes, len(documents), n_matched)
        return self.seed_keys

    def embed_nodes(self, nodes) -> list[list[float]]:
        """Vectors for `nodes`, in order, stored on each node on the way.

        The seeds are the same 56 works on every run and their vectors are
        already in citation.work, so the stored one stands in and only a
        genuinely new (or never-embedded) seed reaches ollama.

        The one caller that legitimately holds every vector at once: the
        centroid is the mean over all of them, and 56 seeds is not a level.
        """
        vectors = []
        for node, vector in self.vectors_for(nodes):
            node.embedding = vector
            vectors.append(vector)
        return vectors

    def vectors_for(self, holders):
        """(holder, vector) pairs for `holders` (anything carrying key/
        title/abstract), in order, a chunk at a time -- frontier.
        vectors_for() with this crawl's two seams bound.
        """
        return frontier_vectors(self.embed, self.known_vectors, holders)

    def write_seeds(self) -> None:
        self.writer.works([self.registry.nodes[k] for k in self.seed_keys])
        self.registry.release_written(self.seed_keys)

    # -- one level -------------------------------------------------------
    def expandable(self, keys: list[str], depth: int) -> list[str]:
        """Which of `keys` open the next level.

        Depth 1 expands the seeds. From depth 2 on, only nodes reached by
        `relation='cites'` -- works that cite the frontier -- expand; a node
        reached by `relation='referenced'` is a LEAF: written, never opened.

        Measured reason (2026-09-02, the aborted first depth-2 attempt): the
        down direction keeps classics -- Higher Transcendental Functions,
        Interpolation of Operators, Spectral Methods in MATLAB -- and asking
        who cites THOSE returned over 51000 works across eight batches
        (meta.count 18904, 13271, 11021, 3227, 3124, 1788, 296, 21), which
        exhausted a 1000-request window without writing a single node. It is
        also the wrong question: someone citing a 1953 handbook says nothing
        about Sharapudinov. survey §8's estimate (4.6 citers/node, 31
        requests) was measured on citers of the five key works and does not
        transfer to the classics the down direction pulls in.
        """
        if depth <= 1:
            return list(keys)
        return [k for k in keys if self.registry.nodes[k].relation == Relation.CITES]

    def gather(self, frontier_keys: list[str]):
        """One level's candidates, delegated to gathering.py.

        Kept as a method so the crawl reads as one object, implemented there
        so this file stays about traversal -- the same division seed() makes
        with seeding.py.
        """
        return gathering.gather(self.client, self.registry, frontier_keys, self.hub_cap)

    def scores_of(self, holders) -> dict[str, tuple[float, list[float] | None]]:
        """{key: (score, vector)} for `holders`, delegated to candidates.py
        with this crawl's centroid and tau bound."""
        return candidates_mod.scores_of(self.vectors_for, self.centroid, self.tau,
                                        holders)

    def score(self, candidates) -> list[dict]:
        """This level's NEW candidates, scored -- candidates.py with this
        crawl's registry and measurement bound.

        Kept as a method so the crawl reads as one object, implemented
        there so this file stays about traversal: the same division seed()
        makes with seeding.py and gather() with gathering.py.
        """
        return candidates_mod.score(self.registry, self.scores_of, candidates)

    def expand(self, frontier_keys: list[str], depth: int) -> list[str]:
        """One level, written. Returns the keys kept at this level."""
        candidates, hubs, references = self.gather(frontier_keys)
        scored = self.score(candidates)
        steps, kept_keys = [], []
        for key, cited_by in hubs:
            steps.append(journal.hub_skip(self.crawl_id, depth, key, cited_by, self.hub_cap))
        # Both counters over ONE population -- the candidates this level
        # judged -- and both crediting EVERY frontier node a candidate was
        # reached from. Counting what the frontier points at against what
        # the filter kept compared a set of references with a set of
        # candidates, and crediting one arbitrary hit lost the rest.
        found: dict[str, int] = {key: 0 for key in frontier_keys}
        kept_per_frontier: dict[str, int] = {key: 0 for key in frontier_keys}
        for item in scored:
            for key in item["hits"]:
                found[key] = found.get(key, 0) + 1

        for item in scored:
            if not keeps(item["score"], self.tau):
                steps.append(journal.drop(
                    self.crawl_id, depth, item["candidate_key"], item["score"],
                    self.tau, item["relation"], item["discovered_from"]))
                continue
            node, is_new = self.registry.add(
                item["record"], kind=WorkKind.EXTERNAL_SKELETON, depth=depth,
                relation=item["relation"], discovered_from=item["discovered_from"])
            # Two kept candidates can be one work: score() cannot know that,
            # because the twin union only happens on add(). The second record
            # merges into the node the first created, so the node is written
            # ONCE (a duplicate in the batch aborts the whole upsert with
            # "ON CONFLICT DO UPDATE command cannot affect row a second time")
            # while both candidates keep their own journal row -- the merge is
            # a decision the journal should show, not hide.
            # The reference list the record no longer carries: a node that
            # stays needs it, both to expand downward at the next level and
            # to own the edges derived from it.
            node.referenced_works |= set(references.get(item["candidate_key"], ()))
            if is_new:
                node.score, node.embedding = item["score"], item["vector"]
                kept_keys.append(node.key)
            for key in item["hits"]:
                kept_per_frontier[key] = kept_per_frontier.get(key, 0) + 1
            steps.append(journal.keep(
                self.crawl_id, depth, item["candidate_key"], node.key,
                item["score"], self.tau, item["relation"], item["discovered_from"]))

        for key in frontier_keys:
            steps.append(journal.fetch(self.crawl_id, depth, key,
                                       found.get(key, 0), kept_per_frontier.get(key, 0)))

        # Everything the filter dropped is now journalled and can go: only a
        # candidate that became -- or already was -- a node can be an edge
        # endpoint, so the rest of the level's reference lists are freed here
        # rather than held to the end of the level.
        references = {key: refs for key, refs in references.items()
                      if self.registry.resolve_openalex(key) is not None}
        self.writer.works([self.registry.nodes[k] for k in kept_keys])
        self.registry.release_written(kept_keys)
        edges = edges_mod.among_known(self.registry, frontier_keys, candidates, references)
        self.writer.edges(edges)
        self.writer.journal(steps)
        kept_rows = sum(1 for s in steps if s["action"] == CrawlAction.KEEP)
        self.per_depth[depth] = {"candidates": len(scored), "kept": kept_rows,
                                 "nodes": len(kept_keys),
                                 "dropped": len(scored) - kept_rows,
                                 "edges": len(edges)}
        return kept_keys

    def run(self, depth: int) -> dict:
        self.write_seeds()
        frontier = list(self.seed_keys)
        for level in range(1, depth + 1):
            frontier = [k for k in self.expandable(frontier, level)
                        if k not in self.skip_keys]
            if not frontier:
                self.log(f"глубина {level}: фронтир пуст, обход остановлен")
                break
            self.log(f"глубина {level}: раскрываю {len(frontier)} узлов")
            frontier = self.expand(frontier, level)
            self.log(f"глубина {level}: оставлено {len(frontier)}")
        return self.per_depth

    def calibrate(self) -> list[dict]:
        """Depth-1 candidate scores; nothing is written to citation.*.

        Fills candidate_refs on the way: what each candidate would make the
        crawl fetch at the next level. Summing per-candidate counts overstates
        that badly (measured: 15177 references against 4262 distinct works --
        neighbours cite the same things), so the cost table needs the sets.
        """
        candidates, _hubs, references = self.gather(self.seed_keys)
        self.candidate_refs = {key: set(refs) for key, refs in references.items()}
        return [{"candidate_key": item["candidate_key"], "depth": 1,
                 "relation": item["relation"], "score": item["score"],
                 "title": item["title"], "year": item["year"],
                 "has_abstract": bool(item["record"].get("abstract_inverted_index")),
                 "n_references": len(references.get(item["candidate_key"], ()))}
                for item in self.score(candidates)]
