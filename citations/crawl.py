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
expanded frontier node (n_found candidates from it, n_kept of them), one
`keep`/`drop` row per candidate, `seed`/`seed-missing` at depth 0.

Every keep and every drop carries its score in the reason, in a fixed
machine-readable form -- `score=0.6123 tau=0.5000 relation=cites` -- so the
score distribution at depths the tau calibration never saw is a query, not a
re-crawl:

    SELECT depth, action,
           substring(reason from 'score=(-?[0-9.]+)')::float8 AS score
    FROM citation.crawl_step WHERE crawl_id = ...
"""
from __future__ import annotations

from . import edges as edges_mod
from . import journal, seeding
from .frontier import candidate_text, centroid, cosine
from .openalex_client import short_id
from .registry import Node, WorkRegistry

# A node cited more than this is not asked "who cites you": the answer is
# tens of thousands of works about the field, not about the node. Default
# measured, not guessed -- see expandable()'s docstring.
HUB_CAP = 1000


class Snowball:
    def __init__(self, client, embed, writer, *, tau, crawl_id, log=print,
                 skip_keys=frozenset(), hub_cap=HUB_CAP):
        """`embed` is a callable list[str] -> list[list[float]]; the model is
        bound by the caller from corpus.embedding_model, never chosen here."""
        self.client = client
        self.embed = embed
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
        """See seeding.seed(): kept as a method so the crawl reads as one
        object, implemented there so this file stays about traversal."""
        return seeding.seed(self, documents, matches, abstracts, names)

    def embed_nodes(self, nodes) -> list[list[float]]:
        vectors = self.embed([candidate_text(n.title, n.abstract) for n in nodes])
        for node, vector in zip(nodes, vectors):
            node.embedding = vector
        return vectors

    def write_seeds(self) -> None:
        self.writer.works([self.registry.nodes[k] for k in self.seed_keys])

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
        return [k for k in keys if self.registry.nodes[k].relation == "cites"]

    def gather(self, frontier_keys: list[str]) -> tuple[list, dict, list]:
        """(candidates, per-frontier-node found counts, hub skips) for a level.

        A candidate is (record, relation, discovered_from) with relation in
        {'cites', 'referenced'}. Records already known to the registry are
        returned too -- their edges still count -- and the caller recognises
        them as not-new rather than re-scoring them.

        A node past the hub cap is not asked upward: its citer set is huge
        and, being huge, is about the field rather than about this work. It
        still expands DOWNWARD, because its references came free with the
        record and cost no request.
        """
        frontier = [self.registry.nodes[k] for k in frontier_keys]
        hubs = [n for n in frontier if n.cited_by_count > self.hub_cap]
        hub_keys = {n.key for n in hubs}
        owner = {i: node.key for node in frontier if node.key not in hub_keys
                 for i in node.openalex_ids()}
        candidates: list[tuple[dict, str, str]] = []
        found: dict[str, int] = {key: 0 for key in frontier_keys}

        for record in self.client.citers_of(sorted(owner)):
            hit = sorted({owner[short_id(r)] for r in (record.get("referenced_works") or [])
                          if short_id(r) in owner})
            if not hit:
                continue
            for key in hit:
                found[key] += 1
            candidates.append((record, "cites", hit[0]))

        wanted: dict[str, str] = {}
        for node in frontier:
            for reference in sorted(node.referenced_works):
                found[node.key] += 1
                if self.registry.resolve_openalex(reference) is None:
                    wanted.setdefault(reference, node.key)
        for record in self.client.works_by_ids(sorted(wanted)):
            candidates.append((record, "referenced",
                               wanted.get(short_id(record.get("id")), "")))
        return candidates, found, [(n.key, n.cited_by_count) for n in hubs]

    def score(self, candidates) -> list[dict]:
        """Cosine to the seed centroid for every NEW candidate, in one pass.

        A candidate with no title carries no semantic content (the predicate
        pg_embed.py applies to a page) and is scored -1.0 rather than
        embedded: an empty string would land somewhere arbitrary on the
        sphere instead of being visibly unusable."""
        fresh, seen = [], set()
        for record, relation, source_key in candidates:
            identity = short_id(record.get("id"))
            if identity in seen or self.registry.find(record) is not None:
                continue
            seen.add(identity)
            fresh.append((record, relation, source_key))

        holders = []
        for record, _relation, _source in fresh:
            holder = Node(key=short_id(record.get("id")), kind="external-skeleton", depth=0)
            holder.absorb(record)
            if holder.title:
                holders.append(holder)
        vectors = self.embed([candidate_text(h.title, h.abstract) for h in holders]) \
            if holders else []
        scored = {h.key: (cosine(v, self.centroid), v) for h, v in zip(holders, vectors)}

        out = []
        for record, relation, source_key in fresh:
            key = short_id(record.get("id"))
            score, vector = scored.get(key, (-1.0, None))
            out.append({
                "record": record, "relation": relation, "discovered_from": source_key,
                "candidate_key": key, "score": score, "vector": vector,
                "title": record.get("title") or record.get("display_name"),
                "year": record.get("publication_year"),
            })
        return out

    def expand(self, frontier_keys: list[str], depth: int) -> list[str]:
        """One level, written. Returns the keys kept at this level."""
        candidates, found, hubs = self.gather(frontier_keys)
        scored = self.score(candidates)
        steps, kept_keys = [], []
        for key, cited_by in hubs:
            steps.append(journal.hub_skip(self.crawl_id, depth, key, cited_by, self.hub_cap))
        kept_per_frontier: dict[str, int] = {key: 0 for key in frontier_keys}

        for item in scored:
            if item["score"] < self.tau:
                steps.append(journal.drop(
                    self.crawl_id, depth, item["candidate_key"], item["score"],
                    self.tau, item["relation"], item["discovered_from"]))
                continue
            node, is_new = self.registry.add(
                item["record"], kind="external-skeleton", depth=depth,
                relation=item["relation"], discovered_from=item["discovered_from"])
            # Two kept candidates can be one work: score() cannot know that,
            # because the twin union only happens on add(). The second record
            # merges into the node the first created, so the node is written
            # ONCE (a duplicate in the batch aborts the whole upsert with
            # "ON CONFLICT DO UPDATE command cannot affect row a second time")
            # while both candidates keep their own journal row -- the merge is
            # a decision the journal should show, not hide.
            if is_new:
                node.score, node.embedding = item["score"], item["vector"]
                kept_keys.append(node.key)
            kept_per_frontier[item["discovered_from"]] = \
                kept_per_frontier.get(item["discovered_from"], 0) + 1
            steps.append(journal.keep(
                self.crawl_id, depth, item["candidate_key"], node.key,
                item["score"], self.tau, item["relation"], item["discovered_from"]))

        for key in frontier_keys:
            steps.append(journal.fetch(self.crawl_id, depth, key,
                                       found.get(key, 0), kept_per_frontier.get(key, 0)))

        self.writer.works([self.registry.nodes[k] for k in kept_keys])
        edges = edges_mod.among_known(self.registry, frontier_keys, candidates)
        self.writer.edges(edges)
        self.writer.journal(steps)
        kept_rows = sum(1 for s in steps if s["action"] == "keep")
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
        candidates, _found, _hubs = self.gather(self.seed_keys)
        self.candidate_refs = {
            short_id(record.get("id")):
                {short_id(r) for r in (record.get("referenced_works") or [])}
            for record, _relation, _source in candidates
        }
        return [{"candidate_key": item["candidate_key"], "depth": 1,
                 "relation": item["relation"], "score": item["score"],
                 "title": item["title"], "year": item["year"],
                 "has_abstract": bool(item["record"].get("abstract_inverted_index")),
                 "n_references": len(item["record"].get("referenced_works") or [])}
                for item in self.score(candidates)]
