# ortopol-pg:17-age1.7.0-pgvector0.8.6 -- Postgres 17 with both Apache AGE
# (graph, the citation graph) and pgvector (already required by corpus.pages).
#
# BOTH inputs are pinned by content, because this image is the one part of
# the artifact that is not: every bundled file is sha256'd into
# manifest.files and the dump carries its own hash, while the database the
# dump is restored INTO used to be a recipe resolved at the recipient's
# build time. The engine is a correctness-relevant input here, not a
# convenience -- the last base-image move changed glibc (2.36 -> 2.41) and
# required REINDEX plus REFRESH COLLATION VERSION (README.md) -- so a
# mutable tag and an apt candidate meant two recipients could certify the
# same package against two different databases. A pin that stops resolving
# is a build that fails loudly, which is the cheaper of the two failures.
#
# Path chosen: FROM apache/age, install pgvector via apt (PGDG), not the
# reverse (FROM pgvector/pgvector:pg17 + build AGE from source). Verified on
# this host before writing this file:
#   - the digest below is apache/age:release_PG17_1.7.0 (multi-arch index,
#     `docker buildx imagetools inspect`): Debian 13 "trixie", postgres 17.11
#     (Debian 17.11-1.pgdg13+2) -- a stock docker-library/postgres image with
#     AGE built in, not a bespoke base.
#   - Its /etc/apt/sources.list.d already carries the PGDG apt repo
#     (trixie-pgdg main 17, signed-by postgres.gpg.asc) -- no key/repo setup
#     needed, `apt-get update` alone resolves postgresql-17-pgvector; the
#     repo carries 0.8.6-1.pgdg13+1 (candidate) and its two predecessors, so
#     a pinned version keeps resolving across a candidate bump.
#   - Its docker-entrypoint.sh is the unmodified docker-library/postgres
#     entrypoint: no shared_preload_libraries wiring, no extra postgresql.conf.
#     AGE's own README (Post Installation) confirms activation is
#     session-local: `CREATE EXTENSION age; LOAD 'age'; SET search_path =
#     ag_catalog, "$user", public;` -- none of that needs a server restart or
#     a preload-library GUC, so the base entrypoint needs no change.
# Building AGE from source instead would mean matching apache/age's exact
# postgres 17.11 headers/toolchain ourselves for no benefit -- the PGDG binary
# path is fewer moving parts and does not touch the AGE build at all.
FROM apache/age@sha256:92a5d223965bc2e436f9eee436e0bd2c0d81f3b59124b3d197ec94706f3450a8

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        postgresql-17-pgvector=0.8.6-1.pgdg13+1 \
    && rm -rf /var/lib/apt/lists/*
