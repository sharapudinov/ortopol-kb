# ortopol-pg:17-age1.7-pgvector

Postgres 17 image carrying both extensions the kb schema needs: `age`
(graph, task 038 citation graph) and `vector` (already in production use,
`corpus.pages.embedding`). Built from `apache/age`, pgvector added via PGDG
apt package -- rationale in the Dockerfile header.

## Build

```bash
docker build -t ortopol-pg:17-age1.7-pgvector kb/deploy/pg
```

## Activation (per session, per AGE's own README)

`00_extensions.sql` runs `CREATE EXTENSION IF NOT EXISTS age;` at init time
(catalog-level, persists). `LOAD 'age'` and
`SET search_path = ag_catalog, "$user", public` are session-local per AGE's
docs and are issued by each client (`pg_search.py`, `psql -c`), not baked
into the image or `postgresql.conf` -- no `shared_preload_libraries` change
is needed either; the base image's docker-entrypoint is untouched.

## Live instance (ortopol-pg, 127.0.0.1:5470, volume ortopol-pg-data)

Switched 2026-09-02 from `pgvector/pgvector:pg17` (17.10) to this image
(17.11) on the same volume: stop + rename the old container to
`ortopol-pg-old`, `docker run -d --name ortopol-pg --restart unless-stopped
-p 127.0.0.1:5470:5432 -v ortopol-pg-data:/var/lib/postgresql/data:z` with
the same POSTGRES_* env, then `CREATE EXTENSION IF NOT EXISTS age;`.

The base image moved from Debian 12 to 13, i.e. glibc 2.36 -> 2.41, and
Postgres logged `collation version mismatch` for every database: btree
indexes over text columns (documents_pkey among them) are ordered by the
old glibc and may miss rows under the new one. Done right after the switch:
`REINDEX DATABASE ortopol;` (0.8 s here) then `ALTER DATABASE <db> REFRESH
COLLATION VERSION;` for ortopol, template1 and postgres. Any later image
change that moves glibc again needs the same two steps -- including a
rollback to the Debian 12 image.

## Rollback

Live: `DROP EXTENSION age CASCADE;` FIRST -- the extension's catalog objects
live in the data volume and the old image has no age.so to serve them --
then remove the new container, rename `ortopol-pg-old` back and start it
(17.11 -> 17.10 on one data directory is a supported minor downgrade).
Artifact: `kb-pg.image` back to `pgvector/pgvector:pg17`, drop `build:`,
drop the `CREATE EXTENSION age` line from `init/00_extensions.sql`.

## Licenses (first-party, checked 2026-09-02)

- Apache AGE: Apache License 2.0 --
  https://github.com/apache/age/blob/master/LICENSE
- pgvector: PostgreSQL License (permissive, BSD/MIT-style) --
  https://github.com/pgvector/pgvector/blob/master/LICENSE

## Integration check on a volume clone (never touches live ortopol-pg)

```bash
docker build -t ortopol-pg:17-age1.7-pgvector kb/deploy/pg
docker run --rm -v ortopol-pg-data:/from:ro -v ortopol-pg-clone:/to alpine \
  sh -c 'cp -a /from/. /to/'
docker run -d --name ortopol-pg-clone -p 127.0.0.1:5472:5432 \
  -v ortopol-pg-clone:/var/lib/postgresql/data ortopol-pg:17-age1.7-pgvector

set -a; . corpus/.pgenv; set +a; export PGPORT=5472
psql -c "CREATE EXTENSION IF NOT EXISTS age;" \
     -c "SELECT extname, extversion FROM pg_extension;"
psql -c "LOAD 'age'; SET search_path = ag_catalog, \"\$user\", public;
  SELECT * FROM create_graph('smoke');
  SELECT * FROM cypher('smoke', \$\$ CREATE (a:T {k:1})-[:R]->(b:T {k:2}) RETURN a \$\$) AS (a agtype);
  SELECT * FROM cypher('smoke', \$\$ MATCH (a)-[:R]->(b) RETURN a.k, b.k \$\$) AS (x agtype, y agtype);
  SELECT * FROM drop_graph('smoke', true);"

cd kb && PGPORT=5472 python3 pg_search.py "повторные средние" --mode hybrid
cd kb && PGPORT=5472 python3 corpus_completeness.py

docker rm -f ortopol-pg-clone && docker volume rm ortopol-pg-clone
```
