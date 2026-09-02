"""Individual predicate checks run by smoke_test.py against a freshly
deployed kb-smoke stack. Each function returns (ok, detail) rather than
raising, so smoke_test.py can run every check and report all of them even
after the first failure -- a partial pass tells the deployer more than an
early exit would.
"""
from __future__ import annotations

from deploy_pathfix import ensure_corpus_importable

ensure_corpus_importable()

from blob_integrity_checks import (  # noqa: E402,F401 -- re-exported for smoke_test.py's `checks.*`
    blob_sha256,
    check_blob_corruption_detected,
    check_blob_roundtrip,
)
from bundled_files_check import (  # noqa: E402,F401 -- re-exported for smoke_test.py's `checks.*`
    OPERATIONAL_ALLOWLIST,
    check_bundled_files,
)
from manifest_keys import (  # noqa: E402,F401 -- MANIFEST_SCHEMA_VERSION re-exported, see smoke_test.py's checks.MANIFEST_SCHEMA_VERSION
    MANIFEST_SCHEMA_VERSION,
    Key,
)
from manifest_contract import CitationMode  # noqa: E402
from ollama_registry import served_model_digest  # noqa: E402
import pg_graph_common  # noqa: E402
from pg_common import scalar, scalar_row  # noqa: E402
from vector_probe_check import (  # noqa: E402,F401 -- re-exported for smoke_test.py's `checks.*`
    VECTOR_PROBE_DISTANCE_TOLERANCE,
    VECTOR_PROBE_RANK_TOLERANCE,
    check_vector,
)


# One round trip for both independent counts, mirroring manifest_probe.
# _MANIFEST_SCALARS_SQL: two scalar subselects with no shared FROM clause,
# folded instead of two separate psql spawns for what is otherwise the
# exact same read manifest_probe.gather_manifest already made in one.
_COUNTS_SQL = """
SELECT
    (SELECT count(*) FROM corpus.documents),
    (SELECT count(*) FROM corpus.pages);
"""


def check_counts(env: dict, manifest: dict) -> tuple[bool, str]:
    docs_str, pages_str = scalar_row(env, _COUNTS_SQL, expected_columns=2)
    docs, pages = int(docs_str), int(pages_str)
    want_docs, want_pages = manifest[Key.DOCUMENTS_COUNT], manifest[Key.PAGES_COUNT]
    ok = docs == want_docs and pages == want_pages
    return ok, f"documents={docs} (manifest {want_docs}), pages={pages} (manifest {want_pages})"


def check_fulltext(env: dict, manifest: dict) -> tuple[bool, str]:
    query = manifest[Key.FULLTEXT_PROBE][Key.QUERY]
    want = manifest[Key.FULLTEXT_PROBE][Key.HITS]
    hits = int(scalar(
        env,
        "SELECT count(*) FROM corpus.pages WHERE tsv @@ phraseto_tsquery('russian', :'q');",
        variables={"q": query},
    ))
    ok = hits == want
    return ok, f'"{query}" -> {hits} страниц (манифест: {want})'


# check_vector/VECTOR_PROBE_RANK_TOLERANCE/VECTOR_PROBE_DISTANCE_TOLERANCE now
# live in vector_probe_check.py (module size) and are re-exported above so
# smoke_test.py's `checks.check_vector(...)` etc. keep working unchanged.


def check_embedding_model_digest(manifest: dict, ollama_url: str) -> tuple[bool, str]:
    """Confirms the smoke stack's ollama is serving the EXACT model build
    build_package.py pinned (manifest["embedding_model"]["digest"]), not
    merely a model with the same name/dims -- two different bge-m3
    builds/quantisations can satisfy both of those while producing vectors
    in different spaces (see check_vector's docstring on cross-process
    tolerance: that tolerance budget is meant for float summation order,
    not for a materially different model). A hard FAIL here, not a
    warning, is the only guard that actually distinguishes model identity
    from model name.
    """
    want = manifest[Key.EMBEDDING_MODEL].get(Key.DIGEST)
    if not want:
        return False, "manifest has no embedding_model.digest (rebuild the package)"
    model_name = manifest[Key.EMBEDDING_MODEL][Key.MODEL]
    try:
        got, _size_bytes = served_model_digest(ollama_url, model_name)
    except RuntimeError as exc:
        return False, str(exc)
    ok = got == want
    return ok, f"served digest={got} (manifest: {want})"


# Both reads were previously two independent scalar_row round trips (the
# declared model/dims, then the aggregate over corpus.pages); folded into
# one, mirroring manifest_probe._MANIFEST_SCALARS_SQL. WHERE id = 1 matches
# pg_search.embed_query and manifest_probe._MANIFEST_SCALARS_SQL -- see the
# comment on the latter for why an unfiltered read is unsafe. Each piece is
# a scalar subselect with no shared FROM clause, so the outer query always
# returns exactly one row even when corpus.embedding_model has no id = 1
# row or corpus.pages has no embedded rows at all; NULLs are checked
# explicitly below rather than relying on scalar_row's row-count guard.
_EMBEDDING_MODEL_DIMS_SQL = """
SELECT
    (SELECT model FROM corpus.embedding_model WHERE id = 1),
    (SELECT dims FROM corpus.embedding_model WHERE id = 1),
    (SELECT count(DISTINCT vector_dims(embedding)) FROM corpus.pages WHERE embedding IS NOT NULL),
    (SELECT coalesce(max(vector_dims(embedding))::text, '') FROM corpus.pages WHERE embedding IS NOT NULL),
    (SELECT count(*) FROM corpus.pages WHERE embedding IS NOT NULL);
"""


def check_embedding_model_dims(env: dict) -> tuple[bool, str]:
    model, declared_dims_str, distinct_count_str, only_dims, embedded_rows_str = scalar_row(
        env, _EMBEDDING_MODEL_DIMS_SQL, expected_columns=5,
    )
    if not model or not declared_dims_str:
        return False, "corpus.embedding_model is empty"
    declared_dims = int(declared_dims_str)
    distinct_count = int(distinct_count_str)
    embedded_rows = int(embedded_rows_str)

    ok = embedded_rows > 0 and distinct_count == 1 and only_dims != "" and int(only_dims) == declared_dims
    return ok, (
        f"model={model}, declared dims={declared_dims}, {embedded_rows} embedded rows, "
        f"{distinct_count} distinct vector_dims value(s)" + (f" ({only_dims})" if only_dims else "")
    )


# One round trip for both independent reads, same rationale as _COUNTS_SQL
# above.
_MEASUREMENTS_RUN_SQL = """
SELECT
    (SELECT count(*) FROM measurements.run),
    (SELECT count(*) FROM measurements.run
     WHERE question IS NULL OR arbiter IS NULL OR reproduce IS NULL);
"""


def check_measurements_run(env: dict, manifest: dict) -> tuple[bool, str]:
    count_str, incomplete_str = scalar_row(env, _MEASUREMENTS_RUN_SQL, expected_columns=2)
    count, incomplete = int(count_str), int(incomplete_str)
    want = manifest[Key.MEASUREMENTS_RUN_COUNT]
    ok = count == want and incomplete == 0
    return ok, f"rows={count} (manifest {want}), missing required fields={incomplete}"


def check_citation_projection(env: dict, manifest: dict) -> tuple[bool | None, str]:
    """AGE is loaded and citation_graph is a faithful projection of the
    restored citation.work/cites, which in turn hold what the manifest
    declares -- confirms deploy/init/02_project_graph.sql actually ran, the
    way check_measurements_run confirms 01_dump.sql.gz did.

    SKIP (None), not FAIL, when the artifact's manifest declares no shipped
    citation mode (an older artifact predating the citation schema, or one built under
    CitationMode.NONE): there is no graph to have projected, and querying
    ag_catalog/citation_graph against a database that never created either
    would be a psql error that says nothing about the package's health --
    same convention as check_measurements_run for the public profile.
    """
    citation = manifest.get(Key.CITATION, {})
    mode = citation.get(Key.CITATION_MODE)
    if mode is None or mode == CitationMode.NONE:
        return None, f"citation mode={mode!r} -- профиль не несёт граф, проверять нечего"
    seen = pg_graph_common.projection_diff(env)
    if seen is None:
        return False, "citation_graph не спроецирован после restore (init/02_project_graph.sql?)"
    # The reading (counts AND content digests) is projection_diff()'s; what
    # this check adds is WHICH counts the graph is held against. Two
    # comparisons, not one: the graph against the restored tables (the
    # projection ran and is faithful) and the restored tables against the
    # manifest (the dump carries what the package says it carries). Only the
    # first was made here before, so a dump short of its own manifest passed
    # as long as the projection reproduced the shortfall faithfully.
    faults = pg_graph_common.projection_faults(seen)
    want_work = citation.get(Key.WORK_COUNT, 0)
    want_cites = citation.get(Key.CITES_COUNT, 0)
    ok = not faults and seen.work_n == want_work and seen.cites_n == want_cites
    return ok, (f"vertices={seen.vertex_n} (work {seen.work_n}), "
                f"edges={seen.edge_n} (cites {seen.cites_n}); "
                f"манифест: work {want_work}, cites {want_cites}"
                + ("; " + "; ".join(faults) if faults else ""))


# blob_sha256/check_blob_roundtrip/check_blob_corruption_detected now live in
# blob_integrity_checks.py (module size) and are re-exported above so
# smoke_test.py's `checks.blob_sha256(...)` etc. keep working unchanged.
# check_bundled_files/OPERATIONAL_ALLOWLIST now live in bundled_files_check.py
# (module size, and it is the one check here with no Postgres/HTTP side)
# and are re-exported the same way, for the same reason.
