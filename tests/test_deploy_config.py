"""Guards that docker-compose.yml and ollama-entrypoint.sh source the
embedding model name from KB_EMBED_MODEL rather than repeating the literal
"bge-m3" in the pull/healthcheck commands (68ea14d3) -- a deployment that
overrides the model must not silently keep pulling/grepping for bge-m3.
"""
from __future__ import annotations

import unittest
from pathlib import Path

_DEPLOY_DIR = Path(__file__).resolve().parent.parent / "deploy"


class EmbedModelParameterizedTests(unittest.TestCase):
    def test_compose_declares_kb_embed_model_env_for_kb_ollama(self):
        text = (_DEPLOY_DIR / "docker-compose.yml").read_text()
        self.assertIn("KB_EMBED_MODEL: ${KB_EMBED_MODEL:-bge-m3}", text)

    def test_compose_healthcheck_greps_the_env_variable_not_a_literal(self):
        text = (_DEPLOY_DIR / "docker-compose.yml").read_text()
        healthcheck_line = next(line for line in text.splitlines() if "ollama list | grep" in line)
        self.assertIn("${KB_EMBED_MODEL", healthcheck_line)
        self.assertNotIn("'^bge-m3'", healthcheck_line)

    def test_entrypoint_resolves_model_from_kb_embed_model_env(self):
        text = (_DEPLOY_DIR / "ollama-entrypoint.sh").read_text()
        self.assertIn('model="${KB_EMBED_MODEL:-bge-m3}"', text)

    def test_entrypoint_pull_and_grep_use_the_resolved_variable_not_a_literal(self):
        text = (_DEPLOY_DIR / "ollama-entrypoint.sh").read_text()
        self.assertIn('ollama pull "$model"', text)
        self.assertIn('grep -q "^${model}"', text)
        self.assertNotIn("ollama pull bge-m3\n", text)

    def test_pgenv_example_documents_kb_embed_model(self):
        text = (_DEPLOY_DIR / ".pgenv.example").read_text()
        self.assertIn("KB_EMBED_MODEL=bge-m3", text)


IMAGE_TAG = "ortopol-pg:17-age1.7.0-pgvector0.8.6"


class PgImageWithAgeAndPgvectorTests(unittest.TestCase):
    """The kb-pg image must carry both extensions the schema needs (age for
    the citation graph, vector already in production use), the
    Dockerfile/README that build it must agree on the image tag, and
    docker-compose.yml must build that very image from the bundled pg/
    directory rather than pull pgvector's.
    """

    def test_pg_image_carries_both_extensions_in_init(self):
        text = (_DEPLOY_DIR / "init" / "00_extensions.sql").read_text()
        self.assertIn("CREATE EXTENSION IF NOT EXISTS vector;", text)
        self.assertIn("CREATE EXTENSION IF NOT EXISTS age;", text)

    def test_dockerfile_and_compose_name_same_image(self):
        dockerfile = (_DEPLOY_DIR / "pg" / "Dockerfile").read_text()
        readme = (_DEPLOY_DIR / "pg" / "README.md").read_text()
        self.assertIn("FROM apache/age@sha256:", dockerfile)
        self.assertIn(IMAGE_TAG, readme)
        # README's build/run commands must tag the image the Dockerfile
        # produces -- a renamed tag on one side and not the other is exactly
        # the drift this test exists to catch.
        self.assertIn(f"docker build -t {IMAGE_TAG}", readme)
        compose = (_DEPLOY_DIR / "docker-compose.yml").read_text()
        self.assertIn("build: ./pg", compose)
        self.assertIn(f"image: {IMAGE_TAG}", compose)
        self.assertNotIn("pgvector/pgvector:pg17", compose)

    def test_both_engine_inputs_are_pinned_by_content_not_by_a_moving_name(self):
        """Every byte of the artifact is content-addressed (manifest.files
        sha256s each bundled file, the dump carries its own hash) except the
        engine it restores INTO, which `build: ./pg` made a recipe resolved
        at the recipient's build time. A mutable tag and an unpinned apt
        candidate mean two recipients can certify the same package against
        two different databases -- and the base image is known here to be a
        correctness-relevant input, not a convenience: the last base move
        changed glibc (2.36 -> 2.41) and needed REINDEX plus REFRESH
        COLLATION VERSION (pg/README.md).
        """
        dockerfile = (_DEPLOY_DIR / "pg" / "Dockerfile").read_text()
        base = [line for line in dockerfile.splitlines() if line.startswith("FROM ")]
        self.assertEqual(len(base), 1, dockerfile)
        self.assertRegex(base[0], r"^FROM apache/age@sha256:[0-9a-f]{64}$")
        self.assertNotIn("FROM apache/age:", dockerfile)
        installed = [line for line in dockerfile.splitlines()
                     if "postgresql-17-pgvector" in line and "#" not in line]
        self.assertEqual(len(installed), 1, dockerfile)
        self.assertRegex(installed[0], r"postgresql-17-pgvector=[0-9]+\.[0-9]+\.[0-9]+\S*")

    def test_bundle_ships_the_dockerfile_compose_builds_from(self):
        import artifact_bundle
        self.assertIn("pg/Dockerfile", artifact_bundle.DEPLOY_FILES)
        self.assertIn("pg/README.md", artifact_bundle.DEPLOY_FILES)


class ProjectGraphInitScriptTests(unittest.TestCase):
    """init/02_project_graph.sql rebuilds citation_graph after the dump
    restores -- test_init_projects_graph_only_when_schema_present.
    """

    def test_init_projects_graph_only_when_schema_present(self):
        text = (_DEPLOY_DIR / "init" / "02_project_graph.sql").read_text()
        # Guarded, not unconditional: an artifact built under
        # CitationMode.NONE (or one predating the citation schema) has no citation
        # schema at all, and this script is bundled into every artifact
        # regardless of mode.
        self.assertIn("to_regclass('citation.work') IS NOT NULL", text)
        # LOAD 'age' cannot be a bare statement inside a DO block (PL/pgSQL
        # has no grammar for the utility command) -- it must go through
        # EXECUTE, confirmed against the live image (see the script's own
        # comment).
        self.assertIn("EXECUTE 'LOAD ''age'''", text)
        self.assertIn("citation.project_graph()", text)

    def test_compose_mounts_the_init_script_as_the_third_initdb_step(self):
        text = (_DEPLOY_DIR / "docker-compose.yml").read_text()
        self.assertIn(
            "./init/02_project_graph.sql:/docker-entrypoint-initdb.d/02_project_graph.sql:ro",
            text,
        )

    def test_bundle_ships_the_init_script(self):
        import artifact_bundle
        self.assertIn("init/02_project_graph.sql", artifact_bundle.DEPLOY_FILES)


if __name__ == "__main__":
    unittest.main()
