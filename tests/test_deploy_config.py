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


class PgImageWithAgeAndPgvectorTests(unittest.TestCase):
    """ortopol-pg:17-age1.7-pgvector (task 038.039.1) must carry both
    extensions the kb schema needs (age for the citation graph, vector
    already in production use) and the Dockerfile/README that build it must
    agree on the image tag, and docker-compose.yml must build that very
    image from the bundled pg/ directory rather than pull pgvector's.
    """

    def test_pg_image_carries_both_extensions_in_init(self):
        text = (_DEPLOY_DIR / "init" / "00_extensions.sql").read_text()
        self.assertIn("CREATE EXTENSION IF NOT EXISTS vector;", text)
        self.assertIn("CREATE EXTENSION IF NOT EXISTS age;", text)

    def test_dockerfile_and_compose_name_same_image(self):
        dockerfile = (_DEPLOY_DIR / "pg" / "Dockerfile").read_text()
        readme = (_DEPLOY_DIR / "pg" / "README.md").read_text()
        self.assertIn("FROM apache/age:release_PG17_1.7.0", dockerfile)
        self.assertIn("ortopol-pg:17-age1.7-pgvector", readme)
        # README's build/run commands must tag the image the Dockerfile
        # produces -- a renamed tag on one side and not the other is exactly
        # the drift this test exists to catch.
        self.assertIn("docker build -t ortopol-pg:17-age1.7-pgvector", readme)
        compose = (_DEPLOY_DIR / "docker-compose.yml").read_text()
        self.assertIn("build: ./pg", compose)
        self.assertIn("image: ortopol-pg:17-age1.7-pgvector", compose)
        self.assertNotIn("pgvector/pgvector:pg17", compose)

    def test_bundle_ships_the_dockerfile_compose_builds_from(self):
        import artifact_bundle
        self.assertIn("pg/Dockerfile", artifact_bundle.DEPLOY_FILES)
        self.assertIn("pg/README.md", artifact_bundle.DEPLOY_FILES)


if __name__ == "__main__":
    unittest.main()
