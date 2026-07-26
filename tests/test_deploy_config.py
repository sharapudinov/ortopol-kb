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


if __name__ == "__main__":
    unittest.main()
