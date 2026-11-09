import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from google.genai import errors
from pydantic import ValidationError

from carta_framework import BaseConfig, create_llm_provider
from carta_framework.llm.google import EMBEDDING_DIMENSION, GoogleLLMProvider
from carta_framework.llm.factory import OPENAI_COMPATIBLE_PROVIDERS


def _embedding(values):
    return SimpleNamespace(embeddings=[SimpleNamespace(values=values)])


class TestGoogleProvider(unittest.TestCase):
    def setUp(self):
        patcher = patch("carta_framework.llm.google.genai.Client")
        self.client = patcher.start().return_value
        self.addCleanup(patcher.stop)
        sleep = patch("carta_framework.llm.google.time.sleep")
        sleep.start()
        self.addCleanup(sleep.stop)
        self.provider = GoogleLLMProvider("key")

    def test_rate_limit_is_retried(self):
        self.client.models.embed_content.side_effect = [
            errors.APIError(429, {}), _embedding([0.1] * 768), _embedding([0.2] * 768)
        ]
        vectors = self.provider.generate_embeddings(["a", "b"])
        self.assertEqual(len(vectors), 2)
        self.assertTrue(all(any(v) for v in vectors))

    def test_failed_embedding_raises_instead_of_zero_vector(self):
        self.client.models.embed_content.side_effect = errors.APIError(400, {})
        with self.assertRaises(errors.APIError):
            self.provider.generate_embeddings(["a", "b"])
        self.client.models.embed_content.reset_mock()
        self.client.models.embed_content.side_effect = errors.APIError(503, {})
        with self.assertRaises(errors.APIError):
            self.provider.generate_embeddings(["a"])
        self.assertEqual(self.client.models.embed_content.call_count, 5)

    def test_chat_sends_turns_with_roles(self):
        self.client.models.generate_content.return_value = SimpleNamespace(text="ok")
        self.provider.chat([
            {"role": "system", "content": "S"},
            {"role": "user", "content": "U1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "U2"},
        ])
        kwargs = self.client.models.generate_content.call_args.kwargs
        self.assertEqual([(c.role, c.parts[0].text) for c in kwargs["contents"]],
                         [("user", "U1"), ("model", "A1"), ("user", "U2")])
        self.assertEqual(kwargs["config"].system_instruction, "S")


class TestGoogleEmbeddingsForOtherProviders(unittest.TestCase):
    @patch("carta_framework.llm.google.genai.Client")
    def test_openrouter_and_groq_embed_like_google(self, MockClient):
        embed = MockClient.return_value.models.embed_content
        embed.return_value = _embedding([0.1] * EMBEDDING_DIMENSION)
        for name in ("openrouter", "groq"):
            with patch("carta_framework.llm.groq.Groq"):
                provider = create_llm_provider(name, "key", google_api_key="gkey")
            vectors = provider.generate_embeddings(["text", ""])
            self.assertEqual([len(v) for v in vectors], [EMBEDDING_DIMENSION] * 2)
            self.assertEqual(provider.get_embedding_dimension(), EMBEDDING_DIMENSION)
            call = embed.call_args.kwargs
            self.assertEqual(call["config"].output_dimensionality, EMBEDDING_DIMENSION)
            self.assertEqual(call["model"], GoogleLLMProvider.__init__.__defaults__[1])


class TestConfigProviders(unittest.TestCase):
    def test_config_accepts_exactly_the_factory_providers(self):
        for name in ("google", "openrouter", "groq", *OPENAI_COMPATIBLE_PROVIDERS):
            self.assertEqual(BaseConfig(llm_provider=name, _env_file=None).llm_provider, name)
        with self.assertRaises(ValidationError):
            BaseConfig(llm_provider="openai", _env_file=None)


if __name__ == "__main__":
    unittest.main()
