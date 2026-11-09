import unittest
from unittest.mock import patch, MagicMock

from carta_framework.llm.openai_compatible import OpenAICompatibleLLMProvider


class TestExtraBody(unittest.TestCase):

    def test_extra_body_is_merged_into_chat_requests(self):
        provider = OpenAICompatibleLLMProvider("http://x/v1", chat_model="m", embedding_model="e",
                                               extra_body={"reasoning": {"enabled": False}})
        response = MagicMock()
        response.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
        with patch("httpx.Client") as client:
            client.return_value.__enter__.return_value.post.return_value = response
            self.assertEqual(provider.chat([{"role": "user", "content": "q"}], max_tokens=10), "ok")
            payload = client.return_value.__enter__.return_value.post.call_args.kwargs["json"]
        self.assertEqual(payload["reasoning"], {"enabled": False})
        self.assertEqual(payload["max_tokens"], 10)


if __name__ == "__main__":
    unittest.main()
