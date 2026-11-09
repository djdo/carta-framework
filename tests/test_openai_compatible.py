import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from carta_framework import create_llm_provider
from carta_framework.llm.openai_compatible import OpenAICompatibleLLMProvider


class _FakeOpenAIServer:
    """Minimal OpenAI-compatible server (chat completions + embeddings) recording requests."""

    def __init__(self, dim=4):
        self.requests = []
        self.dim = dim
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                server.requests.append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body})
                if self.path.endswith("/embeddings"):
                    data = [{"index": i, "embedding": [float(len(t))] * server.dim} for i, t in enumerate(body["input"])]
                    self._json({"data": list(reversed(data))})  # out of order on purpose
                elif self.path.endswith("/chat/completions"):
                    if body.get("stream"):
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream")
                        self.end_headers()
                        for piece in ["Hel", "lo", " world"]:
                            chunk = {"choices": [{"delta": {"content": piece}}]}
                            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                        self.wfile.write(b'data: {"choices": [{"delta": {}}]}\n\n')
                        self.wfile.write(b"data: [DONE]\n\n")
                    else:
                        self._json({"choices": [{"message": {"content": "Hello world"}}]})
                else:
                    self.send_response(404)
                    self.end_headers()

            def _json(self, obj):
                payload = json.dumps(obj).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


class TestOpenAICompatibleProvider(unittest.TestCase):
    def setUp(self):
        self.server = _FakeOpenAIServer()

    def tearDown(self):
        self.server.close()

    def _provider(self, **kwargs):
        return OpenAICompatibleLLMProvider(base_url=self.server.url, chat_model="llama3.1:8b",
                                           embedding_model="nomic-embed-text", **kwargs)

    def test_streaming_chat_with_reply_cap(self):
        chunks = list(self._provider().chat([{"role": "user", "content": "hi"}], temperature=0.1,
                                            stream=True, max_tokens=600))
        self.assertEqual(chunks, ["Hel", "lo", " world"])
        body = self.server.requests[-1]["body"]
        self.assertEqual((body["model"], body["max_tokens"], body["stream"]), ("llama3.1:8b", 600, True))

    def test_plain_chat_without_key_sends_no_auth_header(self):
        self.assertEqual(self._provider().chat([{"role": "user", "content": "hi"}]), "Hello world")
        self.assertIsNone(self.server.requests[-1]["auth"])
        self.assertNotIn("max_tokens", self.server.requests[-1]["body"])

    def test_embeddings_keep_order_batch_and_fill_empty_texts(self):
        provider = self._provider(embedding_batch_size=2, api_key="sk-local")
        vectors = provider.generate_embeddings(["a", "bbb", "", "cc"])

        self.assertEqual([v[0] for v in vectors], [1.0, 3.0, 0.0, 2.0])
        self.assertEqual(vectors[2], [0.0] * 4)
        embed_requests = [r for r in self.server.requests if r["path"].endswith("/embeddings")]
        self.assertEqual([r["body"]["input"] for r in embed_requests], [["a", "bbb"], ["cc"]])
        self.assertEqual(embed_requests[0]["auth"], "Bearer sk-local")
        self.assertEqual(provider.get_embedding_dimension(), 4)

    def test_dimension_is_probed_once(self):
        provider = self._provider()
        self.assertEqual(provider.get_embedding_dimension(), 4)
        self.assertEqual(provider.get_embedding_dimension(), 4)
        self.assertEqual(len(self.server.requests), 1)

    def test_embeddings_can_use_a_separate_server(self):
        other = _FakeOpenAIServer(dim=8)
        try:
            provider = self._provider(embedding_base_url=other.url)
            self.assertEqual(len(provider.generate_embeddings(["x"])[0]), 8)
            self.assertEqual(self.server.requests, [])
            self.assertEqual(len(other.requests), 1)
        finally:
            other.close()

    def test_factory_presets(self):
        ollama = create_llm_provider("ollama", "", chat_model="llama3.1:8b", embedding_model="nomic-embed-text")
        self.assertEqual(ollama.base_url, "http://localhost:11434/v1")
        vllm = create_llm_provider("vllm", "", chat_model="m", embedding_model="e")
        self.assertEqual(vllm.base_url, "http://localhost:8000/v1")
        custom = create_llm_provider("openai_compatible", "k", chat_model="m", embedding_model="e",
                                     base_url=self.server.url + "/")
        self.assertEqual(custom.base_url, self.server.url)
        with self.assertRaises(ValueError):
            create_llm_provider("openai_compatible", "", chat_model="m", embedding_model="e")
        with self.assertRaises(ValueError):
            create_llm_provider("ollama", "", chat_model="m")


if __name__ == "__main__":
    unittest.main()
