import io
import json
import unittest
from contextlib import redirect_stdout, redirect_stderr
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from backend.cli import main
from backend.console import run_query_session
from backend.llm import OllamaConfig, generate_answer
from backend.llm.ollama import load_system_prompt
from test_pipeline import Catalog


class OllamaTests(unittest.TestCase):
    def setUp(self):
        self.config = OllamaConfig(model="test-model")
        self.matches = [{"id": "a", "text": "Roof membrane specification.",
                         "source": "roof.pdf", "page": 2, "chunk_index": 0}]

    def test_single_entry_point_sends_prompt_question_and_full_passages(self):
        response = io.BytesIO(json.dumps({"message": {"content": "Answer [S1]"}, "done": True}).encode())
        with patch("backend.llm.ollama.urlopen", return_value=response) as request:
            result = generate_answer("Which membrane?", self.matches, config=self.config)
        payload = json.loads(request.call_args.args[0].data)
        self.assertEqual(request.call_args.args[0].full_url, "http://localhost:11434/api/chat")
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["model"], "test-model")
        self.assertEqual(payload["messages"][0], {"role": "system", "content": load_system_prompt()})
        content = payload["messages"][1]["content"]
        self.assertIn("USER QUESTION: Which membrane?", content)
        self.assertIn(self.matches[0]["text"], content)
        self.assertIn("[S1] roof.pdf | page 2", content)
        self.assertTrue(content.endswith("ANSWER:"))
        self.assertEqual(result["sources"][0]["citation"], "S1")
        self.assertEqual(result["answer"], "Answer [S1]")

    def test_empty_context_and_budget_do_not_call_server(self):
        with patch("backend.llm.ollama.urlopen") as request:
            self.assertEqual(generate_answer("roof", [], config=self.config)["sources"], [])
            with self.assertRaises(ValueError):
                generate_answer("roof", self.matches, config=OllamaConfig("test", max_context_chars=1))
            request.assert_not_called()

    def test_echoed_question_is_not_accepted_as_answer(self):
        for answer in ("Which membrane?", "**WHICH MEMBRANE?**"):
            response = io.BytesIO(json.dumps({"message": {"content": answer}, "done": True}).encode())
            with patch("backend.llm.ollama.urlopen", return_value=response), \
                 self.assertRaisesRegex(RuntimeError, "repeated the question"):
                generate_answer("Which membrane?", self.matches)

    def test_connection_http_and_malformed_response_errors(self):
        for error in (URLError("offline"), TimeoutError(), HTTPError("url", 404, "missing", {}, None)):
            with patch("backend.llm.ollama.urlopen", side_effect=error), self.assertRaises(RuntimeError):
                generate_answer("roof", self.matches, config=self.config)
        for body in (b"not-json", b"[]", b'{"message":{"content":""},"done":true}',
                     b'{"message":{"content":"partial"},"done":false}'):
            with patch("backend.llm.ollama.urlopen", return_value=io.BytesIO(body)), self.assertRaises(RuntimeError):
                generate_answer("roof", self.matches, config=self.config)

    def test_retrieval_only_never_calls_llm_and_enabled_uses_hybrid(self):
        with patch("backend.llm.generate_answer") as generate, redirect_stdout(io.StringIO()):
            run_query_session(Catalog(), "roof")
            generate.assert_not_called()
        answer = {"answer": "Grounded answer", "sources": [], "model": "test"}
        with patch("backend.llm.generate_answer", return_value=answer) as generate, redirect_stdout(io.StringIO()):
            run_query_session(Catalog(), "roof", llm_config=self.config)
        self.assertEqual(generate.call_args.args[0], "roof")
        self.assertTrue(all("rrf_score" in match for match in generate.call_args.args[1]))
        self.assertEqual(generate.call_args.kwargs["config"], self.config)

    def test_cli_llm_flag_uses_defaults(self):
        with patch("sys.argv", ["backend", "--llm", "roof"]), \
             patch("backend.cli.Vectordb"), patch("backend.cli.run_query_session") as run:
            main()
        self.assertEqual(run.call_args.kwargs["question"], "roof")
        self.assertEqual(run.call_args.kwargs["llm_config"].model, "gemma3:1b")

    def test_default_generation_always_includes_prompt(self):
        response = io.BytesIO(b'{"message":{"content":"Answer"},"done":true}')
        with patch("backend.llm.ollama.urlopen", return_value=response) as request:
            generate_answer("roof", self.matches)
        payload = json.loads(request.call_args.args[0].data)
        self.assertEqual(payload["model"], "gemma3:1b")
        self.assertEqual(payload["messages"][0]["content"], load_system_prompt())


if __name__ == "__main__":
    unittest.main()
