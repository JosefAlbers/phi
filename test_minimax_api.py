import copy
import json
import os
import unittest
from unittest.mock import patch

import requests

from api import MINIMAX_MODELS, minimax_api


class TestMiniMaxAPI(unittest.TestCase):
    """Exercise real Requests serialization without making network calls."""

    def setUp(self):
        self.sent = []
        self.status = 200
        self.message = {"role": "assistant", "content": "  Answer  "}
        self.enterContext(patch.dict(os.environ, {"MINIMAX_API_KEY": "offline-test"}, clear=True))
        self.enterContext(patch("requests.adapters.HTTPAdapter.send", side_effect=self.send))

    def send(self, request, **kwargs):
        self.sent.append(request)
        response = requests.Response()
        response.status_code = self.status
        response.request = request
        response.url = request.url
        response._content = json.dumps({"choices": [{"message": self.message}]}).encode()
        return response

    def payload(self):
        return json.loads(self.sent[-1].body)

    def test_default_and_regions(self):
        self.assertEqual(MINIMAX_MODELS, ("MiniMax-M3.1-Flash", "MiniMax-M3"))
        for region, host in (("global_en", "api.minimax.io"), ("cn_zh", "api.minimaxi.com")):
            with self.subTest(region=region):
                result = minimax_api("Hello", None, verbose=False, region=region)
                request = self.sent[-1]
                self.assertEqual(request.method, "POST")
                self.assertEqual(request.url, f"https://{host}/v1/chat/completions")
                self.assertEqual(request.headers["Authorization"], "Bearer offline-test")
                self.assertEqual(request.headers["Content-Type"], "application/json")
                self.assertEqual(self.payload(), {
                    "model": "MiniMax-M3.1-Flash",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "temperature": 0.9,
                    "top_p": 0.95,
                    "max_tokens": 8192,
                    "stream": False,
                })
                # Omitted thinking/effort use adaptive thinking and default effort.
                self.assertEqual(result["responses"], "Answer")
                self.assertEqual(result["history"][-1], self.message)

    def test_explicit_models_are_not_rewritten(self):
        for region in ("global_en", "cn_zh"):
            for model in (*MINIMAX_MODELS, "MiniMax-M2.7", "saved-custom-model"):
                with self.subTest(region=region, model=model):
                    minimax_api("Hello", [], verbose=False, region=region, api_model=model)
                    self.assertEqual(self.payload()["model"], model)
                    for field in ("thinking", "reasoning_effort", "reasoning_split", "service_tier"):
                        self.assertNotIn(field, self.payload())

    def test_existing_region_fallback(self):
        minimax_api("Hello", None, verbose=False, region="saved-region")
        self.assertEqual(self.sent[-1].url, "https://api.minimax.io/v1/chat/completions")

    def test_history_and_multimodal_parts_are_preserved(self):
        history = [
            {"role": "system", "content": "Keep context."},
            {"role": "assistant", "content": None, "reasoning_content": "Prior reasoning",
             "tool_calls": [{"id": "call_1", "type": "function", "function": {
                 "name": "lookup", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "Result"},
        ]
        prompt = [
            {"type": "text", "text": "Describe these."},
            {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
            {"type": "video_url", "video_url": {"url": "mm_file://video"}},
        ]
        original_history, original_prompt = copy.deepcopy(history), copy.deepcopy(prompt)
        result = minimax_api(prompt, history, verbose=False)
        self.assertEqual(self.payload()["messages"], history + [{"role": "user", "content": prompt}])
        self.assertEqual(history, original_history)
        self.assertEqual(prompt, original_prompt)
        self.assertEqual(result["history"][:-1], self.payload()["messages"])

    def test_full_assistant_message_is_sent_on_next_turn(self):
        self.message = {"role": "assistant", "content": " <think>Keep this</think> Answer ",
                        "reasoning_content": " Preserve this reasoning verbatim. "}
        first = minimax_api("First", None, verbose=False)
        minimax_api("Second", first["history"], verbose=False)
        self.assertEqual(self.payload()["messages"][1], self.message)
        self.assertEqual(len(first["history"]), 2)
        self.assertEqual(len(self.payload()["messages"]), 3)

    def test_null_content_preserves_tool_calls(self):
        self.message = {"role": "assistant", "content": None, "reasoning_content": "Reasoning",
                        "tool_calls": [{"id": "call_2", "type": "function", "function": {
                            "name": "lookup", "arguments": "{}"}}]}
        result = minimax_api("First", None, verbose=False)
        self.assertEqual(result["responses"], "")
        self.assertEqual(result["history"][-1], self.message)

    def test_text_return_mode(self):
        self.assertEqual(minimax_api("Hello", None, verbose=False, return_dict=False), "Answer")

    def test_http_error_does_not_modify_history(self):
        self.status = 400
        history = [{"role": "user", "content": "Earlier"}]
        original = copy.deepcopy(history)
        with self.assertRaises(requests.HTTPError):
            minimax_api("Hello", history, verbose=False)
        self.assertEqual(history, original)


if __name__ == "__main__":
    unittest.main()
