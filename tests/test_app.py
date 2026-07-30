import os
import unittest
from unittest.mock import Mock, patch


os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")

from app import app, parse_deepseek_result  # noqa: E402


class FlowCardsApiTests(unittest.TestCase):
    def setUp(self):
        app.config.update(TESTING=True)
        self.client = app.test_client()

    def test_home_page_loads(self):
        with self.client.get("/") as response:
            self.assertEqual(response.status_code, 200)
            self.assertIn(b"FlowCards", response.data)
            self.assertIn(b"Checking AI", response.data)

    def test_flashcard_answers_scroll_without_moving_rating_buttons(self):
        with self.client.get("/") as response:
            html = response.get_data(as_text=True)
            self.assertIn("height: clamp(340px, 62dvh, 520px)", html)
            self.assertIn(".fc-answer-content", html)
            self.assertIn("overflow-y: auto", html)
            self.assertIn("-webkit-overflow-scrolling: touch", html)
            self.assertIn(
                ".fc-rating {\n  display: flex; gap: var(--space-2);"
                " margin-top: var(--space-4);\n  justify-content: center;\n"
                "  flex: 0 0 auto;",
                html,
            )

    def test_ai_generation_requires_a_healthy_backend(self):
        with self.client.get("/") as response:
            html = response.get_data(as_text=True)
            self.assertIn("checkStudioAIHealth", html)
            self.assertIn("health?.configured", html)
            self.assertIn("There is intentionally no local fake-AI fallback", html)
            self.assertNotIn(
                "return localTopicGeneration(payload.prompt, payload.count)",
                html,
            )
            self.assertNotIn(
                "return localSourceGeneration(payload.sourceText, payload.count, payload.deckName)",
                html,
            )

    def test_rejects_short_topic(self):
        response = self.client.post(
            "/api/generate",
            json={
                "mode": "topic",
                "prompt": "short",
                "sourceText": "",
                "deckName": "Test",
                "count": 8,
            },
        )
        self.assertEqual(response.status_code, 400)

    def test_rejects_non_positive_card_count(self):
        response = self.client.post(
            "/api/generate",
            json={
                "mode": "topic",
                "prompt": "Create a deck about introductory biology.",
                "sourceText": "",
                "deckName": "Biology",
                "count": 0,
            },
        )
        self.assertEqual(response.status_code, 400)

    @patch("app.requests.post")
    def test_generates_cards_from_deepseek_json(self, mock_post):
        deepseek_response = Mock()
        deepseek_response.ok = True
        deepseek_response.status_code = 200
        deepseek_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"deckName":"Photosynthesis","summary":"A biology summary.",'
                            '"knowledgePoints":["Plants transform light energy."],'
                            '"cards":[{"question":"What is photosynthesis?",'
                            '"answer":"It converts light energy into chemical energy."}]}'
                        )
                    }
                }
            ]
        }
        mock_post.return_value = deepseek_response

        response = self.client.post(
            "/api/generate",
            json={
                "mode": "topic",
                "prompt": "Create beginner flashcards about photosynthesis.",
                "sourceText": "",
                "deckName": "Photosynthesis",
                "count": 25,
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body["deckName"], "Photosynthesis")
        self.assertEqual(len(body["cards"]), 1)
        self.assertEqual(body["cards"][0]["question"], "What is photosynthesis?")

        sent_request = mock_post.call_args.kwargs
        self.assertEqual(sent_request["json"]["model"], "deepseek-v4-pro")
        self.assertEqual(sent_request["json"]["response_format"], {"type": "json_object"})
        self.assertEqual(sent_request["json"]["thinking"], {"type": "disabled"})
        self.assertNotIn("reasoning_effort", sent_request["json"])
        self.assertNotIn("temperature", sent_request["json"])
        self.assertEqual(sent_request["json"]["max_tokens"], 3_000)
        self.assertNotIn("test-key", str(sent_request["json"]))
        system_prompt = sent_request["json"]["messages"][0]["content"]
        self.assertIn("genuine, grammatically complete question", system_prompt)
        self.assertIn("Return fewer cards instead of padding", system_prompt)
        self.assertIn("fun facts", system_prompt)

    @patch("app.time.sleep")
    @patch("app.requests.post")
    def test_falls_back_to_flash_when_pro_is_rate_limited(self, mock_post, mock_sleep):
        rate_limited_response = Mock()
        rate_limited_response.ok = False
        rate_limited_response.status_code = 429
        rate_limited_response.headers = {}

        successful_response = Mock()
        successful_response.ok = True
        successful_response.status_code = 200
        successful_response.headers = {}
        successful_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"deckName":"Cells","summary":"Core cell biology.",'
                            '"knowledgePoints":["Cells are the basic unit of life."],'
                            '"cards":[{"question":"What is the basic unit of life?",'
                            '"answer":"The cell."}]}'
                        )
                    }
                }
            ]
        }
        mock_post.side_effect = [rate_limited_response, successful_response]

        response = self.client.post(
            "/api/generate",
            json={
                "mode": "topic",
                "prompt": "Create beginner flashcards about cell biology.",
                "sourceText": "",
                "deckName": "Cells",
                "count": 10,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(mock_post.call_count, 2)
        self.assertEqual(
            mock_post.call_args_list[0].kwargs["json"]["model"],
            "deepseek-v4-pro",
        )
        self.assertEqual(
            mock_post.call_args_list[1].kwargs["json"]["model"],
            "deepseek-v4-flash",
        )
        self.assertEqual(
            response.headers["X-FlowCards-AI-Model"],
            "deepseek-v4-flash",
        )
        mock_sleep.assert_called_once()

    @patch("app.requests.post")
    def test_uses_flash_for_large_decks(self, mock_post):
        deepseek_response = Mock()
        deepseek_response.ok = True
        deepseek_response.status_code = 200
        deepseek_response.headers = {}
        deepseek_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"deckName":"Calculus","summary":"Core calculus ideas.",'
                            '"knowledgePoints":["Rates of change"],'
                            '"cards":[{"question":"What does a derivative measure?",'
                            '"answer":"An instantaneous rate of change."}]}'
                        )
                    }
                }
            ]
        }
        mock_post.return_value = deepseek_response

        response = self.client.post(
            "/api/generate",
            json={
                "mode": "topic",
                "prompt": "Create a comprehensive calculus review deck.",
                "sourceText": "",
                "deckName": "Calculus",
                "count": 50,
            },
        )

        self.assertEqual(response.status_code, 200)
        sent_payload = mock_post.call_args.kwargs["json"]
        # Large decks (>30) use Flash model
        self.assertEqual(sent_payload["model"], "deepseek-v4-flash")
        # Flash path disables thinking
        self.assertEqual(sent_payload["thinking"], {"type": "disabled"})
        self.assertNotIn("reasoning_effort", sent_payload)
        self.assertEqual(response.headers["X-FlowCards-AI-Model"], "deepseek-v4-flash")
        self.assertEqual(response.headers["X-FlowCards-AI-Thinking"], "disabled")

    def test_filters_trivia_statements_and_duplicate_questions(self):
        result = parse_deepseek_result(
            {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"deckName":"Biology","summary":"Core ideas.",'
                                '"knowledgePoints":["Energy conversion"],'
                                '"cards":['
                                '{"question":"Photosynthesis converts light energy.",'
                                '"answer":"This is a statement, not a question."},'
                                '{"question":"Did you know plants make sugar?",'
                                '"answer":"This is trivia framing."},'
                                '{"question":"What is photosynthesis?",'
                                '"answer":"It converts light energy into chemical energy."},'
                                '{"question":"What is photosynthesis?",'
                                '"answer":"Duplicate wording."},'
                                '{"question":"光合作用为什么重要",'
                                '"answer":"它把光能转化为化学能。"}'
                                ']}'
                            )
                        }
                    }
                ]
            },
            requested_count=5,
        )

        self.assertEqual(
            [card["question"] for card in result["cards"]],
            ["What is photosynthesis?", "光合作用为什么重要？"],
        )


    def test_estimate_endpoint_returns_count(self):
        with patch("app.requests.post") as mock_post:
            mock_response = Mock()
            mock_response.ok = True
            mock_response.json.return_value = {
                "choices": [{"message": {"content": '{"estimatedCount":42,"reasoning":"Medium scope."}'}}]
            }
            mock_post.return_value = mock_response

            response = self.client.post(
                "/api/estimate",
                json={"prompt": "comprehensive calculus review for final exam"},
            )

            self.assertEqual(response.status_code, 200)
            body = response.get_json()
            self.assertEqual(body["estimatedCount"], 42)
            self.assertIn("reasoning", body)

    def test_estimate_rejects_empty_payload(self):
        response = self.client.post("/api/estimate", json={})
        self.assertEqual(response.status_code, 400)

    def test_rate_limit_enforced(self):
        import app as app_module
        app_module._request_times.clear()

        for _ in range(11):
            response = self.client.post(
                "/api/generate",
                json={
                    "mode": "topic",
                    "prompt": "Create a test deck about math.",
                    "sourceText": "",
                    "deckName": "Math",
                    "count": 5,
                    "includeImages": False,
                },
            )
        # The 11th request should be rate limited
        self.assertEqual(response.status_code, 429)
        app_module._request_times.clear()

    @patch("app.requests.post")
    def test_include_images_flow(self, mock_post):
        deepseek_response = Mock()
        deepseek_response.ok = True
        deepseek_response.status_code = 200
        deepseek_response.json.return_value = {
            "choices": [{"message": {"content": (
                '{"deckName":"Heart","summary":"Heart anatomy.",'
                '"knowledgePoints":["Four chambers"],'
                '"cards":[{"question":"What are the heart chambers?",'
                '"answer":"RA, RV, LA, LV.","imageQuery":"human heart diagram labeled"}]}'
            )}}]
        }
        mock_post.return_value = deepseek_response

        with patch("app.search_wikimedia_image") as mock_search:
            mock_search.return_value = {
                "url": "https://upload.wikimedia.org/example.jpg",
                "sourceUrl": "https://commons.wikimedia.org/wiki/File:Example",
                "alt": "Heart diagram",
                "credit": "Public domain",
            }

            response = self.client.post(
                "/api/generate",
                json={
                    "mode": "topic",
                    "prompt": "human heart anatomy",
                    "sourceText": "",
                    "deckName": "Heart",
                    "count": 5,
                    "includeImages": True,
                },
            )

            self.assertEqual(response.status_code, 200)
            body = response.get_json()
            card = body["cards"][0]
            self.assertIn("image", card)
            self.assertEqual(card["image"]["url"], "https://upload.wikimedia.org/example.jpg")
            self.assertEqual(card["image"]["alt"], "Heart diagram")
            # Verify image search was called with the query
            mock_search.assert_called_once_with("human heart diagram labeled")

    @patch("app.requests.post")
    def test_images_not_included_when_toggle_off(self, mock_post):
        mock_response = Mock()
        mock_response.ok = True
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": (
                '{"deckName":"Test","summary":"T.","knowledgePoints":[],'
                                '"cards":[{"question":"What is the test asking?","answer":"This is the answer.","imageQuery":"test"}]}'
            )}}]
        }
        mock_post.return_value = mock_response

        response = self.client.post(
            "/api/generate",
            json={
                "mode": "topic",
                "prompt": "test topic for making flashcards",
                "sourceText": "",
                "deckName": "Test",
                "count": 3,
                "includeImages": False,
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertNotIn("image", body["cards"][0])

    def test_rejects_invalid_include_images_type(self):
        response = self.client.post(
            "/api/generate",
            json={
                "mode": "topic",
                "prompt": "test topic for flashcards",
                "sourceText": "",
                "deckName": "Test",
                "count": 5,
                "includeImages": "yes",
            },
        )
        self.assertEqual(response.status_code, 400)

    def test_parse_deepseek_result_with_image_queries(self):
        result = parse_deepseek_result(
            {
                "choices": [{"message": {"content": (
                    '{"deckName":"DNA","summary":"DNA structure.",'
                    '"knowledgePoints":["Double helix"],'
                    '"cards":['
                    '{"question":"What shape is DNA?","answer":"Double helix.","imageQuery":"DNA double helix diagram"},'
                    '{"question":"What pairs with adenine?","answer":"Thymine.","imageQuery":"AT base pair"}'
                    ']}'
                )}}]
            },
            requested_count=5,
        )
        self.assertEqual(len(result["cards"]), 2)
        self.assertEqual(result["cards"][0]["imageQuery"], "DNA double helix diagram")
        self.assertEqual(result["cards"][1]["imageQuery"], "AT base pair")


if __name__ == "__main__":
    unittest.main()
