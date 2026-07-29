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
            self.assertIn(b"DeepSeek V4 Flash", response.data)

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
        self.assertEqual(sent_request["json"]["model"], "deepseek-v4-flash")
        self.assertEqual(sent_request["json"]["response_format"], {"type": "json_object"})
        self.assertEqual(sent_request["json"]["temperature"], 0.2)
        self.assertEqual(sent_request["json"]["max_tokens"], 6_000)
        self.assertNotIn("test-key", str(sent_request["json"]))
        system_prompt = sent_request["json"]["messages"][0]["content"]
        self.assertIn("genuine, grammatically complete question", system_prompt)
        self.assertIn("Return fewer cards instead of padding", system_prompt)
        self.assertIn("fun facts", system_prompt)

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


if __name__ == "__main__":
    unittest.main()
