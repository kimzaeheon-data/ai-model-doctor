import json
import unittest
from pathlib import Path
from types import SimpleNamespace

from model_doctor.llm import LLMError, build_user_input, create_chat_response


ROOT = Path(__file__).resolve().parents[1]


class FakeResponses:
    def __init__(self, output):
        self.output = output
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(output_text=json.dumps(self.output, ensure_ascii=False))


class FakeClient:
    def __init__(self, output):
        self.responses = FakeResponses(output)


def load_json(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


class LLMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evaluation = load_json("artifacts/coco8-evaluation.json")
        cls.diagnosis = load_json("artifacts/coco8-diagnosis.json")
        cls.fixture = load_json("tests/fixtures/chat/summary.json")

    def test_build_user_input_serializes_untrusted_data(self):
        value = build_user_input(
            "전체 결과를 요약해줘",
            self.evaluation,
            self.diagnosis,
        )
        payload = json.loads(value.split("\n", 1)[1])
        self.assertEqual(payload["user_question"], "전체 결과를 요약해줘")
        self.assertEqual(payload["evaluation_json"], self.evaluation)
        self.assertEqual(payload["diagnosis_json"], self.diagnosis)
        self.assertEqual(payload["conversation_history"], [])

    def test_create_chat_response_uses_responses_structured_output(self):
        client = FakeClient(self.fixture)
        result = create_chat_response(
            "전체 결과를 요약해줘",
            self.evaluation,
            self.diagnosis,
            client=client,
        )

        self.assertEqual(result, self.fixture)
        request = client.responses.kwargs
        self.assertEqual(request["model"], "gpt-5.4-mini")
        self.assertEqual(request["reasoning"], {"effort": "low"})
        self.assertEqual(request["text"]["format"]["type"], "json_schema")
        self.assertIs(request["text"]["format"]["strict"], True)
        self.assertNotIn("tools", request)
        api_schema = request["text"]["format"]["schema"]
        self.assertNotIn("$schema", api_schema)
        self.assertEqual(api_schema["properties"]["schema_version"]["enum"], ["0.1.0"])
        evidence = api_schema["$defs"]["evidence"]
        self.assertIn("diagnosis_code", evidence["required"])

    def test_explicit_model_overrides_default(self):
        client = FakeClient(self.fixture)
        create_chat_response(
            "요약해줘",
            self.evaluation,
            self.diagnosis,
            client=client,
            model="test-model",
        )
        self.assertEqual(client.responses.kwargs["model"], "test-model")

    def test_invalid_response_is_rejected(self):
        invalid = dict(self.fixture)
        invalid.pop("limitations")
        with self.assertRaisesRegex(LLMError, "챗봇 응답 스키마"):
            create_chat_response(
                "요약해줘",
                self.evaluation,
                self.diagnosis,
                client=FakeClient(invalid),
            )

    def test_mismatched_schema_versions_are_rejected_before_request(self):
        diagnosis = dict(self.diagnosis)
        diagnosis["evaluation_schema_version"] = "9.9.9"
        with self.assertRaisesRegex(LLMError, "일치하지 않습니다"):
            create_chat_response(
                "요약해줘",
                self.evaluation,
                diagnosis,
                client=FakeClient(self.fixture),
            )

    def test_empty_question_is_rejected(self):
        with self.assertRaisesRegex(LLMError, "비어 있을 수 없습니다"):
            build_user_input("   ", self.evaluation, self.diagnosis)


if __name__ == "__main__":
    unittest.main()
