import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "chat"
SCHEMA = ROOT / "schemas" / "chat-response.schema.json"


class ChatFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
        cls.required = set(cls.schema["required"])
        cls.allowed = set(cls.schema["properties"])

    def load(self, name):
        return json.loads((FIXTURES / name).read_text(encoding="utf-8"))

    def assert_contract(self, response):
        self.assertFalse(self.required - set(response))
        self.assertFalse(set(response) - self.allowed)
        self.assertEqual(response["schema_version"], "0.1.0")
        self.assertTrue(response["message"])
        self.assertTrue(response["referenced_diagnoses"])
        for evidence in response["evidence"]:
            self.assertRegex(evidence["comparison"], "[가-힣]")
        for action in response["suggested_actions"]:
            self.assertIs(action["requires_approval"], False)

    def test_all_fixtures_follow_v01_contract(self):
        fixture_paths = sorted(FIXTURES.glob("*.json"))
        self.assertEqual(len(fixture_paths), 3)
        for fixture_path in fixture_paths:
            with self.subTest(fixture=fixture_path.name):
                self.assert_contract(
                    json.loads(fixture_path.read_text(encoding="utf-8"))
                )

    def test_summary_references_all_diagnoses(self):
        response = self.load("summary.json")
        self.assertEqual(response["answer_type"], "summary")
        self.assertEqual(len(response["referenced_diagnoses"]), 5)
        self.assertIn("판단 보류", response["message"])

    def test_overfitting_answer_does_not_claim_diagnosis(self):
        response = self.load("overfitting-question.json")
        self.assertEqual(response["answer_type"], "explanation")
        self.assertIn("판단할 수 없습니다", response["message"])
        self.assertNotIn("과적합됐습니다", response["message"])

    def test_settings_request_does_not_create_mutating_action(self):
        response = self.load("settings-change-request.json")
        self.assertEqual(response["answer_type"], "follow_up")
        self.assertIn("권한이 없습니다", response["message"])
        action_types = {action["type"] for action in response["suggested_actions"]}
        self.assertNotIn("change_training_config", action_types)
        self.assertNotIn("run_training", action_types)


if __name__ == "__main__":
    unittest.main()
