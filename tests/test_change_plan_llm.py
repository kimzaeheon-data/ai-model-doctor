import copy
import json
import unittest
from pathlib import Path
from types import SimpleNamespace

from model_doctor.change_plan_llm import (
    ChangePlanLLMError,
    create_change_plan,
    validate_plan_against_inputs,
)


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


class ChangePlanLLMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = json.loads(
            (ROOT / "examples" / "change-plan.example.json").read_text(encoding="utf-8")
        )
        cls.evaluation = {
            "schema_version": "0.1.0",
            "source": {"run_name": "sample-overfit"},
            "training": {"patience": 100, "weight_decay": 0.0005},
        }
        cls.diagnosis = {
            "schema_version": "0.1.0",
            "evaluation_schema_version": "0.1.0",
            "diagnoses": [
                {"code": "suspected_overfitting", "status": "warning"},
                {"code": "post_peak_degradation", "status": "warning"},
                {"code": "training_plateau", "status": "pass"},
            ],
        }

    def test_create_plan_uses_structured_outputs_and_server_metadata(self):
        client = FakeClient(self.plan)
        result = create_change_plan(self.evaluation, self.diagnosis, client=client)
        self.assertTrue(result["plan_id"].startswith("plan-"))
        self.assertEqual(result["status"], "awaiting_approval")
        self.assertFalse(result["execution_allowed"])
        request = client.responses.kwargs
        self.assertEqual(request["model"], "gpt-5.4-mini")
        self.assertEqual(request["text"]["format"]["type"], "json_schema")
        self.assertNotIn("tools", request)

    def test_non_warning_diagnosis_cannot_support_change(self):
        plan = copy.deepcopy(self.plan)
        plan["changes"][0]["referenced_diagnoses"] = ["training_plateau"]
        with self.assertRaisesRegex(ChangePlanLLMError, "warning이 아닌 진단"):
            validate_plan_against_inputs(plan, self.evaluation, self.diagnosis)

    def test_before_value_must_match_evaluation(self):
        plan = copy.deepcopy(self.plan)
        plan["changes"][0]["before"] = 50
        with self.assertRaisesRegex(ChangePlanLLMError, "변경 전 값"):
            validate_plan_against_inputs(plan, self.evaluation, self.diagnosis)

    def test_no_warning_stops_before_api_request(self):
        diagnosis = copy.deepcopy(self.diagnosis)
        for item in diagnosis["diagnoses"]:
            item["status"] = "pass"
        client = FakeClient(self.plan)
        with self.assertRaisesRegex(ChangePlanLLMError, "warning 진단이 없어"):
            create_change_plan(self.evaluation, diagnosis, client=client)
        self.assertIsNone(client.responses.kwargs)


if __name__ == "__main__":
    unittest.main()
