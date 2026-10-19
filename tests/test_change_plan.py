import copy
import json
import unittest
from pathlib import Path

from model_doctor.change_plan import (
    ChangePlanError,
    authorized_changes,
    create_approval,
    load_policy,
    plan_digest,
    validate_change_plan,
)


ROOT = Path(__file__).resolve().parents[1]


class ChangePlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = json.loads(
            (ROOT / "examples" / "change-plan.example.json").read_text(encoding="utf-8")
        )
        cls.policy = load_policy(ROOT / "configs" / "change-policy.yaml")

    def test_example_plan_is_valid(self):
        validate_change_plan(self.plan, policy=self.policy)

    def test_approve_all_returns_every_change(self):
        approval = create_approval(self.plan, "approve_all")
        result = authorized_changes(self.plan, approval, policy=self.policy)
        self.assertEqual(
            [item["change_id"] for item in result],
            ["reduce-patience", "increase-weight-decay"],
        )

    def test_partial_approval_returns_only_selected_change(self):
        approval = create_approval(
            self.plan,
            "approve_selected",
            selected_change_ids=["reduce-patience"],
        )
        result = authorized_changes(self.plan, approval, policy=self.policy)
        self.assertEqual([item["change_id"] for item in result], ["reduce-patience"])

    def test_revision_request_does_not_authorize_execution(self):
        approval = create_approval(
            self.plan,
            "request_revision",
            requested_revisions=["patience 값을 50으로 다시 검토해 주세요."],
        )
        self.assertFalse(approval["execution_authorized"])
        self.assertEqual(authorized_changes(self.plan, approval), [])

    def test_cancel_does_not_authorize_execution(self):
        approval = create_approval(self.plan, "cancel")
        self.assertEqual(authorized_changes(self.plan, approval), [])

    def test_unknown_parameter_is_rejected(self):
        plan = copy.deepcopy(self.plan)
        plan["changes"][0]["parameter"] = "training.model"
        with self.assertRaisesRegex(ChangePlanError, "허용되지 않은 변경 파라미터"):
            validate_change_plan(plan, policy=self.policy)

    def test_wrong_target_file_is_rejected(self):
        plan = copy.deepcopy(self.plan)
        plan["changes"][0]["target_file"] = "candidate/inference.yaml"
        with self.assertRaisesRegex(ChangePlanError, "대상 파일"):
            validate_change_plan(plan, policy=self.policy)

    def test_out_of_range_value_is_rejected(self):
        plan = copy.deepcopy(self.plan)
        plan["changes"][1]["after"] = 0.5
        with self.assertRaisesRegex(ChangePlanError, "허용 최댓값"):
            validate_change_plan(plan, policy=self.policy)

    def test_unchanged_value_is_rejected(self):
        plan = copy.deepcopy(self.plan)
        plan["changes"][0]["after"] = plan["changes"][0]["before"]
        with self.assertRaisesRegex(ChangePlanError, "변경 전후 값이 같습니다"):
            validate_change_plan(plan, policy=self.policy)

    def test_unknown_selected_change_is_rejected(self):
        with self.assertRaisesRegex(ChangePlanError, "계획에 없는 change_id"):
            create_approval(
                self.plan,
                "approve_selected",
                selected_change_ids=["unknown-change"],
            )

    def test_approval_and_revision_cannot_be_combined(self):
        with self.assertRaisesRegex(ChangePlanError, "함께 기록할 수 없습니다"):
            create_approval(
                self.plan,
                "approve_all",
                requested_revisions=["값을 다시 검토해 주세요."],
            )

    def test_plan_tampering_invalidates_approval(self):
        approval = create_approval(self.plan, "approve_all")
        changed = copy.deepcopy(self.plan)
        changed["changes"][0]["after"] = 40
        with self.assertRaisesRegex(ChangePlanError, "기존 승인이 무효"):
            authorized_changes(changed, approval, policy=self.policy)

    def test_digest_is_independent_of_dictionary_key_order(self):
        reversed_plan = dict(reversed(list(self.plan.items())))
        self.assertEqual(plan_digest(self.plan), plan_digest(reversed_plan))


if __name__ == "__main__":
    unittest.main()
