import json
import unittest
from pathlib import Path

from model_doctor.diagnosis import diagnose, load_rules


ROOT = Path(__file__).resolve().parents[1]


def evaluation(
    epochs: int,
    precision=0.7,
    recall=0.65,
    map_values=None,
    train_totals=None,
    validation_totals=None,
    epoch_numbers=None,
):
    if map_values is None:
        map_values = [0.30 + index * 0.01 for index in range(epochs)]
    if len(map_values) != epochs:
        raise ValueError("map_values length must match epochs")
    train_totals = train_totals or [3.0] * epochs
    validation_totals = validation_totals or [3.0] * epochs
    if len(train_totals) != epochs or len(validation_totals) != epochs:
        raise ValueError("loss sequence length must match epochs")
    epoch_numbers = epoch_numbers or list(range(1, epochs + 1))
    if len(epoch_numbers) != epochs:
        raise ValueError("epoch_numbers length must match epochs")
    rows = []
    for epoch in range(1, epochs + 1):
        train_loss = train_totals[epoch - 1] / 3
        validation_loss = validation_totals[epoch - 1] / 3
        rows.append(
            {
                "epoch": epoch_numbers[epoch - 1],
                "train": {
                    "box_loss": train_loss,
                    "cls_loss": train_loss,
                    "dfl_loss": train_loss,
                },
                "validation": {
                    "box_loss": validation_loss,
                    "cls_loss": validation_loss,
                    "dfl_loss": validation_loss,
                    "precision": precision,
                    "recall": recall,
                    "map50": 0.6,
                    "map50_95": map_values[epoch - 1],
                },
            }
        )
    return {
        "schema_version": "0.1.0",
        "metrics": {
            "summary": {"precision": precision, "recall": recall},
            "epochs": rows,
        },
    }


class DiagnosisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rules = load_rules(ROOT / "configs" / "diagnosis-rules.yaml")

    def test_short_history_is_insufficient(self):
        result = diagnose(evaluation(1), self.rules)
        item = result["diagnoses"][0]
        self.assertEqual(item["code"], "insufficient_training_history")
        self.assertEqual(item["title"], "학습 기록 부족")
        self.assertEqual(item["status"], "insufficient_data")
        self.assertEqual(item["status_label"], "판단 보류")
        self.assertEqual(result["summary"]["overall_status"], "insufficient_data")
        self.assertEqual(result["diagnoses"][1]["status"], "insufficient_data")

    def test_reliable_history_passes(self):
        result = diagnose(evaluation(10), self.rules)
        self.assertEqual(result["diagnoses"][0]["status"], "pass")

    def test_balanced_precision_recall_passes(self):
        result = diagnose(evaluation(10, precision=0.70, recall=0.65), self.rules)
        self.assertEqual(result["diagnoses"][1]["status"], "pass")
        self.assertEqual(result["diagnoses"][1]["status_label"], "이상 징후 없음")
        self.assertEqual(result["summary"]["overall_status"], "pass")

    def test_preliminary_precision_recall_has_lower_confidence(self):
        result = diagnose(evaluation(5, precision=0.70, recall=0.65), self.rules)
        item = result["diagnoses"][1]
        self.assertEqual(item["status"], "pass")
        self.assertEqual(item["confidence"], 0.6)
        self.assertIn("예비 진단", item["diagnosis"])

    def test_preliminary_imbalance_warning_has_lower_confidence(self):
        result = diagnose(evaluation(5, precision=0.82, recall=0.54), self.rules)
        item = result["diagnoses"][1]
        self.assertEqual(item["status"], "warning")
        self.assertEqual(item["confidence"], 0.6)

    def test_low_recall_creates_warning(self):
        result = diagnose(evaluation(10, precision=0.82, recall=0.54), self.rules)
        item = result["diagnoses"][1]
        self.assertEqual(item["status"], "warning")
        self.assertEqual(item["status_label"], "확인 필요")
        self.assertEqual(item["severity"], "medium")
        self.assertIn("놓치는", item["diagnosis"])
        self.assertEqual(result["summary"]["overall_status"], "warning")

    def test_missing_metric_is_insufficient(self):
        sample = evaluation(10)
        sample["metrics"]["summary"]["recall"] = None
        result = diagnose(sample, self.rules)
        self.assertEqual(result["diagnoses"][1]["status"], "insufficient_data")

    def test_high_gap_has_high_severity(self):
        result = diagnose(evaluation(10, precision=0.9, recall=0.5), self.rules)
        self.assertEqual(result["diagnoses"][1]["severity"], "high")

    def test_plateau_is_insufficient_below_ten_epochs(self):
        result = diagnose(evaluation(8), self.rules)
        item = result["diagnoses"][2]
        self.assertEqual(item["code"], "training_plateau")
        self.assertEqual(item["status"], "insufficient_data")

    def test_improving_map_passes_plateau_rule(self):
        values = [0.35, 0.37, 0.38, 0.39, 0.40, 0.405, 0.412, 0.419, 0.425, 0.43]
        result = diagnose(evaluation(10, map_values=values), self.rules)
        self.assertEqual(result["diagnoses"][2]["status"], "pass")

    def test_small_improvement_creates_plateau_warning(self):
        values = [0.410, 0.415, 0.419, 0.420, 0.418, 0.419, 0.421, 0.420, 0.423, 0.421]
        result = diagnose(evaluation(10, map_values=values), self.rules)
        item = result["diagnoses"][2]
        self.assertEqual(item["status"], "warning")
        self.assertEqual(item["severity"], "medium")

    def test_exact_plateau_threshold_passes(self):
        values = [0.40, 0.41, 0.415, 0.418, 0.420, 0.421, 0.422, 0.423, 0.424, 0.425]
        result = diagnose(evaluation(10, map_values=values), self.rules)
        self.assertEqual(result["diagnoses"][2]["status"], "pass")

    def test_declining_map_creates_plateau_warning(self):
        values = [0.42, 0.43, 0.44, 0.45, 0.44, 0.43, 0.42, 0.425, 0.43, 0.42]
        result = diagnose(evaluation(10, map_values=values), self.rules)
        self.assertEqual(result["diagnoses"][2]["status"], "warning")

    def test_missing_recent_map_is_insufficient(self):
        values = [0.40, 0.41, 0.42, 0.42, 0.43, 0.43, None, None, 0.431, None]
        result = diagnose(evaluation(10, map_values=values), self.rules)
        self.assertEqual(result["diagnoses"][2]["status"], "insufficient_data")

    def test_overfitting_is_insufficient_below_ten_epochs(self):
        result = diagnose(evaluation(8), self.rules)
        self.assertEqual(result["diagnoses"][3]["status"], "insufficient_data")

    def test_joint_train_and_validation_improvement_passes(self):
        train = [3.0] * 5 + [2.55] * 5
        validation = [3.2] * 5 + [2.9] * 5
        maps = [0.38, 0.39, 0.40, 0.39, 0.40, 0.41, 0.42, 0.43, 0.44, 0.43]
        result = diagnose(
            evaluation(10, map_values=maps, train_totals=train, validation_totals=validation),
            self.rules,
        )
        self.assertEqual(result["diagnoses"][3]["status"], "pass")

    def test_combined_signals_create_overfitting_warning(self):
        train = [3.0] * 5 + [2.4] * 5
        validation = [3.0] * 5 + [3.45] * 5
        maps = [0.48, 0.49, 0.50, 0.49, 0.50, 0.498, 0.501, 0.502, 0.500, 0.501]
        result = diagnose(
            evaluation(10, map_values=maps, train_totals=train, validation_totals=validation),
            self.rules,
        )
        item = result["diagnoses"][3]
        self.assertEqual(item["status"], "warning")
        self.assertEqual(item["severity"], "medium")

    def test_partial_overfitting_conditions_pass(self):
        train = [3.0] * 5 + [2.58] * 5
        validation = [3.0] * 5 + [3.18] * 5
        maps = [0.49, 0.50, 0.50, 0.49, 0.50, 0.501, 0.502, 0.503, 0.502, 0.501]
        result = diagnose(
            evaluation(10, map_values=maps, train_totals=train, validation_totals=validation),
            self.rules,
        )
        self.assertEqual(result["diagnoses"][3]["status"], "pass")

    def test_exact_overfitting_thresholds_warn(self):
        train = [3.0] * 5 + [2.7] * 5
        validation = [3.0] * 5 + [3.3] * 5
        maps = [0.49, 0.50, 0.50, 0.49, 0.50, 0.501, 0.502, 0.503, 0.505, 0.504]
        result = diagnose(
            evaluation(10, map_values=maps, train_totals=train, validation_totals=validation),
            self.rules,
        )
        self.assertEqual(result["diagnoses"][3]["status"], "warning")

    def test_missing_loss_is_insufficient_for_overfitting(self):
        sample = evaluation(10)
        sample["metrics"]["epochs"][8]["validation"]["dfl_loss"] = None
        result = diagnose(sample, self.rules)
        self.assertEqual(result["diagnoses"][3]["status"], "insufficient_data")

    def test_zero_previous_loss_is_insufficient_for_overfitting(self):
        train = [0.0] * 5 + [0.0] * 5
        result = diagnose(evaluation(10, train_totals=train), self.rules)
        self.assertEqual(result["diagnoses"][3]["status"], "insufficient_data")

    def test_declining_map_can_create_overfitting_warning(self):
        train = [3.0] * 5 + [2.46] * 5
        validation = [3.0] * 5 + [3.39] * 5
        maps = [0.43, 0.44, 0.45, 0.44, 0.45, 0.42, 0.43, 0.425, 0.42, 0.43]
        result = diagnose(
            evaluation(10, map_values=maps, train_totals=train, validation_totals=validation),
            self.rules,
        )
        self.assertEqual(result["diagnoses"][3]["status"], "warning")

    def test_post_peak_is_insufficient_below_ten_epochs(self):
        result = diagnose(evaluation(8), self.rules)
        self.assertEqual(result["diagnoses"][4]["status"], "insufficient_data")

    def test_final_epoch_peak_passes(self):
        maps = [0.30, 0.32, 0.35, 0.37, 0.39, 0.40, 0.42, 0.43, 0.44, 0.46]
        result = diagnose(evaluation(10, map_values=maps), self.rules)
        self.assertEqual(result["diagnoses"][4]["status"], "pass")

    def test_small_post_peak_drop_passes(self):
        maps = [0.38, 0.40, 0.42, 0.43, 0.44, 0.45, 0.447, 0.446, 0.445, 0.444]
        result = diagnose(evaluation(10, map_values=maps), self.rules)
        self.assertEqual(result["diagnoses"][4]["status"], "pass")

    def test_significant_post_peak_drop_warns(self):
        maps = [0.38, 0.40, 0.42, 0.43, 0.44, 0.45, 0.445, 0.44, 0.435, 0.43]
        result = diagnose(evaluation(10, map_values=maps), self.rules)
        item = result["diagnoses"][4]
        self.assertEqual(item["status"], "warning")
        self.assertEqual(item["severity"], "medium")

    def test_exact_post_peak_boundaries_warn(self):
        maps = [0.38, 0.40, 0.42, 0.43, 0.44, 0.445, 0.446, 0.45, 0.445, 0.44]
        result = diagnose(evaluation(10, map_values=maps), self.rules)
        self.assertEqual(result["diagnoses"][4]["status"], "warning")

    def test_recent_peak_has_insufficient_observation_period(self):
        maps = [0.38, 0.40, 0.42, 0.43, 0.44, 0.445, 0.446, 0.447, 0.45, 0.43]
        result = diagnose(evaluation(10, map_values=maps), self.rules)
        self.assertEqual(result["diagnoses"][4]["status"], "pass")

    def test_latest_duplicate_peak_is_used(self):
        maps = [0.40, 0.42, 0.43, 0.45, 0.44, 0.43, 0.44, 0.445, 0.45, 0.43]
        result = diagnose(evaluation(10, map_values=maps), self.rules)
        item = result["diagnoses"][4]
        self.assertEqual(item["status"], "pass")
        self.assertEqual(item["evidence"][0]["observed"], 1)

    def test_missing_map_is_insufficient_for_post_peak(self):
        maps = [0.40, 0.41, 0.42, 0.43, None, 0.44, 0.45, 0.44, 0.43, 0.42]
        result = diagnose(evaluation(10, map_values=maps), self.rules)
        self.assertEqual(result["diagnoses"][4]["status"], "insufficient_data")

    def test_nonconsecutive_epoch_numbers_use_record_count(self):
        maps = [0.40, 0.41, 0.42, 0.43, 0.44, 0.45, 0.445, 0.44, 0.435, 0.43]
        numbers = [1, 2, 3, 5, 6, 8, 9, 10, 12, 13]
        result = diagnose(
            evaluation(10, map_values=maps, epoch_numbers=numbers), self.rules
        )
        item = result["diagnoses"][4]
        self.assertEqual(item["status"], "warning")
        self.assertIn("epoch 8", item["observation"])
        self.assertEqual(item["evidence"][0]["observed"], 4)

    def test_report_is_json_serializable(self):
        json.dumps(diagnose(evaluation(1), self.rules))


if __name__ == "__main__":
    unittest.main()
