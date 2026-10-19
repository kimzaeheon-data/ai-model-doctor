import copy
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import yaml

from model_doctor.change_plan import create_approval
from model_doctor.executor import (
    ExecutionError,
    build_training_command,
    prepare_candidate,
    run_candidate_training,
)


ROOT = Path(__file__).resolve().parents[1]


class ExecutorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = json.loads(
            (ROOT / "examples" / "change-plan.example.json").read_text(encoding="utf-8")
        )

    def source_config(self, root):
        path = Path(root) / "args.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "model": "yolo11n.pt",
                    "data": "data.yaml",
                    "patience": 100,
                    "weight_decay": 0.0005,
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        return path

    def prepare(self, root, selected=None):
        if selected is None:
            approval = create_approval(self.plan, "approve_all")
        else:
            approval = create_approval(
                self.plan, "approve_selected", selected_change_ids=selected
            )
        source = self.source_config(root)
        return prepare_candidate(
            self.plan,
            approval,
            source_training=source,
            output_root=Path(root) / "candidates",
            execution_id="execution-123456789abc",
        )

    def test_prepare_candidate_changes_only_approved_values(self):
        with tempfile.TemporaryDirectory() as temp:
            record, candidate = self.prepare(temp, ["reduce-patience"])
            result = yaml.safe_load(
                (candidate / "candidate" / "args.yaml").read_text(encoding="utf-8")
            )
            original = yaml.safe_load((Path(temp) / "args.yaml").read_text(encoding="utf-8"))
            self.assertEqual(result["patience"], 30)
            self.assertEqual(result["weight_decay"], 0.0005)
            self.assertEqual(original["patience"], 100)
            self.assertEqual(record["applied_change_ids"], ["reduce-patience"])

    def test_candidate_contains_audit_records(self):
        with tempfile.TemporaryDirectory() as temp:
            _, candidate = self.prepare(temp)
            self.assertTrue((candidate / "audit" / "change-plan.json").is_file())
            self.assertTrue((candidate / "audit" / "approval.json").is_file())
            self.assertTrue((candidate / "audit" / "execution.json").is_file())

    def test_unapproved_plan_cannot_prepare_candidate(self):
        with tempfile.TemporaryDirectory() as temp:
            approval = create_approval(self.plan, "cancel")
            with self.assertRaisesRegex(ExecutionError, "승인된 변경이 없어"):
                prepare_candidate(
                    self.plan,
                    approval,
                    source_training=self.source_config(temp),
                    output_root=Path(temp) / "candidates",
                )

    def test_tampered_plan_cannot_prepare_candidate(self):
        with tempfile.TemporaryDirectory() as temp:
            approval = create_approval(self.plan, "approve_all")
            changed = copy.deepcopy(self.plan)
            changed["changes"][0]["after"] = 40
            with self.assertRaisesRegex(ExecutionError, "승인 게이트"):
                prepare_candidate(
                    changed,
                    approval,
                    source_training=self.source_config(temp),
                    output_root=Path(temp) / "candidates",
                )

    def test_training_command_uses_candidate_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            _, candidate = self.prepare(temp)
            command = build_training_command(candidate)
            self.assertEqual(command[:3], ["yolo", "detect", "train"])
            self.assertIn(f"cfg={candidate / 'candidate' / 'args.yaml'}", command)
            self.assertIn(f"project={candidate / 'runs'}", command)

    def test_successful_training_is_recorded(self):
        with tempfile.TemporaryDirectory() as temp:
            _, candidate = self.prepare(temp)

            def runner(*args, **kwargs):
                checkpoint = Path(kwargs["cwd"]) / "runs" / "train" / "weights" / "best.pt"
                checkpoint.parent.mkdir(parents=True)
                checkpoint.write_bytes(b"candidate checkpoint")
                return SimpleNamespace(returncode=0)

            record = run_candidate_training(
                candidate, timeout_seconds=30, runner=runner
            )
            self.assertEqual(record["status"], "completed")
            self.assertEqual(record["training"]["return_code"], 0)

    def test_missing_checkpoint_marks_training_failed(self):
        with tempfile.TemporaryDirectory() as temp:
            _, candidate = self.prepare(temp)

            def runner(*args, **kwargs):
                return SimpleNamespace(returncode=0)

            record = run_candidate_training(
                candidate, timeout_seconds=30, runner=runner
            )
            self.assertEqual(record["status"], "failed")
            self.assertIn("체크포인트", record["error"])

    def test_failed_training_preserves_candidate_and_log(self):
        with tempfile.TemporaryDirectory() as temp:
            _, candidate = self.prepare(temp)

            def runner(*args, **kwargs):
                kwargs["stdout"].write("training failed\n")
                return SimpleNamespace(returncode=2)

            record = run_candidate_training(
                candidate, timeout_seconds=30, runner=runner
            )
            self.assertEqual(record["status"], "failed")
            self.assertTrue(candidate.is_dir())
            self.assertIn(
                "training failed",
                (candidate / "logs" / "training.log").read_text(encoding="utf-8"),
            )

    def test_training_timeout_is_recorded(self):
        with tempfile.TemporaryDirectory() as temp:
            _, candidate = self.prepare(temp)

            def runner(*args, **kwargs):
                raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

            record = run_candidate_training(
                candidate, timeout_seconds=10, runner=runner
            )
            self.assertEqual(record["status"], "timed_out")
            self.assertIn("10초", record["error"])

    def test_existing_candidate_directory_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            approval = create_approval(self.plan, "approve_all")
            source = self.source_config(temp)
            output = Path(temp) / "candidates"
            (output / "execution-123456789abc").mkdir(parents=True)
            with self.assertRaisesRegex(ExecutionError, "이미 존재"):
                prepare_candidate(
                    self.plan,
                    approval,
                    source_training=source,
                    output_root=output,
                    execution_id="execution-123456789abc",
                )


if __name__ == "__main__":
    unittest.main()
