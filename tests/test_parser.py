import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from model_doctor.parser import ParserError, parse_input


ARGS = """\
task: detect
model: yolo11n.pt
data: data.yaml
epochs: 2
batch: 16
imgsz: 640
optimizer: auto
lr0: 0.01
lrf: 0.01
weight_decay: 0.0005
patience: 20
seed: 0
mosaic: 1.0
mixup: 0.0
fliplr: 0.5
name: test-run
"""

DATA = """\
path: /datasets/helmet
train: images/train
val: images/val
names:
  0: person
  1: helmet
"""

RESULTS = """\
epoch,time,train/box_loss,train/cls_loss,train/dfl_loss,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B),val/box_loss,val/cls_loss,val/dfl_loss,lr/pg0
1,10.2,1.4,0.9,1.1,0.40,0.30,0.35,0.18,1.5,1.0,1.2,0.003
2,21.0,1.0,0.7,0.9,0.70,0.60,0.75,0.45,1.1,0.8,1.0,0.001
"""


class ParserTests(unittest.TestCase):
    def create_run(self, root: Path) -> Path:
        run = root / "train"
        (run / "weights").mkdir(parents=True)
        (run / "args.yaml").write_text(ARGS, encoding="utf-8")
        (run / "data.yaml").write_text(DATA, encoding="utf-8")
        (run / "results.csv").write_text(RESULTS, encoding="utf-8")
        (run / "confusion_matrix.png").write_bytes(b"fake-png")
        (run / "BoxPR_curve.png").write_bytes(b"fake-png")
        (run / "weights" / "best.pt").write_bytes(b"not-loaded")
        return run

    def test_parses_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            result = parse_input(self.create_run(Path(temp)))

        self.assertEqual(result["model"]["architecture"], "yolo11n")
        self.assertEqual(result["dataset"]["class_names"], ["person", "helmet"])
        self.assertEqual(result["dataset"]["config_reference"], "data.yaml")
        self.assertEqual(result["metrics"]["best_epoch"], 2)
        self.assertEqual(result["metrics"]["summary"]["map50_95"], 0.45)
        self.assertTrue(any(a["type"] == "pr_curve" for a in result["artifacts"]))
        self.assertFalse(
            next(a for a in result["artifacts"] if a["type"] == "checkpoint_best")[
                "available_for_ai"
            ]
        )

    def test_parses_zip_with_root_folder(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = self.create_run(root)
            archive = root / "result.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                for path in run.rglob("*"):
                    if path.is_file():
                        bundle.write(path, Path("uploaded-run") / path.relative_to(run))
            result = parse_input(archive)

        self.assertEqual(result["source"]["required_files"]["args_yaml"], "uploaded-run/args.yaml")
        self.assertEqual(len(result["metrics"]["epochs"]), 2)

    def test_rejects_missing_required_file(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "args.yaml").write_text(ARGS, encoding="utf-8")
            with self.assertRaisesRegex(ParserError, "results.csv"):
                parse_input(root)

    def test_continues_without_data_yaml_and_preserves_reference(self):
        with tempfile.TemporaryDirectory() as temp:
            run = self.create_run(Path(temp))
            (run / "data.yaml").unlink()
            result = parse_input(run)

        self.assertIsNone(result["dataset"]["config_path"])
        self.assertEqual(result["dataset"]["config_reference"], "data.yaml")
        self.assertEqual(result["dataset"]["class_names"], [])
        self.assertTrue(any("data.yaml" in warning for warning in result["parser"]["warnings"]))

    def test_rejects_non_detection_task(self):
        with tempfile.TemporaryDirectory() as temp:
            run = self.create_run(Path(temp))
            (run / "args.yaml").write_text(
                ARGS.replace("task: detect", "task: segment"), encoding="utf-8"
            )
            with self.assertRaisesRegex(ParserError, "supports detect only"):
                parse_input(run)

    def test_rejects_zip_path_traversal(self):
        with tempfile.TemporaryDirectory() as temp:
            archive = Path(temp) / "unsafe.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("../escape.txt", "unsafe")
            with self.assertRaisesRegex(ParserError, "Unsafe ZIP path"):
                parse_input(archive)

    def test_result_is_json_serializable(self):
        with tempfile.TemporaryDirectory() as temp:
            result = parse_input(self.create_run(Path(temp)))
        json.dumps(result)


if __name__ == "__main__":
    unittest.main()
