from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
import re
import stat
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

PARSER_VERSION = "0.1.0"
MAX_ARCHIVE_FILES = 2_000
MAX_MEMBER_BYTES = 100 * 1024 * 1024
MAX_EXTRACTED_BYTES = 500 * 1024 * 1024


class ParserError(ValueError):
    """Raised when an uploaded YOLO result cannot be parsed safely."""


def _scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return None
    lowered = value.lower()
    if lowered in {"null", "none", "~"}:
        return None
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value.strip("'\"")


def _read_simple_yaml(path: Path) -> tuple[dict[str, Any], list[str]]:
    """Read the conservative YAML subset emitted by common YOLO runs.

    Top-level scalar values are supported everywhere. ``names`` additionally
    supports an indented list or integer-keyed mapping from data.yaml.
    """
    result: dict[str, Any] = {}
    warnings: list[str] = []
    active_key: str | None = None
    nested: list[Any] | dict[int, Any] | None = None

    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip())
        stripped = raw_line.strip()

        if indent and active_key == "names":
            if stripped.startswith("-"):
                if not isinstance(nested, list):
                    nested = []
                nested.append(_scalar(stripped[1:]))
                result[active_key] = nested
                continue
            match = re.match(r"^(\d+)\s*:\s*(.+)$", stripped)
            if match:
                if not isinstance(nested, dict):
                    nested = {}
                nested[int(match.group(1))] = _scalar(match.group(2))
                result[active_key] = nested
                continue

        if indent:
            warnings.append(f"Ignored unsupported nested YAML at {path.name}:{line_number}")
            continue

        if ":" not in stripped:
            warnings.append(f"Ignored malformed YAML at {path.name}:{line_number}")
            continue
        key, value = stripped.split(":", 1)
        key = key.strip()
        active_key = key
        nested = None
        if value.strip():
            result[key] = _scalar(value)
        elif key == "names":
            nested = []
            result[key] = nested
        else:
            result[key] = None
    return result, warnings


def _safe_float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _safe_int(value: Any, default: int | None = None) -> int | None:
    number = _safe_float(value)
    return int(number) if number is not None else default


def _column(row: dict[str, str], *names: str) -> float | None:
    normalized = {key.strip(): value for key, value in row.items() if key is not None}
    for name in names:
        if name in normalized:
            return _safe_float(normalized[name])
    return None


def _score(value: float | None, warnings: list[str], label: str) -> float | None:
    if value is None:
        return None
    if 0 <= value <= 1:
        return value
    warnings.append(f"Ignored out-of-range score for {label}: {value}")
    return None


def _loss(value: float | None, warnings: list[str], label: str) -> float | None:
    if value is None or value >= 0:
        return value
    warnings.append(f"Ignored negative loss for {label}: {value}")
    return None


def _read_results(path: Path, warnings: list[str]) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ParserError("results.csv does not contain any epoch rows")

    epochs: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        epoch = _safe_int(row.get("epoch"), index)
        epochs.append(
            {
                "epoch": epoch,
                "elapsed_seconds": _column(row, "time", "elapsed_seconds"),
                "train": {
                    "box_loss": _loss(_column(row, "train/box_loss"), warnings, "train/box_loss"),
                    "cls_loss": _loss(_column(row, "train/cls_loss"), warnings, "train/cls_loss"),
                    "dfl_loss": _loss(_column(row, "train/dfl_loss"), warnings, "train/dfl_loss"),
                },
                "validation": {
                    "box_loss": _loss(_column(row, "val/box_loss"), warnings, "val/box_loss"),
                    "cls_loss": _loss(_column(row, "val/cls_loss"), warnings, "val/cls_loss"),
                    "dfl_loss": _loss(_column(row, "val/dfl_loss"), warnings, "val/dfl_loss"),
                    "precision": _score(
                        _column(row, "metrics/precision(B)", "metrics/precision"), warnings, "precision"
                    ),
                    "recall": _score(
                        _column(row, "metrics/recall(B)", "metrics/recall"), warnings, "recall"
                    ),
                    "map50": _score(
                        _column(row, "metrics/mAP50(B)", "metrics/mAP50"), warnings, "mAP50"
                    ),
                    "map50_95": _score(
                        _column(row, "metrics/mAP50-95(B)", "metrics/mAP50-95"),
                        warnings,
                        "mAP50-95",
                    ),
                },
                "learning_rate": _column(row, "lr/pg0", "learning_rate"),
            }
        )
    return epochs


def _best_epoch(epochs: list[dict[str, Any]]) -> dict[str, Any]:
    def ranking(epoch: dict[str, Any]) -> tuple[float, float]:
        validation = epoch["validation"]
        return (validation["map50_95"] or -1.0, validation["map50"] or -1.0)

    return max(epochs, key=ranking)


def _find_unique(root: Path, filename: str, required: bool) -> Path | None:
    matches = [path for path in root.rglob(filename) if path.is_file()]
    if not matches:
        if required:
            raise ParserError(f"Required file is missing: {filename}")
        return None
    if len(matches) > 1:
        raise ParserError(f"Multiple {filename} files found; upload one YOLO run at a time")
    return matches[0]


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _extract_zip_safely(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        if len(members) > MAX_ARCHIVE_FILES:
            raise ParserError(f"ZIP contains more than {MAX_ARCHIVE_FILES} files")
        total = 0
        for member in members:
            member_path = Path(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ParserError(f"Unsafe ZIP path: {member.filename}")
            mode = member.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ParserError(f"Symbolic links are not allowed: {member.filename}")
            if member.file_size > MAX_MEMBER_BYTES:
                raise ParserError(f"ZIP member is too large: {member.filename}")
            total += member.file_size
            if total > MAX_EXTRACTED_BYTES:
                raise ParserError("ZIP expands beyond the allowed total size")
        bundle.extractall(destination)


def _class_names(data: dict[str, Any]) -> list[str]:
    names = data.get("names", [])
    if isinstance(names, dict):
        return [str(names[key]) for key in sorted(names)]
    if isinstance(names, (list, tuple)):
        return [str(name) for name in names]
    return []


def _artifact_entries(root: Path) -> list[dict[str, Any]]:
    recognized = {
        "confusion_matrix.png": ("confusion_matrix", "image/png", True),
        "confusion_matrix_normalized.png": ("confusion_matrix_normalized", "image/png", True),
        "PR_curve.png": ("pr_curve", "image/png", True),
        "BoxPR_curve.png": ("pr_curve", "image/png", True),
        "F1_curve.png": ("f1_curve", "image/png", True),
        "BoxF1_curve.png": ("f1_curve", "image/png", True),
        "P_curve.png": ("precision_curve", "image/png", True),
        "BoxP_curve.png": ("precision_curve", "image/png", True),
        "R_curve.png": ("recall_curve", "image/png", True),
        "BoxR_curve.png": ("recall_curve", "image/png", True),
        "results.png": ("results_plot", "image/png", True),
        "best.pt": ("checkpoint_best", "application/octet-stream", False),
        "last.pt": ("checkpoint_last", "application/octet-stream", False),
    }
    artifacts: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name in recognized:
            artifact_type, media_type, available = recognized[path.name]
            artifacts.append(
                {
                    "type": artifact_type,
                    "path": _relative(path, root),
                    "media_type": media_type,
                    "available_for_ai": available,
                }
            )
    return artifacts


def parse_directory(root: Path) -> dict[str, Any]:
    root = root.resolve()
    args_path = _find_unique(root, "args.yaml", required=True)
    results_path = _find_unique(root, "results.csv", required=True)
    data_path = _find_unique(root, "data.yaml", required=False)
    assert args_path is not None and results_path is not None

    args, warnings = _read_simple_yaml(args_path)
    task = str(args.get("task") or "detect")
    if task != "detect":
        raise ParserError(
            f"Unsupported YOLO task: {task}. MVP currently supports detect only"
        )
    data: dict[str, Any] = {}
    if data_path:
        data, data_warnings = _read_simple_yaml(data_path)
        warnings.extend(data_warnings)
    else:
        warnings.append("data.yaml was not included; class and dataset analysis is limited.")

    epochs = _read_results(results_path, warnings)
    best = _best_epoch(epochs)
    artifacts = _artifact_entries(root)
    if any(item["type"].startswith("checkpoint_") for item in artifacts):
        warnings.append("Checkpoint was indexed but not loaded for security.")

    model_value = str(args.get("model") or "unknown")
    architecture = Path(model_value).stem or "unknown"
    names = _class_names(data)
    optional_paths = [
        _relative(path, root)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path not in {args_path, results_path}
    ]
    validation = best["validation"]

    return {
        "schema_version": "0.1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "format": "ultralytics-yolo-result",
            "run_name": args.get("name"),
            "ultralytics_version": args.get("version"),
            "required_files": {
                "args_yaml": _relative(args_path, root),
                "results_csv": _relative(results_path, root),
            },
            "optional_files_found": optional_paths,
        },
        "model": {
            "framework": "ultralytics",
            "task": "detect",
            "architecture": architecture,
            "pretrained_checkpoint": model_value,
            "best_checkpoint": next(
                (item["path"] for item in artifacts if item["type"] == "checkpoint_best"), None
            ),
        },
        "training": {
            "epochs_configured": _safe_int(args.get("epochs"), len(epochs)) or len(epochs),
            "epochs_completed": len(epochs),
            "batch": _safe_int(args.get("batch"), 1) or 1,
            "imgsz": _safe_int(args.get("imgsz"), 640) or 640,
            "device": None if args.get("device") is None else str(args.get("device")),
            "optimizer": None if args.get("optimizer") is None else str(args.get("optimizer")),
            "learning_rate_initial": _safe_float(args.get("lr0")),
            "learning_rate_final_ratio": _safe_float(args.get("lrf")),
            "weight_decay": _safe_float(args.get("weight_decay")),
            "patience": _safe_int(args.get("patience")),
            "seed": _safe_int(args.get("seed")),
            "augmentation": {
                key: args[key]
                for key in ("mosaic", "mixup", "fliplr", "flipud", "degrees", "translate", "scale")
                if key in args
            },
        },
        "dataset": {
            "config_path": _relative(data_path, root) if data_path else None,
            "config_reference": None if args.get("data") is None else str(args.get("data")),
            "class_names": names,
            "train_image_count": None,
            "validation_image_count": None,
            "class_distribution": None,
            "bounding_box_size_distribution": None,
        },
        "metrics": {
            "best_epoch": best["epoch"],
            "summary": {
                "precision": validation["precision"],
                "recall": validation["recall"],
                "map50": validation["map50"],
                "map50_95": validation["map50_95"],
                "fitness": None,
            },
            "per_class": None,
            "epochs": epochs,
        },
        "artifacts": artifacts,
        "parser": {
            "name": "ai-model-doctor-yolo-parser",
            "version": PARSER_VERSION,
            "warnings": list(dict.fromkeys(warnings)),
        },
    }


def parse_input(source: str | os.PathLike[str]) -> dict[str, Any]:
    source_path = Path(source)
    if source_path.is_dir():
        return parse_directory(source_path)
    if source_path.is_file() and zipfile.is_zipfile(source_path):
        with tempfile.TemporaryDirectory(prefix="model-doctor-") as temp_dir:
            root = Path(temp_dir)
            _extract_zip_safely(source_path, root)
            return parse_directory(root)
    raise ParserError("Input must be a YOLO result directory or ZIP file")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert an Ultralytics YOLO result directory or ZIP into evaluation.json"
    )
    parser.add_argument("input", help="Path to a YOLO result directory or ZIP")
    parser.add_argument("-o", "--output", default="evaluation.json", help="Output JSON path")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    arguments = build_argument_parser().parse_args(argv)
    try:
        result = parse_input(arguments.input)
        output = Path(arguments.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (OSError, ParserError, zipfile.BadZipFile) as error:
        print(f"error: {error}", file=os.sys.stderr)
        return 1
    print(f"Created {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
