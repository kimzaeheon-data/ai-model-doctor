from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable
from uuid import uuid4

import yaml
from jsonschema import Draft202012Validator

from .change_plan import (
    ChangePlanError,
    authorized_changes,
    load_json,
    load_policy,
    plan_digest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXECUTION_SCHEMA_PATH = PROJECT_ROOT / "schemas" / "execution.schema.json"


class ExecutionError(RuntimeError):
    """후보 실험 준비 또는 실행이 안전하게 완료되지 못하면 발생합니다."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_yaml(path: Path, label: str) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ExecutionError(f"{label} YAML을 읽을 수 없습니다. 세부 정보: {error}") from error
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ExecutionError(f"{label} YAML의 최상위 값은 객체여야 합니다.")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _validate_execution(record: dict[str, Any]) -> None:
    schema = json.loads(DEFAULT_EXECUTION_SCHEMA_PATH.read_text(encoding="utf-8"))
    errors = sorted(
        Draft202012Validator(schema).iter_errors(record),
        key=lambda error: list(error.absolute_path),
    )
    if errors:
        location = ".".join(str(item) for item in errors[0].absolute_path) or "root"
        raise ExecutionError(f"실행 기록이 {location} 위치에서 스키마를 충족하지 않습니다.")


def _set_nested(config: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    target = config
    for part in parts[:-1]:
        current = target.get(part)
        if current is None:
            current = {}
            target[part] = current
        if not isinstance(current, dict):
            raise ExecutionError(f"설정 경로를 객체로 확장할 수 없습니다: {path}")
        target = current
    target[parts[-1]] = value


def _safe_candidate_directory(output_root: Path, execution_id: str) -> Path:
    root = output_root.resolve()
    candidate = (root / execution_id).resolve()
    if candidate.parent != root:
        raise ExecutionError("후보 실험 경로가 지정한 출력 디렉터리를 벗어났습니다.")
    if candidate.exists():
        raise ExecutionError(f"후보 실험 디렉터리가 이미 존재합니다: {candidate}")
    return candidate


def prepare_candidate(
    plan: dict[str, Any],
    approval: dict[str, Any],
    *,
    source_training: str | Path | None,
    source_inference: str | Path | None = None,
    output_root: str | Path = "candidate-runs",
    execution_id: str | None = None,
) -> tuple[dict[str, Any], Path]:
    try:
        approved = authorized_changes(plan, approval)
    except ChangePlanError as error:
        raise ExecutionError(f"승인 게이트를 통과하지 못했습니다. 세부 정보: {error}") from error
    if not approved:
        raise ExecutionError("승인된 변경이 없어 후보 실험을 준비할 수 없습니다.")

    policy = load_policy()
    identifier = execution_id or f"execution-{uuid4().hex[:12]}"
    if not identifier.startswith("execution-") or len(identifier) != 22:
        raise ExecutionError("execution_id 형식이 올바르지 않습니다.")
    candidate_dir = _safe_candidate_directory(Path(output_root), identifier)

    source_paths = {
        "training": Path(source_training).resolve() if source_training else None,
        "inference": Path(source_inference).resolve() if source_inference else None,
    }
    categories = {change["category"] for change in approved}
    for category in categories:
        source = source_paths[category]
        if source is None or not source.is_file():
            raise ExecutionError(f"{category} 변경에 필요한 원본 설정 파일이 없습니다.")

    candidate_dir.mkdir(parents=True)
    try:
        configs: dict[str, dict[str, Any]] = {}
        candidate_files: list[str] = []
        for category in categories:
            source = source_paths[category]
            assert source is not None
            configs[category] = _read_yaml(source, f"원본 {category} 설정")

        for change in approved:
            category = change["category"]
            rules = policy["allowed_targets"][category]["parameters"][change["parameter"]]
            _set_nested(configs[category], str(rules["yaml_key"]), change["after"])

        for category in sorted(categories):
            relative = Path(policy["allowed_targets"][category]["file"])
            destination = candidate_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(
                yaml.safe_dump(configs[category], allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            candidate_files.append(relative.as_posix())

        audit_dir = candidate_dir / "audit"
        audit_dir.mkdir()
        _write_json(audit_dir / "change-plan.json", plan)
        _write_json(audit_dir / "approval.json", approval)

        created = _now()
        record = {
            "schema_version": "0.1.0",
            "execution_id": identifier,
            "plan_id": plan["plan_id"],
            "approval_id": approval["approval_id"],
            "plan_digest": plan_digest(plan),
            "created_at": created,
            "updated_at": created,
            "status": "prepared",
            "candidate_directory": str(candidate_dir),
            "source_files": {
                key: str(value) if value is not None else None
                for key, value in source_paths.items()
            },
            "candidate_files": candidate_files,
            "applied_change_ids": [change["change_id"] for change in approved],
            "training": {
                "command": [],
                "timeout_seconds": 1,
                "return_code": None,
                "log_path": str(candidate_dir / "logs" / "training.log"),
                "checkpoint_directory": str(candidate_dir / "runs" / "train" / "weights"),
            },
            "error": None,
        }
        _validate_execution(record)
        _write_json(audit_dir / "execution.json", record)
        return record, candidate_dir
    except Exception:
        shutil.rmtree(candidate_dir, ignore_errors=True)
        raise


def build_training_command(
    candidate_dir: str | Path, *, executable: str = "yolo"
) -> list[str]:
    root = Path(candidate_dir).resolve()
    config = root / "candidate" / "args.yaml"
    if not config.is_file():
        raise ExecutionError("후보 학습 설정 파일이 없습니다.")
    return [
        executable,
        "detect",
        "train",
        f"cfg={config}",
        f"project={root / 'runs'}",
        "name=train",
        "exist_ok=false",
    ]


def run_candidate_training(
    candidate_dir: str | Path,
    *,
    timeout_seconds: int,
    executable: str = "yolo",
    runner: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
) -> dict[str, Any]:
    if timeout_seconds < 1:
        raise ExecutionError("학습 제한 시간은 1초 이상이어야 합니다.")
    root = Path(candidate_dir).resolve()
    record_path = root / "audit" / "execution.json"
    record = load_json(record_path, "실행 기록")
    if record["status"] != "prepared":
        raise ExecutionError("prepared 상태의 후보 실험만 학습을 시작할 수 있습니다.")

    command = build_training_command(root, executable=executable)
    log_path = root / "logs" / "training.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record["status"] = "running"
    record["updated_at"] = _now()
    record["training"].update(
        {"command": command, "timeout_seconds": timeout_seconds, "return_code": None}
    )
    _validate_execution(record)
    _write_json(record_path, record)

    try:
        with log_path.open("w", encoding="utf-8") as log:
            result = runner(
                command,
                cwd=root,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        record["training"]["return_code"] = result.returncode
        if result.returncode == 0:
            checkpoint_dir = Path(record["training"]["checkpoint_directory"])
            checkpoints = [checkpoint_dir / "best.pt", checkpoint_dir / "last.pt"]
            if any(path.is_file() for path in checkpoints):
                record["status"] = "completed"
            else:
                record["status"] = "failed"
                record["error"] = "학습 프로세스는 성공했지만 후보 체크포인트가 생성되지 않았습니다."
        else:
            record["status"] = "failed"
            record["error"] = f"YOLO 학습 프로세스가 종료 코드 {result.returncode}로 실패했습니다."
    except subprocess.TimeoutExpired:
        record["status"] = "timed_out"
        record["error"] = f"학습이 제한 시간 {timeout_seconds}초를 초과했습니다."
    except KeyboardInterrupt:
        record["status"] = "interrupted"
        record["error"] = "사용자가 학습을 중단했습니다."
        record["updated_at"] = _now()
        _validate_execution(record)
        _write_json(record_path, record)
        raise
    except OSError as error:
        record["status"] = "failed"
        record["error"] = f"YOLO 학습 프로세스를 시작할 수 없습니다. 세부 정보: {error}"

    record["updated_at"] = _now()
    _validate_execution(record)
    _write_json(record_path, record)
    return record


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="승인된 변경으로 격리된 YOLO 후보 실험을 준비하고 선택적으로 실행합니다."
    )
    parser.add_argument("plan", help="change-plan.json 파일 경로")
    parser.add_argument("approval", help="approval.json 파일 경로")
    parser.add_argument("--training-config", help="원본 YOLO args.yaml 파일 경로")
    parser.add_argument("--inference-config", help="원본 추론 YAML 파일 경로")
    parser.add_argument("--output-root", default="candidate-runs", help="후보 실험 저장 경로")
    parser.add_argument("--run-training", action="store_true", help="후보 준비 후 YOLO 학습 실행")
    parser.add_argument("--timeout", type=int, default=3600, help="학습 제한 시간(초)")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    arguments = build_argument_parser().parse_args(argv)
    try:
        record, candidate = prepare_candidate(
            load_json(arguments.plan, "변경 계획"),
            load_json(arguments.approval, "승인 기록"),
            source_training=arguments.training_config,
            source_inference=arguments.inference_config,
            output_root=arguments.output_root,
        )
        if arguments.run_training:
            record = run_candidate_training(candidate, timeout_seconds=arguments.timeout)
    except (ExecutionError, ChangePlanError) as error:
        print(f"error: {error}", file=os.sys.stderr)
        return 1
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
