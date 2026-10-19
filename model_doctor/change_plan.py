from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

import yaml
from jsonschema import Draft202012Validator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY_PATH = PROJECT_ROOT / "configs" / "change-policy.yaml"
DEFAULT_PLAN_SCHEMA_PATH = PROJECT_ROOT / "schemas" / "change-plan.schema.json"
DEFAULT_APPROVAL_SCHEMA_PATH = PROJECT_ROOT / "schemas" / "approval.schema.json"


class ChangePlanError(ValueError):
    """변경 계획 또는 승인 기록이 안전 조건을 충족하지 못하면 발생합니다."""


def load_json(path: str | Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ChangePlanError(f"{label} JSON을 읽을 수 없습니다. 세부 정보: {error}") from error
    if not isinstance(value, dict):
        raise ChangePlanError(f"{label} JSON의 최상위 값은 객체여야 합니다.")
    return value


def load_policy(path: str | Path = DEFAULT_POLICY_PATH) -> dict[str, Any]:
    try:
        value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ChangePlanError(f"변경 정책을 읽을 수 없습니다. 세부 정보: {error}") from error
    if not isinstance(value, dict) or str(value.get("version")) != "0.1.0":
        raise ChangePlanError("변경 정책은 0.1.0 버전만 지원합니다.")
    return value


def _load_schema(path: str | Path, label: str) -> dict[str, Any]:
    return load_json(path, label)


def _validate_schema(value: dict[str, Any], schema: dict[str, Any], label: str) -> None:
    errors = sorted(
        Draft202012Validator(schema).iter_errors(value),
        key=lambda error: list(error.absolute_path),
    )
    if errors:
        location = ".".join(str(item) for item in errors[0].absolute_path) or "root"
        raise ChangePlanError(f"{label}이 {location} 위치에서 스키마를 충족하지 않습니다.")


def plan_digest(plan: dict[str, Any]) -> str:
    canonical = json.dumps(
        plan, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _parameter_rules(policy: dict[str, Any], category: str, parameter: str) -> tuple[str, dict[str, Any]]:
    target = policy.get("allowed_targets", {}).get(category)
    if not isinstance(target, dict):
        raise ChangePlanError(f"허용되지 않은 변경 범주입니다: {category}")
    rules = target.get("parameters", {}).get(parameter)
    if not isinstance(rules, dict):
        raise ChangePlanError(f"허용되지 않은 변경 파라미터입니다: {parameter}")
    return str(target["file"]), rules


def _validate_value(value: Any, rules: dict[str, Any], parameter: str) -> None:
    expected = rules["type"]
    valid_type = {
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "string": isinstance(value, str),
    }.get(expected, False)
    if not valid_type:
        raise ChangePlanError(f"{parameter}의 제안 값 타입이 변경 정책과 다릅니다.")
    if "enum" in rules and value not in rules["enum"]:
        raise ChangePlanError(f"{parameter}의 제안 값이 허용 목록에 없습니다.")
    if "minimum" in rules and value < rules["minimum"]:
        raise ChangePlanError(f"{parameter}의 제안 값이 허용 최솟값보다 작습니다.")
    if "maximum" in rules and value > rules["maximum"]:
        raise ChangePlanError(f"{parameter}의 제안 값이 허용 최댓값보다 큽니다.")


def validate_change_plan(
    plan: dict[str, Any],
    *,
    policy: dict[str, Any] | None = None,
    schema: dict[str, Any] | None = None,
) -> None:
    policy = policy or load_policy()
    schema = schema or _load_schema(DEFAULT_PLAN_SCHEMA_PATH, "변경 계획 스키마")
    _validate_schema(plan, schema, "변경 계획")

    seen: set[str] = set()
    for change in plan["changes"]:
        change_id = change["change_id"]
        if change_id in seen:
            raise ChangePlanError(f"중복된 change_id입니다: {change_id}")
        seen.add(change_id)
        target_file, rules = _parameter_rules(
            policy, change["category"], change["parameter"]
        )
        if change["target_file"] != target_file:
            raise ChangePlanError(
                f"{change['parameter']}의 대상 파일이 변경 정책과 일치하지 않습니다."
            )
        if change["before"] == change["after"]:
            raise ChangePlanError(f"{change_id}의 변경 전후 값이 같습니다.")
        _validate_value(change["after"], rules, change["parameter"])


def create_approval(
    plan: dict[str, Any],
    decision: str,
    *,
    selected_change_ids: Iterable[str] = (),
    requested_revisions: Iterable[str] = (),
) -> dict[str, Any]:
    validate_change_plan(plan)
    all_ids = [change["change_id"] for change in plan["changes"]]
    selected = list(dict.fromkeys(selected_change_ids))
    revisions = [item.strip() for item in requested_revisions if item.strip()]

    if decision == "approve_all":
        if revisions:
            raise ChangePlanError("전체 승인에는 수정 요청을 함께 기록할 수 없습니다.")
        approved = all_ids
    elif decision == "approve_selected":
        if revisions:
            raise ChangePlanError("일부 승인에는 수정 요청을 함께 기록할 수 없습니다.")
        if not selected:
            raise ChangePlanError("일부 승인에는 한 개 이상의 change_id가 필요합니다.")
        unknown = set(selected) - set(all_ids)
        if unknown:
            raise ChangePlanError(f"계획에 없는 change_id입니다: {sorted(unknown)}")
        approved = selected
    elif decision == "request_revision":
        if not revisions:
            raise ChangePlanError("수정 요청에는 한 개 이상의 요청 내용이 필요합니다.")
        approved = []
    elif decision == "cancel":
        if revisions:
            raise ChangePlanError("취소 결정에는 수정 요청을 함께 기록할 수 없습니다.")
        approved = []
    else:
        raise ChangePlanError(f"지원하지 않는 승인 결정입니다: {decision}")

    approval = {
        "schema_version": "0.1.0",
        "approval_id": f"approval-{uuid4().hex[:12]}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "actor": "user",
        "plan_id": plan["plan_id"],
        "plan_digest": plan_digest(plan),
        "decision": decision,
        "approved_change_ids": approved,
        "requested_revisions": revisions,
        "execution_authorized": bool(approved),
    }
    schema = _load_schema(DEFAULT_APPROVAL_SCHEMA_PATH, "승인 스키마")
    _validate_schema(approval, schema, "승인 기록")
    return approval


def authorized_changes(
    plan: dict[str, Any],
    approval: dict[str, Any],
    *,
    policy: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    validate_change_plan(plan, policy=policy)
    approval_schema = _load_schema(DEFAULT_APPROVAL_SCHEMA_PATH, "승인 스키마")
    _validate_schema(approval, approval_schema, "승인 기록")

    if approval["plan_id"] != plan["plan_id"]:
        raise ChangePlanError("승인 기록의 plan_id가 변경 계획과 일치하지 않습니다.")
    if approval["plan_digest"] != plan_digest(plan):
        raise ChangePlanError("변경 계획이 승인 이후 변경되어 기존 승인이 무효입니다.")

    expected = create_approval(
        plan,
        approval["decision"],
        selected_change_ids=approval["approved_change_ids"],
        requested_revisions=approval["requested_revisions"],
    )
    if approval["approved_change_ids"] != expected["approved_change_ids"]:
        raise ChangePlanError("승인 결정과 승인된 변경 범위가 일치하지 않습니다.")
    if approval["execution_authorized"] != expected["execution_authorized"]:
        raise ChangePlanError("승인 기록의 실행 허용 상태가 올바르지 않습니다.")
    if not approval["execution_authorized"]:
        return []

    approved = set(approval["approved_change_ids"])
    return [change for change in plan["changes"] if change["change_id"] in approved]
