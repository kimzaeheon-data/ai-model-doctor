from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

ENGINE_VERSION = "0.1.0"

STATUS_LABELS = {
    "pass": "이상 징후 없음",
    "warning": "확인 필요",
    "insufficient_data": "판단 보류",
}


class DiagnosisError(ValueError):
    """Raised when an evaluation or rule configuration cannot be diagnosed."""


def _evidence(
    metric: str,
    observed: int | float | str | None,
    operator: str,
    threshold: int | float | str | None,
) -> dict[str, Any]:
    return {
        "metric": metric,
        "observed": observed,
        "operator": operator,
        "threshold": threshold,
    }


def diagnose_training_history(
    evaluation: dict[str, Any], rules: dict[str, Any]
) -> dict[str, Any]:
    epochs = evaluation.get("metrics", {}).get("epochs", [])
    completed = len(epochs)
    history_rules = rules["history"]
    preliminary_min = int(history_rules["preliminary_min_epochs"])
    reliable_min = int(history_rules["reliable_min_epochs"])

    if completed >= reliable_min:
        return {
            "code": "insufficient_training_history",
            "title": "학습 기록 부족",
            "status": "pass",
            "status_label": STATUS_LABELS["pass"],
            "severity": "none",
            "confidence": 1.0,
            "observation": f"학습 기록이 {completed} epoch 존재합니다.",
            "diagnosis": "신뢰 가능한 추세 분석을 위한 최소 학습 기록을 충족합니다.",
            "evidence": [
                _evidence("epochs_completed", completed, ">=", reliable_min)
            ],
            "possible_causes": [],
            "recommended_checks": [],
        }

    if completed < preliminary_min:
        detail = "추세 진단을 수행할 수 없습니다."
    else:
        detail = "예비 분석은 가능하지만 신뢰 가능한 추세 진단에는 부족합니다."
    return {
        "code": "insufficient_training_history",
        "title": "학습 기록 부족",
        "status": "insufficient_data",
        "status_label": STATUS_LABELS["insufficient_data"],
        "severity": "none",
        "confidence": 1.0,
        "observation": f"학습 기록이 {completed} epoch만 존재합니다.",
        "diagnosis": f"학습 기록이 부족하여 {detail}",
        "evidence": [
            _evidence("epochs_completed", completed, ">=", reliable_min)
        ],
        "possible_causes": [],
        "recommended_checks": [
            f"최소 {reliable_min} epoch 이상의 학습 결과로 다시 분석하세요."
        ],
    }


def diagnose_precision_recall_imbalance(
    evaluation: dict[str, Any], rules: dict[str, Any]
) -> dict[str, Any]:
    summary = evaluation.get("metrics", {}).get("summary", {})
    precision = summary.get("precision")
    recall = summary.get("recall")
    completed = len(evaluation.get("metrics", {}).get("epochs", []))
    history_rules = rules["history"]
    preliminary_min = int(history_rules["preliminary_min_epochs"])
    reliable_min = int(history_rules["reliable_min_epochs"])
    imbalance_rules = rules["precision_recall_imbalance"]
    warning_gap = float(imbalance_rules["warning_gap"])
    high_gap = float(imbalance_rules["high_severity_gap"])

    if completed < preliminary_min:
        values = (
            f"precision은 {precision:.5f}, recall은 {recall:.5f}이지만 "
            if isinstance(precision, (int, float)) and isinstance(recall, (int, float))
            else "Precision 또는 recall 값이 있더라도 "
        )
        return {
            "code": "precision_recall_imbalance",
            "title": "Precision·Recall 불균형",
            "status": "insufficient_data",
            "status_label": STATUS_LABELS["insufficient_data"],
            "severity": "none",
            "confidence": 1.0,
            "observation": f"{values}학습 기록이 {completed} epoch뿐입니다.",
            "diagnosis": "현재 차이가 모델의 안정적인 특성인지 판단할 수 없습니다.",
            "evidence": [
                _evidence("epochs_completed", completed, ">=", preliminary_min)
            ],
            "possible_causes": [],
            "recommended_checks": [
                f"최소 {preliminary_min} epoch 이상의 결과로 예비 진단을 다시 실행하세요."
            ],
        }

    if not isinstance(precision, (int, float)) or not isinstance(recall, (int, float)):
        return {
            "code": "precision_recall_imbalance",
            "title": "Precision·Recall 불균형",
            "status": "insufficient_data",
            "status_label": STATUS_LABELS["insufficient_data"],
            "severity": "none",
            "confidence": 1.0,
            "observation": "precision 또는 recall 값이 제공되지 않았습니다.",
            "diagnosis": "Precision·Recall 불균형을 판단할 수 없습니다.",
            "evidence": [
                _evidence("precision_recall_gap", None, "not_available", warning_gap)
            ],
            "possible_causes": [],
            "recommended_checks": [
                "precision과 recall이 포함된 검증 결과를 제공하세요."
            ],
        }

    gap = abs(float(precision) - float(recall))
    confidence = 0.6 if completed < reliable_min else 0.9
    preliminary_note = " 예비 진단 결과입니다." if completed < reliable_min else ""
    observation = (
        f"precision은 {precision:.5f}, recall은 {recall:.5f}이며 "
        f"차이는 {gap:.5f}입니다."
    )
    evidence = [_evidence("precision_recall_gap", round(gap, 8), ">=", warning_gap)]
    if gap < warning_gap:
        return {
            "code": "precision_recall_imbalance",
            "title": "Precision·Recall 불균형",
            "status": "pass",
            "status_label": STATUS_LABELS["pass"],
            "severity": "none",
            "confidence": confidence,
            "observation": observation,
            "diagnosis": f"전체 precision과 recall 사이에서 설정 기준 이상의 불균형이 발견되지 않았습니다.{preliminary_note}",
            "evidence": evidence,
            "possible_causes": [],
            "recommended_checks": [],
        }

    precision_is_higher = precision > recall
    return {
        "code": "precision_recall_imbalance",
        "title": "Precision·Recall 불균형",
        "status": "warning",
        "status_label": STATUS_LABELS["warning"],
        "severity": "high" if gap >= high_gap else "medium",
        "confidence": confidence,
        "observation": observation,
        "diagnosis": (
            "Precision에 비해 recall이 낮아 객체를 놓치는 경향이 있습니다."
            if precision_is_higher
            else "Recall에 비해 precision이 낮아 오탐이 발생하는 경향이 있습니다."
        ) + preliminary_note,
        "evidence": evidence,
        "possible_causes": (
            [
                "confidence threshold가 높을 가능성",
                "특정 클래스 또는 작은 객체 탐지 실패 가능성",
                "학습 데이터 다양성 부족 가능성",
            ]
            if precision_is_higher
            else [
                "confidence threshold가 낮을 가능성",
                "배경 또는 유사 객체 혼동 가능성",
                "학습 데이터의 음성 예제 부족 가능성",
            ]
        ),
        "recommended_checks": [
            "클래스별 precision과 recall을 확인하세요.",
            "BoxP_curve와 BoxR_curve를 함께 확인하세요.",
            "confidence threshold별 변화를 확인하세요.",
        ],
    }


def diagnose_training_plateau(
    evaluation: dict[str, Any], rules: dict[str, Any]
) -> dict[str, Any]:
    epochs = evaluation.get("metrics", {}).get("epochs", [])
    plateau_rules = rules["training_plateau"]
    min_epochs = int(plateau_rules["min_epochs"])
    window_ratio = float(plateau_rules["window_ratio"])
    min_window = int(plateau_rules["min_window_epochs"])
    threshold = float(plateau_rules["max_map50_95_improvement"])
    completed = len(epochs)

    if completed < min_epochs:
        return {
            "code": "training_plateau",
            "title": "학습 정체",
            "status": "insufficient_data",
            "status_label": STATUS_LABELS["insufficient_data"],
            "severity": "none",
            "confidence": 1.0,
            "observation": f"학습 기록이 {completed} epoch로 최소 기준인 {min_epochs} epoch보다 짧습니다.",
            "diagnosis": "mAP50-95의 최근 개선 추세를 판단할 수 없습니다.",
            "evidence": [
                _evidence("epochs_completed", completed, ">=", min_epochs)
            ],
            "possible_causes": [],
            "recommended_checks": [
                f"최소 {min_epochs} epoch 이상의 mAP50-95 기록을 제공하세요."
            ],
        }

    window_size = max(min_window, math.ceil(completed * window_ratio))
    comparison_epochs = epochs[-(window_size * 2) :]
    previous = [
        item.get("validation", {}).get("map50_95")
        for item in comparison_epochs[:window_size]
    ]
    recent = [
        item.get("validation", {}).get("map50_95")
        for item in comparison_epochs[window_size:]
    ]
    values = previous + recent
    if any(not isinstance(value, (int, float)) for value in values):
        return {
            "code": "training_plateau",
            "title": "학습 정체",
            "status": "insufficient_data",
            "status_label": STATUS_LABELS["insufficient_data"],
            "severity": "none",
            "confidence": 1.0,
            "observation": "최근 두 비교 구간에 mAP50-95 누락값이 있습니다.",
            "diagnosis": "학습 정체 여부를 안정적으로 판단할 수 없습니다.",
            "evidence": [
                _evidence("map50_95_window_complete", "incomplete", "==", "complete")
            ],
            "possible_causes": [],
            "recommended_checks": [
                "최근 두 비교 구간의 모든 mAP50-95 값을 제공하세요."
            ],
        }

    previous_peak = max(float(value) for value in previous)
    recent_peak = max(float(value) for value in recent)
    improvement = recent_peak - previous_peak
    observation = (
        f"이전 {window_size} epoch 최고 mAP50-95는 {previous_peak:.5f}, "
        f"최근 {window_size} epoch 최고값은 {recent_peak:.5f}이며 "
        f"개선 폭은 {improvement:.5f}입니다."
    )
    evidence = [
        _evidence(
            "map50_95_peak_improvement",
            round(improvement, 8),
            "<",
            threshold,
        )
    ]
    if improvement >= threshold:
        return {
            "code": "training_plateau",
            "title": "학습 정체",
            "status": "pass",
            "status_label": STATUS_LABELS["pass"],
            "severity": "none",
            "confidence": 0.9,
            "observation": observation,
            "diagnosis": "최근 구간에서 설정 기준 이상의 성능 개선이 확인됐습니다.",
            "evidence": evidence,
            "possible_causes": [],
            "recommended_checks": [],
        }

    return {
        "code": "training_plateau",
        "title": "학습 정체",
        "status": "warning",
        "status_label": STATUS_LABELS["warning"],
        "severity": "medium",
        "confidence": 0.85,
        "observation": observation,
        "diagnosis": "최근 mAP50-95 개선 폭이 기준보다 작아 학습이 정체됐을 가능성이 있습니다.",
        "evidence": evidence,
        "possible_causes": [
            "현재 learning rate에서 추가 개선이 어려울 가능성",
            "모델 용량 또는 입력 해상도의 한계 가능성",
            "데이터 품질이나 다양성의 한계 가능성",
        ],
        "recommended_checks": [
            "learning-rate 변화를 확인하세요.",
            "train 및 validation loss 추세를 확인하세요.",
            "추가 학습보다 데이터 개선이 필요한지 검토하세요.",
        ],
    }


def diagnose_suspected_overfitting(
    evaluation: dict[str, Any], rules: dict[str, Any]
) -> dict[str, Any]:
    epochs = evaluation.get("metrics", {}).get("epochs", [])
    overfit_rules = rules["suspected_overfitting"]
    min_epochs = int(overfit_rules["min_epochs"])
    window_ratio = float(overfit_rules["window_ratio"])
    min_window = int(overfit_rules["min_window_epochs"])
    min_train_decrease = float(overfit_rules["min_train_loss_decrease_ratio"])
    min_validation_increase = float(
        overfit_rules["min_validation_loss_increase_ratio"]
    )
    max_map_change = float(overfit_rules["max_map50_95_change"])
    completed = len(epochs)

    if completed < min_epochs:
        return {
            "code": "suspected_overfitting",
            "title": "과적합 의심",
            "status": "insufficient_data",
            "status_label": STATUS_LABELS["insufficient_data"],
            "severity": "none",
            "confidence": 1.0,
            "observation": f"학습 기록이 {completed} epoch로 최소 기준인 {min_epochs} epoch보다 짧습니다.",
            "diagnosis": "Train·validation loss와 mAP50-95 추세를 비교할 수 없습니다.",
            "evidence": [
                _evidence("epochs_completed", completed, ">=", min_epochs)
            ],
            "possible_causes": [],
            "recommended_checks": [
                f"최소 {min_epochs} epoch 이상의 학습 결과를 제공하세요."
            ],
        }

    window_size = max(min_window, math.ceil(completed * window_ratio))
    comparison_epochs = epochs[-(window_size * 2) :]

    def total_loss(item: dict[str, Any], section: str) -> float | None:
        values = [
            item.get(section, {}).get(name)
            for name in ("box_loss", "cls_loss", "dfl_loss")
        ]
        if any(not isinstance(value, (int, float)) for value in values):
            return None
        return sum(float(value) for value in values)

    train_losses = [total_loss(item, "train") for item in comparison_epochs]
    validation_losses = [
        total_loss(item, "validation") for item in comparison_epochs
    ]
    map_values = [
        item.get("validation", {}).get("map50_95") for item in comparison_epochs
    ]
    required_values = train_losses + validation_losses + map_values
    if any(not isinstance(value, (int, float)) for value in required_values):
        return {
            "code": "suspected_overfitting",
            "title": "과적합 의심",
            "status": "insufficient_data",
            "status_label": STATUS_LABELS["insufficient_data"],
            "severity": "none",
            "confidence": 1.0,
            "observation": "최근 두 비교 구간에 필요한 loss 또는 mAP50-95 누락값이 있습니다.",
            "diagnosis": "과적합 여부를 안정적으로 판단할 수 없습니다.",
            "evidence": [
                _evidence("overfitting_window_complete", "incomplete", "==", "complete")
            ],
            "possible_causes": [],
            "recommended_checks": [
                "최근 두 비교 구간의 box, cls, DFL loss와 mAP50-95를 모두 제공하세요."
            ],
        }

    previous_train = sum(train_losses[:window_size]) / window_size
    recent_train = sum(train_losses[window_size:]) / window_size
    previous_validation = sum(validation_losses[:window_size]) / window_size
    recent_validation = sum(validation_losses[window_size:]) / window_size
    if previous_train <= 0 or previous_validation <= 0:
        return {
            "code": "suspected_overfitting",
            "title": "과적합 의심",
            "status": "insufficient_data",
            "status_label": STATUS_LABELS["insufficient_data"],
            "severity": "none",
            "confidence": 1.0,
            "observation": "이전 비교 구간의 평균 total loss가 0 이하입니다.",
            "diagnosis": "Loss 변화율을 안전하게 계산할 수 없습니다.",
            "evidence": [
                _evidence("previous_window_total_loss", 0, ">", 0)
            ],
            "possible_causes": [],
            "recommended_checks": [
                "원본 loss 값과 Parser 변환 결과를 확인하세요."
            ],
        }

    train_decrease = (previous_train - recent_train) / previous_train
    validation_increase = (
        recent_validation - previous_validation
    ) / previous_validation
    previous_map_peak = max(float(value) for value in map_values[:window_size])
    recent_map_peak = max(float(value) for value in map_values[window_size:])
    map_change = recent_map_peak - previous_map_peak
    epsilon = 1e-12
    is_suspected = (
        train_decrease + epsilon >= min_train_decrease
        and validation_increase + epsilon >= min_validation_increase
        and map_change <= max_map_change + epsilon
    )
    observation = (
        f"Train loss는 {train_decrease * 100:.2f}% 감소했고 validation loss는 "
        f"{validation_increase * 100:.2f}% 증가했으며 mAP50-95 개선 폭은 "
        f"{map_change:.5f}입니다."
    )
    evidence = [
        _evidence(
            "train_total_loss_decrease_ratio",
            round(train_decrease, 8),
            ">=",
            min_train_decrease,
        ),
        _evidence(
            "validation_total_loss_increase_ratio",
            round(validation_increase, 8),
            ">=",
            min_validation_increase,
        ),
        _evidence(
            "map50_95_peak_change", round(map_change, 8), "<=", max_map_change
        ),
    ]
    if not is_suspected:
        return {
            "code": "suspected_overfitting",
            "title": "과적합 의심",
            "status": "pass",
            "status_label": STATUS_LABELS["pass"],
            "severity": "none",
            "confidence": 0.9,
            "observation": observation,
            "diagnosis": "과적합 의심에 필요한 복합 조건이 모두 충족되지 않았습니다.",
            "evidence": evidence,
            "possible_causes": [],
            "recommended_checks": [],
        }

    return {
        "code": "suspected_overfitting",
        "title": "과적합 의심",
        "status": "warning",
        "status_label": STATUS_LABELS["warning"],
        "severity": "medium",
        "confidence": 0.9,
        "observation": observation,
        "diagnosis": "Train loss는 감소하지만 validation loss가 증가하고 mAP 개선도 제한되어 과적합 가능성이 있습니다.",
        "evidence": evidence,
        "possible_causes": [
            "학습 데이터의 양이나 다양성 부족 가능성",
            "augmentation 또는 정규화 부족 가능성",
            "현재 데이터 규모에 비해 모델 용량이 클 가능성",
            "필요 이상으로 긴 학습 가능성",
        ],
        "recommended_checks": [
            "최고 성능 epoch 이후 validation loss와 mAP 변화를 확인하세요.",
            "Train·validation 데이터 분포 차이를 확인하세요.",
            "augmentation, weight decay 및 early stopping 설정을 검토하세요.",
        ],
    }


def diagnose_post_peak_degradation(
    evaluation: dict[str, Any], rules: dict[str, Any]
) -> dict[str, Any]:
    epochs = evaluation.get("metrics", {}).get("epochs", [])
    degradation_rules = rules["post_peak_degradation"]
    min_epochs = int(degradation_rules["min_epochs"])
    min_after_peak_ratio = float(
        degradation_rules["min_epochs_after_peak_ratio"]
    )
    min_drop = float(degradation_rules["min_map50_95_drop"])
    completed = len(epochs)

    if completed < min_epochs:
        return {
            "code": "post_peak_degradation",
            "title": "최적 epoch 이후 성능 하락",
            "status": "insufficient_data",
            "status_label": STATUS_LABELS["insufficient_data"],
            "severity": "none",
            "confidence": 1.0,
            "observation": f"학습 기록이 {completed} epoch로 최소 기준인 {min_epochs} epoch보다 짧습니다.",
            "diagnosis": "최고점 이후 성능 변화를 판단할 수 없습니다.",
            "evidence": [
                _evidence("epochs_completed", completed, ">=", min_epochs)
            ],
            "possible_causes": [],
            "recommended_checks": [
                f"최소 {min_epochs} epoch 이상의 mAP50-95 기록을 제공하세요."
            ],
        }

    map_values = [
        item.get("validation", {}).get("map50_95") for item in epochs
    ]
    if any(not isinstance(value, (int, float)) for value in map_values):
        return {
            "code": "post_peak_degradation",
            "title": "최적 epoch 이후 성능 하락",
            "status": "insufficient_data",
            "status_label": STATUS_LABELS["insufficient_data"],
            "severity": "none",
            "confidence": 1.0,
            "observation": "전체 학습 기록 중 일부 epoch의 mAP50-95가 누락됐습니다.",
            "diagnosis": "전체 최고점과 마지막 성능을 안정적으로 비교할 수 없습니다.",
            "evidence": [
                _evidence("map50_95_history_complete", "incomplete", "==", "complete")
            ],
            "possible_causes": [],
            "recommended_checks": [
                "모든 epoch의 validation mAP50-95 값을 제공하세요."
            ],
        }

    peak_value = max(float(value) for value in map_values)
    peak_index = max(
        index for index, value in enumerate(map_values) if float(value) == peak_value
    )
    peak_epoch = epochs[peak_index].get("epoch", peak_index)
    final_value = float(map_values[-1])
    final_epoch = epochs[-1].get("epoch", completed - 1)
    after_peak = completed - peak_index - 1
    required_after_peak = math.ceil(completed * min_after_peak_ratio)
    drop = peak_value - final_value
    epsilon = 1e-12
    is_degraded = (
        after_peak >= required_after_peak and drop + epsilon >= min_drop
    )
    observation = (
        f"최고 mAP50-95는 epoch {peak_epoch}의 {peak_value:.5f}, 마지막 epoch "
        f"{final_epoch}의 값은 {final_value:.5f}이며 하락 폭은 {drop:.5f}입니다. "
        f"최고점 이후 기록은 {after_peak}개입니다."
    )
    evidence = [
        _evidence(
            "records_after_peak", after_peak, ">=", required_after_peak
        ),
        _evidence("map50_95_drop", round(drop, 8), ">=", min_drop),
    ]
    if not is_degraded:
        return {
            "code": "post_peak_degradation",
            "title": "최적 epoch 이후 성능 하락",
            "status": "pass",
            "status_label": STATUS_LABELS["pass"],
            "severity": "none",
            "confidence": 0.9,
            "observation": observation,
            "diagnosis": "최고점 이후 유의미한 성능 하락에 필요한 조건이 모두 충족되지 않았습니다.",
            "evidence": evidence,
            "possible_causes": [],
            "recommended_checks": [],
        }

    return {
        "code": "post_peak_degradation",
        "title": "최적 epoch 이후 성능 하락",
        "status": "warning",
        "status_label": STATUS_LABELS["warning"],
        "severity": "medium",
        "confidence": 0.9,
        "observation": observation,
        "diagnosis": "최고 성능 이후 충분한 기간이 지났고 마지막 mAP50-95가 유의미하게 하락했습니다.",
        "evidence": evidence,
        "possible_causes": [
            "최고 성능 이후 과도한 추가 학습 가능성",
            "과적합 가능성",
            "후반부 learning rate 또는 augmentation 영향 가능성",
            "Validation 지표 진동 가능성",
        ],
        "recommended_checks": [
            "마지막 체크포인트 대신 best.pt를 평가에 사용하는지 확인하세요.",
            "최고점 이후 validation loss 변화를 확인하세요.",
            "Patience 및 early stopping 설정을 검토하세요.",
        ],
    }


def _overall_status(diagnoses: list[dict[str, Any]]) -> str:
    statuses = {item["status"] for item in diagnoses}
    if "warning" in statuses:
        return "warning"
    if "insufficient_data" in statuses:
        return "insufficient_data"
    return "pass"


def diagnose(evaluation: dict[str, Any], rules: dict[str, Any]) -> dict[str, Any]:
    if evaluation.get("schema_version") != "0.1.0":
        raise DiagnosisError("Only evaluation schema version 0.1.0 is supported")
    if str(rules.get("version")) != "0.1.0":
        raise DiagnosisError("Only diagnosis rules version 0.1.0 is supported")

    diagnoses = [
        diagnose_training_history(evaluation, rules),
        diagnose_precision_recall_imbalance(evaluation, rules),
        diagnose_training_plateau(evaluation, rules),
        diagnose_suspected_overfitting(evaluation, rules),
        diagnose_post_peak_degradation(evaluation, rules),
    ]
    counts = {
        status: sum(item["status"] == status for item in diagnoses)
        for status in ("pass", "warning", "insufficient_data")
    }
    return {
        "schema_version": "0.1.0",
        "evaluation_schema_version": evaluation["schema_version"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "engine": {
            "name": "ai-model-doctor-rule-engine",
            "version": ENGINE_VERSION,
            "rules_version": str(rules["version"]),
        },
        "summary": {
            "overall_status": _overall_status(diagnoses),
            **counts,
        },
        "diagnoses": diagnoses,
    }


def load_evaluation(path: str | os.PathLike[str]) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DiagnosisError(f"Could not read evaluation JSON: {error}") from error
    if not isinstance(value, dict):
        raise DiagnosisError("Evaluation JSON root must be an object")
    return value


def load_rules(path: str | os.PathLike[str]) -> dict[str, Any]:
    try:
        value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise DiagnosisError(f"Could not read diagnosis rules: {error}") from error
    if not isinstance(value, dict):
        raise DiagnosisError("Diagnosis rules root must be a mapping")
    return value


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose a normalized YOLO evaluation JSON"
    )
    parser.add_argument("evaluation", help="Path to evaluation.json")
    parser.add_argument(
        "--rules",
        default="configs/diagnosis-rules.yaml",
        help="Path to diagnosis rule configuration",
    )
    parser.add_argument("-o", "--output", default="diagnosis.json", help="Output JSON path")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    arguments = build_argument_parser().parse_args(argv)
    try:
        result = diagnose(
            load_evaluation(arguments.evaluation), load_rules(arguments.rules)
        )
        output = Path(arguments.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except DiagnosisError as error:
        print(f"error: {error}", file=os.sys.stderr)
        return 1
    print(f"Created {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
