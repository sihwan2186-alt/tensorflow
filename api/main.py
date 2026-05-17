from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    import tensorflow as tf
except Exception as exc:  # pragma: no cover - shown through /health when TensorFlow is missing
    tf = None
    TF_IMPORT_ERROR = exc


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "model" / "edurisk_model.keras"
PREPROCESSOR_PATH = ROOT_DIR / "model" / "preprocessor.joblib"
METADATA_PATH = ROOT_DIR / "model" / "metadata.json"

DEFAULT_THRESHOLD = 0.5
MEDIUM_RISK_THRESHOLD = 0.4
CRITICAL_RISK_THRESHOLD = 0.9

GRADE_RISK_POINTS = {
    "A": 0.0,
    "B": 2.0,
    "C": 6.0,
    "D": 10.0,
    "F": 14.0,
}

app = FastAPI(
    title="EduRisk AI Prediction API",
    description="학생 학습 활동 데이터를 기반으로 중도 포기 위험 확률, 위험 등급, 설명 가능한 개입 계획을 제공하는 FastAPI 서비스",
    version="1.1.0",
)

model: Any | None = None
models: list[Any] = []
preprocessor: Any | None = None
metadata: dict[str, Any] = {}
load_error: str | None = None


class BatchPredictionRequest(BaseModel):
    students: list[dict[str, Any]] = Field(..., min_length=1, max_length=500)


def get_payload_dict(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        return {str(key).strip().strip('"'): value for key, value in payload.items()}
    if hasattr(payload, "model_dump"):
        return payload.model_dump()
    return payload.dict()


def resolve_model_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return ROOT_DIR / path


def load_artifacts() -> None:
    global model, models, preprocessor, metadata, load_error

    if tf is None:
        load_error = f"TensorFlow import failed: {TF_IMPORT_ERROR}"
        return
    if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
        load_error = "Model artifact is missing. Run python scripts/train_model.py first."
        return

    if METADATA_PATH.exists():
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    ensemble_paths = [resolve_model_path(path) for path in metadata.get("ensemble_model_paths", [])]
    if ensemble_paths:
        missing_paths = [str(path) for path in ensemble_paths if not path.exists()]
        if missing_paths:
            load_error = f"Ensemble model artifact is missing: {missing_paths}"
            return
        models = [tf.keras.models.load_model(path) for path in ensemble_paths]
    else:
        models = [tf.keras.models.load_model(MODEL_PATH)]

    model = models[0]
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    load_error = None


@app.on_event("startup")
def startup_event() -> None:
    load_artifacts()


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "EduRisk AI", "status": "ready" if load_error is None else "not_ready"}


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ready": load_error is None,
        "error": load_error,
        "model_path": str(MODEL_PATH),
        "model_strategy": metadata.get("model_strategy", "single_tensorflow_model"),
        "model_count": len(models),
        "preprocessor_path": str(PREPROCESSOR_PATH),
        "threshold": metadata.get("threshold", DEFAULT_THRESHOLD),
        "medium_risk_threshold": MEDIUM_RISK_THRESHOLD,
        "critical_risk_threshold": CRITICAL_RISK_THRESHOLD,
    }


@app.get("/preprocessing-info")
def preprocessing_info() -> dict[str, Any]:
    return {
        "mission": "미션 1: 데이터 로드 및 preprocessing",
        "data_source": "data/student_dropout_real.csv",
        "external_source": metadata.get(
            "dataset_source",
            {
                "name": "Predict Students' Dropout and Academic Success",
                "repository": "UCI Machine Learning Repository",
                "url": "https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success",
            },
        ),
        "target_column": metadata.get("target_column", "dropout"),
        "drop_columns": metadata.get("drop_columns", ["student_id"]),
        "categorical_features": {
            "columns": metadata.get("categorical_features", ["gender", "course_type", "previous_grade"]),
            "method": "OneHotEncoder(handle_unknown='ignore')",
            "reason": "성별, 과목 유형, 이전 성적처럼 순서가 고정되지 않은 범주형 값을 TensorFlow 입력 벡터로 변환합니다.",
        },
        "numeric_features": {
            "columns": metadata.get(
                "numeric_features",
                [
                    "age",
                    "attendance_rate",
                    "assignment_submit_rate",
                    "quiz_average",
                    "login_count",
                    "video_watch_time",
                    "forum_activity",
                ],
            ),
            "method": "StandardScaler",
            "reason": "출석률, 점수, 접속 횟수처럼 단위가 다른 수치형 값의 스케일 차이를 줄입니다.",
        },
        "engineered_features": metadata.get("feature_engineering", {}),
        "split_strategy": "train_test_split(..., stratify=y)",
        "saved_artifact": str(PREPROCESSOR_PATH),
    }


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    return {
        "mission": "미션 2: TensorFlow 분류 모델",
        "framework": "TensorFlow Keras",
        "task": "binary classification",
        "input": "전처리된 학생 학습 활동 벡터",
        "model_strategy": metadata.get("model_strategy", "single_tensorflow_model"),
        "ensemble_size": metadata.get("ensemble_size", 1),
        "model_preset": metadata.get("model_preset", "compact"),
        "model_preset_config": metadata.get("model_preset_config", {}),
        "architecture": [
            "Input",
            "Dense hidden layers from model_preset_config.hidden_units",
            "BatchNormalization + ReLU + Dropout for regularized preset",
            "Dense(1, activation='sigmoid')",
        ],
        "loss": "binary_crossentropy",
        "optimizer": "Adam",
        "threshold": metadata.get("threshold", DEFAULT_THRESHOLD),
        "metrics": metadata.get("metrics", {}),
        "saved_model": str(MODEL_PATH),
    }


@app.get("/service-info")
def service_info() -> dict[str, Any]:
    return {
        "mission": "미션 3: n8n 서비스 완성",
        "service_flow": [
            "Webhook Trigger에서 학생 학습 데이터 수신",
            "Set Node에서 API 입력 필드 정리",
            "HTTP Request Node에서 FastAPI /predict 호출",
            "IF Node에서 위험도와 위험 등급 기준 분기",
            "AI Agent가 교수자용 알림 문장 생성",
            "Slack/Gmail로 알림 전송",
            "Google Sheets에 예측 결과와 개입 계획 기록",
        ],
        "workflows": {
            "credential_free_test": "n8n/edurisk_workflow_local_test.json",
            "advanced_local_routing": "n8n/edurisk_workflow_advanced_local.json",
            "full_service": "n8n/edurisk_workflow.json",
        },
        "idea_points": [
            "중도 포기 위험을 4단계로 세분화",
            "설명 가능한 위험 요인 점수 제공",
            "학생별 맞춤 개입 계획 자동 생성",
            "배치 예측으로 여러 학생을 한 번에 조기경보 처리",
        ],
    }


def ensure_ready() -> None:
    if load_error is not None or not models or preprocessor is None:
        raise HTTPException(status_code=503, detail=load_error)


def get_feature_order() -> list[str]:
    return get_categorical_features() + get_numeric_features()


def get_categorical_features() -> list[str]:
    return metadata.get("categorical_features", ["gender", "course_type", "previous_grade"])


def get_numeric_features() -> list[str]:
    return metadata.get(
        "numeric_features",
        [
            "age",
            "attendance_rate",
            "assignment_submit_rate",
            "quiz_average",
            "login_count",
            "video_watch_time",
            "forum_activity",
        ],
    )


def get_raw_numeric_features() -> list[str]:
    return metadata.get("raw_numeric_features", get_numeric_features())


def add_engineered_features_to_frame(df: pd.DataFrame) -> pd.DataFrame:
    if not metadata.get("engineered_numeric_features"):
        return df

    df = df.copy()
    first_enrolled = df["Curricular units 1st sem (enrolled)"].where(
        df["Curricular units 1st sem (enrolled)"] != 0
    )
    second_enrolled = df["Curricular units 2nd sem (enrolled)"].where(
        df["Curricular units 2nd sem (enrolled)"] != 0
    )
    first_approved = df["Curricular units 1st sem (approved)"]
    second_approved = df["Curricular units 2nd sem (approved)"]
    first_approval_rate = (first_approved / first_enrolled).fillna(0.0)
    second_approval_rate = (second_approved / second_enrolled).fillna(0.0)

    df["first_sem_approval_rate"] = first_approval_rate
    df["second_sem_approval_rate"] = second_approval_rate
    df["first_sem_failure_count"] = (
        df["Curricular units 1st sem (enrolled)"] - df["Curricular units 1st sem (approved)"]
    ).clip(lower=0)
    df["second_sem_failure_count"] = (
        df["Curricular units 2nd sem (enrolled)"] - df["Curricular units 2nd sem (approved)"]
    ).clip(lower=0)
    df["grade_delta_second_minus_first"] = (
        df["Curricular units 2nd sem (grade)"] - df["Curricular units 1st sem (grade)"]
    )
    df["approved_units_total"] = (
        df["Curricular units 1st sem (approved)"] + df["Curricular units 2nd sem (approved)"]
    )
    df["evaluation_load_total"] = (
        df["Curricular units 1st sem (evaluations)"] + df["Curricular units 2nd sem (evaluations)"]
    )
    df["without_evaluations_total"] = (
        df["Curricular units 1st sem (without evaluations)"]
        + df["Curricular units 2nd sem (without evaluations)"]
    )
    df["financial_risk_score"] = df["Debtor"] + (1 - df["Tuition fees up to date"])
    df["academic_momentum"] = df["second_sem_approval_rate"] - df["first_sem_approval_rate"]
    return df


def as_float(row: dict[str, Any], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return 0.0


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def points_below(value: float, target: float, max_points: float) -> float:
    if value >= target:
        return 0.0
    return clamp((target - value) / target * max_points, 0.0, max_points)


def build_risk_score_breakdown(row: dict[str, Any]) -> list[dict[str, Any]]:
    if "Tuition fees up to date" in row or "Curricular units 2nd sem (approved)" in row:
        return build_uci_risk_score_breakdown(row)
    return build_legacy_risk_score_breakdown(row)


def build_uci_risk_score_breakdown(row: dict[str, Any]) -> list[dict[str, Any]]:
    tuition_up_to_date = as_float(row, "Tuition fees up to date")
    debtor = as_float(row, "Debtor")
    age = as_float(row, "Age at enrollment")
    admission_grade = as_float(row, "Admission grade")
    previous_grade = as_float(row, "Previous qualification (grade)")
    first_enrolled = as_float(row, "Curricular units 1st sem (enrolled)")
    first_approved = as_float(row, "Curricular units 1st sem (approved)")
    first_grade = as_float(row, "Curricular units 1st sem (grade)")
    second_enrolled = as_float(row, "Curricular units 2nd sem (enrolled)")
    second_approved = as_float(row, "Curricular units 2nd sem (approved)")
    second_grade = as_float(row, "Curricular units 2nd sem (grade)")
    scholarship = as_float(row, "Scholarship holder")

    first_approval_rate = first_approved / first_enrolled if first_enrolled > 0 else 0.0
    second_approval_rate = second_approved / second_enrolled if second_enrolled > 0 else 0.0

    factors = [
        {
            "feature": "Tuition fees up to date",
            "label": "등록금 납부 상태",
            "value": tuition_up_to_date,
            "baseline": 1,
            "contribution": 20.0 if tuition_up_to_date == 0 else 0.0,
            "reason": "등록금 납부가 최신 상태가 아닙니다.",
            "recommended_action": "행정 지원 또는 장학/분납 상담을 연결합니다.",
        },
        {
            "feature": "Debtor",
            "label": "채무 여부",
            "value": debtor,
            "baseline": 0,
            "contribution": 15.0 if debtor == 1 else 0.0,
            "reason": "채무 상태가 학업 지속 위험을 높일 수 있습니다.",
            "recommended_action": "재정 상담과 납부 계획 안내를 우선 진행합니다.",
        },
        {
            "feature": "Curricular units 2nd sem (approved)",
            "label": "2학기 이수 승인율",
            "value": round(second_approval_rate, 2),
            "baseline": 0.6,
            "contribution": points_below(second_approval_rate, 0.6, 20),
            "reason": "2학기 이수 승인율이 낮습니다.",
            "recommended_action": "2학기 미이수 과목과 재수강 가능 과목을 점검합니다.",
        },
        {
            "feature": "Curricular units 2nd sem (grade)",
            "label": "2학기 평균 성적",
            "value": second_grade,
            "baseline": 10,
            "contribution": points_below(second_grade, 10, 15),
            "reason": "2학기 평균 성적이 낮습니다.",
            "recommended_action": "최근 평가 결과를 바탕으로 보충 학습 계획을 제안합니다.",
        },
        {
            "feature": "Curricular units 1st sem (approved)",
            "label": "1학기 이수 승인율",
            "value": round(first_approval_rate, 2),
            "baseline": 0.6,
            "contribution": points_below(first_approval_rate, 0.6, 12),
            "reason": "1학기 이수 승인율이 낮습니다.",
            "recommended_action": "초기 학업 적응 문제를 확인하고 튜터링을 연결합니다.",
        },
        {
            "feature": "Curricular units 1st sem (grade)",
            "label": "1학기 평균 성적",
            "value": first_grade,
            "baseline": 10,
            "contribution": points_below(first_grade, 10, 10),
            "reason": "1학기 평균 성적이 낮습니다.",
            "recommended_action": "기초 과목 이해도와 학습 전략을 점검합니다.",
        },
        {
            "feature": "Admission grade",
            "label": "입학 성적",
            "value": admission_grade,
            "baseline": 120,
            "contribution": points_below(admission_grade, 120, 8),
            "reason": "입학 성적이 기준보다 낮습니다.",
            "recommended_action": "선수 지식 보강 자료를 제공합니다.",
        },
        {
            "feature": "Previous qualification (grade)",
            "label": "이전 학력 성적",
            "value": previous_grade,
            "baseline": 120,
            "contribution": points_below(previous_grade, 120, 8),
            "reason": "이전 학력 성적이 기준보다 낮습니다.",
            "recommended_action": "입학 전 학업 배경에 맞춘 보충 학습을 안내합니다.",
        },
        {
            "feature": "Age at enrollment",
            "label": "입학 나이",
            "value": age,
            "baseline": 25,
            "contribution": clamp((age - 25) / 20 * 6, 0.0, 6.0),
            "reason": "입학 나이가 높아 학업 병행 부담이 있을 수 있습니다.",
            "recommended_action": "학업 일정과 생활 부담을 함께 점검합니다.",
        },
        {
            "feature": "Scholarship holder",
            "label": "장학금 수혜 여부",
            "value": scholarship,
            "baseline": 1,
            "contribution": 3.0 if scholarship == 0 else 0.0,
            "reason": "장학금 미수혜로 재정 부담 가능성이 있습니다.",
            "recommended_action": "장학금 또는 학비 지원 제도를 안내합니다.",
        },
    ]

    for factor in factors:
        factor["contribution"] = round(float(factor["contribution"]), 1)
        if factor["contribution"] <= 0:
            factor["reason"] = "기준을 충족했습니다."

    return sorted(factors, key=lambda item: item["contribution"], reverse=True)


def build_legacy_risk_score_breakdown(row: dict[str, Any]) -> list[dict[str, Any]]:
    attendance = as_float(row, "attendance_rate")
    assignment = as_float(row, "assignment_submit_rate")
    quiz = as_float(row, "quiz_average")
    login_count = as_float(row, "login_count")
    video_watch_time = as_float(row, "video_watch_time")
    forum_activity = as_float(row, "forum_activity")
    previous_grade = str(row.get("previous_grade", "")).upper()

    factors = [
        {
            "feature": "attendance_rate",
            "label": "출석률",
            "value": attendance,
            "baseline": 70,
            "contribution": points_below(attendance, 70, 25),
            "reason": "출석률이 70% 기준보다 낮습니다.",
            "recommended_action": "최근 결석 사유를 확인하고 출석 회복 계획을 잡습니다.",
        },
        {
            "feature": "assignment_submit_rate",
            "label": "과제 제출률",
            "value": assignment,
            "baseline": 70,
            "contribution": points_below(assignment, 70, 22),
            "reason": "과제 제출률이 70% 기준보다 낮습니다.",
            "recommended_action": "미제출 과제를 확인하고 제출 마감 일정을 다시 안내합니다.",
        },
        {
            "feature": "quiz_average",
            "label": "퀴즈 평균",
            "value": quiz,
            "baseline": 60,
            "contribution": points_below(quiz, 60, 18),
            "reason": "퀴즈 평균 점수가 60점 기준보다 낮습니다.",
            "recommended_action": "오답 유형을 확인하고 보충 학습 자료를 제공합니다.",
        },
        {
            "feature": "login_count",
            "label": "LMS 접속 횟수",
            "value": login_count,
            "baseline": 12,
            "contribution": points_below(login_count, 12, 12),
            "reason": "LMS 접속 횟수가 기준보다 적습니다.",
            "recommended_action": "LMS 접속 알림과 주간 학습 체크인을 설정합니다.",
        },
        {
            "feature": "video_watch_time",
            "label": "강의 시청 시간",
            "value": video_watch_time,
            "baseline": 180,
            "contribution": points_below(video_watch_time, 180, 10),
            "reason": "강의 시청 시간이 기준보다 부족합니다.",
            "recommended_action": "핵심 강의 구간과 복습 순서를 짧게 안내합니다.",
        },
        {
            "feature": "forum_activity",
            "label": "게시판 활동",
            "value": forum_activity,
            "baseline": 2,
            "contribution": points_below(forum_activity, 2, 6),
            "reason": "게시판 질문 또는 답변 활동이 적습니다.",
            "recommended_action": "질문 템플릿을 제공하고 조교 피드백을 연결합니다.",
        },
        {
            "feature": "previous_grade",
            "label": "이전 성적",
            "value": previous_grade,
            "baseline": "A/B",
            "contribution": GRADE_RISK_POINTS.get(previous_grade, 4.0),
            "reason": "이전 성적이 현재 과목 적응 위험을 높일 수 있습니다.",
            "recommended_action": "선수 지식 점검과 기초 개념 보강을 진행합니다.",
        },
    ]

    for factor in factors:
        factor["contribution"] = round(float(factor["contribution"]), 1)
        if factor["contribution"] <= 0:
            factor["reason"] = "기준을 충족했습니다."

    return sorted(factors, key=lambda item: item["contribution"], reverse=True)


def build_risk_reasons(breakdown: list[dict[str, Any]]) -> list[str]:
    reasons = [factor["reason"] for factor in breakdown if factor["contribution"] > 0]
    return reasons or ["학습 활동 지표에서 큰 위험 요인이 감지되지 않음"]


def classify_risk(probability: float, threshold: float) -> dict[str, Any]:
    if probability >= CRITICAL_RISK_THRESHOLD:
        return {
            "risk_level": "Critical",
            "risk_level_label": "긴급 위험",
            "intervention_priority": 1,
            "notification_policy": "즉시 Slack과 이메일 알림",
        }
    if probability >= threshold:
        return {
            "risk_level": "High",
            "risk_level_label": "높은 위험",
            "intervention_priority": 2,
            "notification_policy": "교수자 이메일 알림 및 시트 기록",
        }
    if probability >= MEDIUM_RISK_THRESHOLD:
        return {
            "risk_level": "Medium",
            "risk_level_label": "관찰 필요",
            "intervention_priority": 3,
            "notification_policy": "시트 기록 후 추이 모니터링",
        }
    return {
        "risk_level": "Low",
        "risk_level_label": "낮은 위험",
        "intervention_priority": 4,
        "notification_policy": "정기 모니터링",
    }


def build_intervention_plan(
    probability: float,
    risk_info: dict[str, Any],
    top_factors: list[dict[str, Any]],
) -> dict[str, Any]:
    risk_level = risk_info["risk_level"]
    factor_actions = [factor["recommended_action"] for factor in top_factors]

    if risk_level == "Critical":
        base_actions = ["당일 개별 상담 일정 확정", "결석 및 미제출 과제 현황 즉시 확인", "1주 단위 학습 회복 계획 수립"]
        owner = "교수자 + 조교"
        due = "24시간 이내"
    elif risk_level == "High":
        base_actions = ["개별 상담 진행", "미제출 과제 확인 및 제출 독려", "보충 학습 자료 안내"]
        owner = "담당 조교"
        due = "3일 이내"
    elif risk_level == "Medium":
        base_actions = ["학습 활동 추이 모니터링", "출석 및 과제 제출 상태 점검"]
        owner = "조교"
        due = "1주 이내"
    else:
        base_actions = ["정기 모니터링 유지"]
        owner = "시스템"
        due = "다음 정기 점검"

    actions = list(dict.fromkeys(base_actions + factor_actions))
    return {
        "owner": owner,
        "due": due,
        "probability_band": f"{round(probability * 100, 1)}%",
        "actions": actions,
    }


def build_prediction(row: dict[str, Any], probability: float) -> dict[str, Any]:
    threshold = float(metadata.get("threshold", DEFAULT_THRESHOLD))
    is_risky = probability >= threshold
    breakdown = build_risk_score_breakdown(row)
    positive_factors = [factor for factor in breakdown if factor["contribution"] > 0]
    top_factors = positive_factors[:3]
    risk_score = round(min(sum(factor["contribution"] for factor in breakdown), 100.0), 1)
    risk_info = classify_risk(probability, threshold)
    intervention_plan = build_intervention_plan(probability, risk_info, top_factors)

    return {
        "student_id": row.get("student_id"),
        "dropout_probability": round(probability, 4),
        "dropout_probability_percent": round(probability * 100, 1),
        "prediction": "중도 포기 위험" if is_risky else "정상 수강 가능성 높음",
        "label": 1 if is_risky else 0,
        "threshold": threshold,
        **risk_info,
        "risk_score": risk_score,
        "risk_score_breakdown": breakdown,
        "top_risk_factors": top_factors,
        "risk_reasons": build_risk_reasons(breakdown),
        "recommended_actions": intervention_plan["actions"],
        "intervention_plan": intervention_plan,
        "explanation_note": "risk_score는 모델 확률을 보완하기 위한 규칙 기반 설명 점수입니다.",
    }


def predict_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ensure_ready()

    categorical_features = get_categorical_features()
    raw_numeric_features = get_raw_numeric_features()
    feature_order = get_feature_order()
    required_features = categorical_features + raw_numeric_features
    normalized_rows = [get_payload_dict(row) for row in rows]
    missing_by_row = [
        sorted(set(required_features) - set(row.keys()))
        for row in normalized_rows
    ]
    missing_by_row = [missing for missing in missing_by_row if missing]
    if missing_by_row:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "예측에 필요한 실제 데이터 컬럼이 누락되었습니다.",
                "missing_columns_first_invalid_row": missing_by_row[0],
                "required_columns": required_features,
            },
        )

    df = pd.DataFrame([{key: row[key] for key in required_features} for row in normalized_rows])
    df = add_engineered_features_to_frame(df)
    transformed = preprocessor.transform(df)
    member_probabilities = [member.predict(transformed, verbose=0).ravel() for member in models]
    probabilities = np.mean(np.vstack(member_probabilities), axis=0)

    return [build_prediction(row, float(probability)) for row, probability in zip(normalized_rows, probabilities)]


def build_batch_summary(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    risk_counts = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0}
    factor_counts: dict[str, int] = {}

    for prediction in predictions:
        risk_counts[prediction["risk_level"]] += 1
        for factor in prediction["top_risk_factors"]:
            label = str(factor["label"])
            factor_counts[label] = factor_counts.get(label, 0) + 1

    top_factor_counts = [
        {"factor": factor, "count": count}
        for factor, count in sorted(factor_counts.items(), key=lambda item: item[1], reverse=True)
    ]
    average_probability = sum(prediction["dropout_probability"] for prediction in predictions) / len(predictions)

    return {
        "total_students": len(predictions),
        "alert_count": sum(1 for prediction in predictions if prediction["label"] == 1),
        "critical_count": risk_counts["Critical"],
        "average_dropout_probability": round(average_probability, 4),
        "average_dropout_probability_percent": round(average_probability * 100, 1),
        "risk_counts": risk_counts,
        "top_factor_counts": top_factor_counts[:5],
    }


@app.post("/predict")
def predict(payload: dict[str, Any]) -> dict[str, Any]:
    row = get_payload_dict(payload)
    return predict_rows([row])[0]


@app.post("/predict-batch")
def predict_batch(payload: BatchPredictionRequest) -> dict[str, Any]:
    rows = [get_payload_dict(student) for student in payload.students]
    predictions = predict_rows(rows)
    return {
        "summary": build_batch_summary(predictions),
        "predictions": predictions,
    }
