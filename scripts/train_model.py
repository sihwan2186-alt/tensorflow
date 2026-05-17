from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "student_dropout_real.csv"
MODEL_DIR = ROOT_DIR / "model"
MODEL_PATH = MODEL_DIR / "edurisk_model.keras"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"

CATEGORICAL_FEATURES = [
    "Marital status",
    "Application mode",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International",
]
RAW_NUMERIC_FEATURES = [
    "Application order",
    "Previous qualification (grade)",
    "Admission grade",
    "Age at enrollment",
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
    "Unemployment rate",
    "Inflation rate",
    "GDP",
]
ENGINEERED_NUMERIC_FEATURES = [
    "first_sem_approval_rate",
    "second_sem_approval_rate",
    "first_sem_failure_count",
    "second_sem_failure_count",
    "grade_delta_second_minus_first",
    "approved_units_total",
    "evaluation_load_total",
    "without_evaluations_total",
    "financial_risk_score",
    "academic_momentum",
]
FEATURE_ENGINEERING_DESCRIPTIONS = {
    "first_sem_approval_rate": "1학기 승인 과목 수 / 1학기 수강 과목 수",
    "second_sem_approval_rate": "2학기 승인 과목 수 / 2학기 수강 과목 수",
    "first_sem_failure_count": "1학기 수강 과목 수 - 1학기 승인 과목 수",
    "second_sem_failure_count": "2학기 수강 과목 수 - 2학기 승인 과목 수",
    "grade_delta_second_minus_first": "2학기 평균 성적 - 1학기 평균 성적",
    "approved_units_total": "1학기와 2학기 승인 과목 수 합계",
    "evaluation_load_total": "1학기와 2학기 평가 횟수 합계",
    "without_evaluations_total": "1학기와 2학기 미평가 과목 수 합계",
    "financial_risk_score": "채무 여부 + 등록금 미납 여부",
    "academic_momentum": "2학기 승인율 - 1학기 승인율",
}
TARGET_COLUMN = "Target"
POSITIVE_TARGET = "Dropout"
DROP_COLUMNS: list[str] = []

DATASET_SOURCE = {
    "name": "Predict Students' Dropout and Academic Success",
    "repository": "UCI Machine Learning Repository",
    "dataset_id": 697,
    "url": "https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success",
    "doi": "10.24432/C5MC89",
    "license": "CC BY 4.0",
}


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def get_numeric_features(use_engineered_features: bool) -> list[str]:
    if use_engineered_features:
        return RAW_NUMERIC_FEATURES + ENGINEERED_NUMERIC_FEATURES
    return RAW_NUMERIC_FEATURES


def load_dataset(path: Path, use_engineered_features: bool) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 파일이 없습니다. UCI 실제 데이터 student_dropout_real.csv를 data 폴더에 준비하세요."
        )

    df = pd.read_csv(path, sep=";")
    df.columns = [column.strip().strip('"') for column in df.columns]
    missing_columns = set(CATEGORICAL_FEATURES + RAW_NUMERIC_FEATURES + [TARGET_COLUMN]) - set(df.columns)
    if missing_columns:
        raise ValueError(f"데이터셋에 필요한 컬럼이 없습니다: {sorted(missing_columns)}")

    df = df.drop(columns=[column for column in DROP_COLUMNS if column in df.columns])
    if use_engineered_features:
        df = add_engineered_features(df)
    numeric_features = get_numeric_features(use_engineered_features)
    X = df[CATEGORICAL_FEATURES + numeric_features]
    y = (df[TARGET_COLUMN] == POSITIVE_TARGET).astype(int)
    return X, y, numeric_features


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    return (numerator / denominator).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    first_enrolled = df["Curricular units 1st sem (enrolled)"]
    first_approved = df["Curricular units 1st sem (approved)"]
    second_enrolled = df["Curricular units 2nd sem (enrolled)"]
    second_approved = df["Curricular units 2nd sem (approved)"]
    first_grade = df["Curricular units 1st sem (grade)"]
    second_grade = df["Curricular units 2nd sem (grade)"]

    df["first_sem_approval_rate"] = safe_divide(first_approved, first_enrolled)
    df["second_sem_approval_rate"] = safe_divide(second_approved, second_enrolled)
    df["first_sem_failure_count"] = (first_enrolled - first_approved).clip(lower=0)
    df["second_sem_failure_count"] = (second_enrolled - second_approved).clip(lower=0)
    df["grade_delta_second_minus_first"] = second_grade - first_grade
    df["approved_units_total"] = first_approved + second_approved
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


def build_preprocessor(numeric_features: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_features),
            ("categorical", make_one_hot_encoder(), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def build_model(input_dim: int, learning_rate: float) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def tune_threshold(y_true: pd.Series, probabilities: np.ndarray) -> dict[str, float]:
    best: dict[str, float] | None = None
    for threshold in np.linspace(0.2, 0.8, 121):
        predictions = (probabilities >= threshold).astype(int)
        metrics = {
            "threshold": float(threshold),
            "accuracy": float(accuracy_score(y_true, predictions)),
            "precision": float(precision_score(y_true, predictions, zero_division=0)),
            "recall": float(recall_score(y_true, predictions, zero_division=0)),
            "f1_score": float(f1_score(y_true, predictions, zero_division=0)),
        }
        if best is None:
            best = metrics
            continue

        current_key = (metrics["f1_score"], metrics["accuracy"], metrics["recall"])
        best_key = (best["f1_score"], best["accuracy"], best["recall"])
        if current_key > best_key:
            best = metrics

    if best is None:
        raise ValueError("threshold tuning failed")
    best["threshold"] = round(best["threshold"], 3)
    return best


def get_class_weight(y_train: pd.Series) -> dict[int, float]:
    classes = np.array([0, 1])
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    return {int(label): float(weight) for label, weight in zip(classes, weights)}


def train(args: argparse.Namespace) -> dict[str, object]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    X, y, numeric_features = load_dataset(args.data, args.use_engineered_features)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=args.validation_size,
        random_state=args.seed,
        stratify=y_train_val,
    )

    preprocessor = build_preprocessor(numeric_features)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    model = build_model(input_dim=X_train_processed.shape[1], learning_rate=args.learning_rate)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=args.patience,
        restore_best_weights=True,
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_auc",
        mode="max",
        factor=0.5,
        patience=max(2, args.patience // 2),
        min_lr=1e-5,
    )

    fit_kwargs: dict[str, object] = {}
    class_weight = get_class_weight(y_train) if args.class_weight else None
    if class_weight is not None:
        fit_kwargs["class_weight"] = class_weight

    history = model.fit(
        X_train_processed,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_val_processed, y_val),
        callbacks=[early_stopping, reduce_lr],
        verbose=1,
        **fit_kwargs,
    )

    evaluation_values = model.evaluate(X_test_processed, y_test, verbose=0)
    test_loss = float(evaluation_values[0])
    val_probabilities = model.predict(X_val_processed, verbose=0).ravel()
    probabilities = model.predict(X_test_processed, verbose=0).ravel()
    tuned_threshold_metrics = tune_threshold(y_val, val_probabilities)
    threshold = tuned_threshold_metrics["threshold"] if args.auto_threshold else float(args.threshold)
    predictions = (probabilities >= threshold).astype(int)
    default_predictions = (probabilities >= 0.5).astype(int)

    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_test, predictions).tolist()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)

    metadata = {
        "threshold": threshold,
        "threshold_strategy": "validation_f1_tuned" if args.auto_threshold else "manual",
        "validation_threshold_tuning": tuned_threshold_metrics,
        "target_column": TARGET_COLUMN,
        "positive_target": POSITIVE_TARGET,
        "drop_columns": DROP_COLUMNS,
        "categorical_features": CATEGORICAL_FEATURES,
        "raw_numeric_features": RAW_NUMERIC_FEATURES,
        "engineered_numeric_features": ENGINEERED_NUMERIC_FEATURES if args.use_engineered_features else [],
        "numeric_features": numeric_features,
        "feature_engineering": FEATURE_ENGINEERING_DESCRIPTIONS if args.use_engineered_features else {},
        "available_feature_engineering": FEATURE_ENGINEERING_DESCRIPTIONS,
        "dataset_source": DATASET_SOURCE,
        "train_rows": int(len(X_train)),
        "validation_rows": int(len(X_val)),
        "test_rows": int(len(X_test)),
        "positive_label_rate": float(y.mean()),
        "class_weight": class_weight,
        "metrics": {
            "test_loss": test_loss,
            "default_threshold_0_5": {
                "accuracy": float(accuracy_score(y_test, default_predictions)),
                "precision": float(precision_score(y_test, default_predictions, zero_division=0)),
                "recall": float(recall_score(y_test, default_predictions, zero_division=0)),
                "f1_score": float(f1_score(y_test, default_predictions, zero_division=0)),
            },
            "threshold_accuracy": float(accuracy_score(y_test, predictions)),
            "threshold_precision": float(precision_score(y_test, predictions, zero_division=0)),
            "threshold_recall": float(recall_score(y_test, predictions, zero_division=0)),
            "threshold_f1_score": float(f1_score(y_test, predictions, zero_division=0)),
            "classification_report": report,
            "confusion_matrix": matrix,
        },
        "history": {
            "loss": [float(value) for value in history.history.get("loss", [])],
            "val_loss": [float(value) for value in history.history.get("val_loss", [])],
            "accuracy": [float(value) for value in history.history.get("accuracy", [])],
            "val_accuracy": [float(value) for value in history.history.get("val_accuracy", [])],
            "auc": [float(value) for value in history.history.get("auc", [])],
            "val_auc": [float(value) for value in history.history.get("val_auc", [])],
        },
    }

    METADATA_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the EduRiskAI TensorFlow dropout classifier.")
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--auto-threshold", action="store_true")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--class-weight", action="store_true")
    parser.add_argument("--use-engineered-features", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    metadata = train(parse_args())
    metrics = metadata["metrics"]
    print("\nSaved artifacts")
    print(f"- model: {MODEL_PATH}")
    print(f"- preprocessor: {PREPROCESSOR_PATH}")
    print(f"- metadata: {METADATA_PATH}")
    print("\nEvaluation")
    print(f"- threshold: {metadata['threshold']:.3f} ({metadata['threshold_strategy']})")
    print(f"- accuracy: {metrics['threshold_accuracy']:.4f}")
    print(f"- precision: {metrics['classification_report']['1']['precision']:.4f}")
    print(f"- recall: {metrics['classification_report']['1']['recall']:.4f}")
    print(f"- f1-score: {metrics['threshold_f1_score']:.4f}")
    print(f"- confusion matrix: {metrics['confusion_matrix']}")


if __name__ == "__main__":
    main()
