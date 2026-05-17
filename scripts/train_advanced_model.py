from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from train_model import (
    CATEGORICAL_FEATURES,
    DATA_PATH,
    DATASET_SOURCE,
    DROP_COLUMNS,
    ENGINEERED_NUMERIC_FEATURES,
    FEATURE_ENGINEERING_DESCRIPTIONS,
    METADATA_PATH,
    MODEL_DIR,
    MODEL_PATH,
    POSITIVE_TARGET,
    PREPROCESSOR_PATH,
    RAW_NUMERIC_FEATURES,
    TARGET_COLUMN,
    build_preprocessor,
    get_class_weight,
    load_dataset,
    to_dense_float32,
    tune_threshold,
)


ENSEMBLE_DIR = MODEL_DIR / "ensemble"

MODEL_PRESETS: dict[str, dict[str, Any]] = {
    "compact": {
        "hidden_units": [64, 32, 16],
        "dropout": [0.30, 0.15, 0.00],
        "batch_norm": False,
        "l2": 0.0,
    },
    "regularized": {
        "hidden_units": [96, 48, 24],
        "dropout": [0.25, 0.15, 0.05],
        "batch_norm": True,
        "l2": 1e-4,
    },
    "wide": {
        "hidden_units": [128, 64, 32],
        "dropout": [0.35, 0.20, 0.10],
        "batch_norm": True,
        "l2": 5e-5,
    },
    "deep": {
        "hidden_units": [128, 96, 64, 32, 16],
        "dropout": [0.30, 0.25, 0.20, 0.10, 0.00],
        "batch_norm": True,
        "l2": 1e-4,
    },
}


def build_advanced_model(input_dim: int, learning_rate: float, preset: str) -> tf.keras.Model:
    config = MODEL_PRESETS[preset]
    regularizer = tf.keras.regularizers.l2(config["l2"]) if config["l2"] else None

    layers: list[tf.keras.layers.Layer] = [tf.keras.layers.Input(shape=(input_dim,))]
    for index, units in enumerate(config["hidden_units"]):
        layers.append(tf.keras.layers.Dense(units, kernel_regularizer=regularizer))
        if config["batch_norm"]:
            layers.append(tf.keras.layers.BatchNormalization())
        layers.append(tf.keras.layers.Activation("relu"))
        dropout_rate = config["dropout"][index]
        if dropout_rate > 0:
            layers.append(tf.keras.layers.Dropout(dropout_rate))
    layers.append(tf.keras.layers.Dense(1, activation="sigmoid"))

    model = tf.keras.Sequential(layers)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
        ],
    )
    return model


def evaluate_predictions(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "classification_report": classification_report(y_true, predictions, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
    }


def train_member(
    member_index: int,
    seed: int,
    input_dim: int,
    X_train_processed: np.ndarray,
    y_train_array: np.ndarray,
    X_val_processed: np.ndarray,
    y_val_array: np.ndarray,
    args: argparse.Namespace,
    class_weight: dict[int, float] | None,
) -> tuple[tf.keras.Model, dict[str, Any]]:
    tf.keras.backend.clear_session()
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    model = build_advanced_model(input_dim=input_dim, learning_rate=args.learning_rate, preset=args.preset)
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

    history = model.fit(
        X_train_processed,
        y_train_array,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_val_processed, y_val_array),
        callbacks=[early_stopping, reduce_lr],
        class_weight=class_weight,
        verbose=args.verbose,
    )

    val_probabilities = model.predict(X_val_processed, verbose=0).ravel()
    val_metrics = evaluate_predictions(y_val_array, val_probabilities, threshold=0.5)
    member_info = {
        "member": member_index,
        "seed": seed,
        "epochs_ran": len(history.history.get("loss", [])),
        "best_val_auc": float(max(history.history.get("val_auc", [0.0]))),
        "best_val_pr_auc": float(max(history.history.get("val_pr_auc", [0.0]))),
        "validation_f1_at_0_5": val_metrics["f1_score"],
    }
    return model, member_info


def train(args: argparse.Namespace) -> dict[str, Any]:
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
    X_train_processed = to_dense_float32(preprocessor.fit_transform(X_train))
    X_val_processed = to_dense_float32(preprocessor.transform(X_val))
    X_test_processed = to_dense_float32(preprocessor.transform(X_test))
    y_train_array = y_train.to_numpy(dtype=np.float32)
    y_val_array = y_val.to_numpy(dtype=np.float32)

    class_weight = get_class_weight(y_train) if args.class_weight else None
    models: list[tf.keras.Model] = []
    member_details: list[dict[str, Any]] = []
    val_probability_members: list[np.ndarray] = []
    test_probability_members: list[np.ndarray] = []

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)

    for member_index in range(args.ensemble_size):
        member_seed = args.seed + member_index * 17
        model, member_info = train_member(
            member_index=member_index + 1,
            seed=member_seed,
            input_dim=X_train_processed.shape[1],
            X_train_processed=X_train_processed,
            y_train_array=y_train_array,
            X_val_processed=X_val_processed,
            y_val_array=y_val_array,
            args=args,
            class_weight=class_weight,
        )
        models.append(model)
        member_details.append(member_info)
        val_probability_members.append(model.predict(X_val_processed, verbose=0).ravel())
        test_probability_members.append(model.predict(X_test_processed, verbose=0).ravel())

    val_probabilities = np.mean(np.vstack(val_probability_members), axis=0)
    test_probabilities = np.mean(np.vstack(test_probability_members), axis=0)
    tuned_threshold_metrics = tune_threshold(y_val, val_probabilities)
    threshold = tuned_threshold_metrics["threshold"] if args.auto_threshold else float(args.threshold)

    default_metrics = evaluate_predictions(y_test, test_probabilities, threshold=0.5)
    final_metrics = evaluate_predictions(y_test, test_probabilities, threshold=threshold)

    model_paths: list[str] = []
    for member_index, model in enumerate(models, start=1):
        path = ENSEMBLE_DIR / f"edurisk_{args.preset}_{member_index:02d}.keras"
        model.save(path)
        model_paths.append(str(path.relative_to(MODEL_DIR.parent)).replace("\\", "/"))
        if member_index == 1:
            model.save(MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)

    metadata: dict[str, Any] = {
        "threshold": threshold,
        "threshold_strategy": "validation_f1_tuned" if args.auto_threshold else "manual",
        "validation_threshold_tuning": tuned_threshold_metrics,
        "model_strategy": "tensorflow_probability_ensemble",
        "ensemble_size": args.ensemble_size,
        "ensemble_model_paths": model_paths,
        "model_preset": args.preset,
        "model_preset_config": MODEL_PRESETS[args.preset],
        "ensemble_members": member_details,
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
            "default_threshold_0_5": {
                key: value
                for key, value in default_metrics.items()
                if key not in {"classification_report", "confusion_matrix"}
            },
            "threshold_accuracy": final_metrics["accuracy"],
            "threshold_precision": final_metrics["precision"],
            "threshold_recall": final_metrics["recall"],
            "threshold_f1_score": final_metrics["f1_score"],
            "roc_auc": final_metrics["roc_auc"],
            "average_precision": final_metrics["average_precision"],
            "brier_score": final_metrics["brier_score"],
            "classification_report": final_metrics["classification_report"],
            "confusion_matrix": final_metrics["confusion_matrix"],
        },
    }

    METADATA_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an advanced TensorFlow ensemble for EduRiskAI.")
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--auto-threshold", action="store_true")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--class-weight", action="store_true")
    parser.add_argument("--use-engineered-features", action="store_true")
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--preset", choices=sorted(MODEL_PRESETS), default="regularized")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", type=int, choices=[0, 1, 2], default=0)
    return parser.parse_args()


def main() -> None:
    metadata = train(parse_args())
    metrics = metadata["metrics"]
    print("\nSaved advanced artifacts")
    print(f"- strategy: {metadata['model_strategy']}")
    print(f"- ensemble size: {metadata['ensemble_size']}")
    print(f"- preset: {metadata['model_preset']}")
    print(f"- preprocessor: {PREPROCESSOR_PATH}")
    print(f"- metadata: {METADATA_PATH}")
    print("\nEvaluation")
    print(f"- threshold: {metadata['threshold']:.3f} ({metadata['threshold_strategy']})")
    print(f"- accuracy: {metrics['threshold_accuracy']:.4f}")
    print(f"- precision: {metrics['threshold_precision']:.4f}")
    print(f"- recall: {metrics['threshold_recall']:.4f}")
    print(f"- f1-score: {metrics['threshold_f1_score']:.4f}")
    print(f"- roc-auc: {metrics['roc_auc']:.4f}")
    print(f"- average precision: {metrics['average_precision']:.4f}")
    print(f"- brier score: {metrics['brier_score']:.4f}")
    print(f"- confusion matrix: {metrics['confusion_matrix']}")


if __name__ == "__main__":
    main()
