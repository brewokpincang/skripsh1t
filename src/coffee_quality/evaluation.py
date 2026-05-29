from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_classifier(model: Any, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    y_pred = model.predict(x_test)
    labels = list(model.classes_)
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "labels": labels,
        "classification_report": classification_report(
            y_test,
            y_pred,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=labels).tolist(),
    }

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x_test)
        try:
            metrics["roc_auc_ovr_weighted"] = float(
                roc_auc_score(y_test, probabilities, labels=labels, multi_class="ovr", average="weighted")
            )
            metrics["roc_auc_ovr_macro"] = float(
                roc_auc_score(y_test, probabilities, labels=labels, multi_class="ovr", average="macro")
            )
        except ValueError:
            metrics["roc_auc_ovr_weighted"] = None
            metrics["roc_auc_ovr_macro"] = None

    return metrics


def save_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def save_model(model: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def save_predictions(
    model_name: str,
    model: Any,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    probabilities = model.predict_proba(x_test) if hasattr(model, "predict_proba") else None
    output = x_test.copy()
    output["actual_quality"] = y_test.values
    output["predicted_quality"] = model.predict(x_test)
    output["model"] = model_name

    if probabilities is not None:
        for index, label in enumerate(model.classes_):
            output[f"probability_{label}"] = np.round(probabilities[:, index], 6)

    output.to_csv(path, index=False)
