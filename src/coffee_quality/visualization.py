from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.preprocessing import label_binarize

from coffee_quality.config import TrainingConfig


sns.set_theme(style="whitegrid", palette="Set2")


def _save_current(path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_target_distribution(target: pd.Series, path: str | Path) -> None:
    plt.figure(figsize=(8, 5))
    order = target.value_counts().index
    ax = sns.countplot(x=target, order=order)
    ax.set_title("Distribusi Kelas Mutu Kopi Arabika")
    ax.set_xlabel("Kelas Mutu")
    ax.set_ylabel("Jumlah Sampel")
    for container in ax.containers:
        ax.bar_label(container)
    _save_current(path)


def plot_score_distribution(data: pd.DataFrame, target_column: str, path: str | Path) -> None:
    plt.figure(figsize=(9, 5))
    ax = sns.histplot(data=data, x=target_column, hue="quality_label", kde=True, bins=30, multiple="stack")
    ax.set_title("Distribusi Total Cup Points Berdasarkan Kelas Mutu")
    ax.set_xlabel("Total Cup Points")
    ax.set_ylabel("Jumlah Sampel")
    _save_current(path)


def plot_correlation_heatmap(data: pd.DataFrame, numeric_features: list[str], path: str | Path) -> None:
    available = [column for column in numeric_features if column in data.columns]
    corr = data[available].corr(numeric_only=True)
    plt.figure(figsize=(12, 9))
    ax = sns.heatmap(corr, cmap="vlag", center=0, annot=False, linewidths=0.3)
    ax.set_title("Korelasi Fitur Numerik")
    _save_current(path)


def plot_quality_boxplots(data: pd.DataFrame, features: list[str], path: str | Path) -> None:
    selected = [feature for feature in features if feature in data.columns][:10]
    melted = data.melt(id_vars="quality_label", value_vars=selected, var_name="feature", value_name="score")
    plt.figure(figsize=(13, 7))
    ax = sns.boxplot(data=melted, x="feature", y="score", hue="quality_label")
    ax.set_title("Sebaran Fitur Penilaian Sensorik per Kelas Mutu")
    ax.set_xlabel("Fitur")
    ax.set_ylabel("Nilai")
    ax.tick_params(axis="x", rotation=35)
    _save_current(path)


def plot_confusion_matrix(model_name: str, labels: list[str], matrix: list[list[int]], path: str | Path) -> None:
    plt.figure(figsize=(7, 6))
    display = ConfusionMatrixDisplay(confusion_matrix=np.array(matrix), display_labels=labels)
    display.plot(cmap="Blues", values_format="d", ax=plt.gca(), colorbar=False)
    plt.title(f"Confusion Matrix - {model_name}")
    _save_current(path)


def plot_roc_curves(model_name: str, model: Any, x_test: pd.DataFrame, y_test: pd.Series, path: str | Path) -> None:
    if not hasattr(model, "predict_proba"):
        return

    classes = list(model.classes_)
    y_binary = label_binarize(y_test, classes=classes)
    probabilities = model.predict_proba(x_test)

    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    for class_index, class_name in enumerate(classes):
        RocCurveDisplay.from_predictions(
            y_binary[:, class_index],
            probabilities[:, class_index],
            name=class_name,
            ax=ax,
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_title(f"ROC One-vs-Rest - {model_name}")
    _save_current(path)


def plot_feature_importance(model_name: str, model: Any, path: str | Path, top_n: int = 20) -> None:
    classifier = model.named_steps.get("model")
    preprocessor = model.named_steps.get("preprocessor")
    if classifier is None or preprocessor is None or not hasattr(classifier, "feature_importances_"):
        return

    names = preprocessor.get_feature_names_out()
    importances = classifier.feature_importances_
    frame = (
        pd.DataFrame({"feature": names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(top_n)
    )

    plt.figure(figsize=(10, 7))
    ax = sns.barplot(data=frame, x="importance", y="feature", color="#4C78A8")
    ax.set_title(f"Top {top_n} Feature Importance - {model_name}")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    _save_current(path)


def plot_permutation_importance(
    model_name: str,
    model: Any,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    path: str | Path,
    random_state: int,
    top_n: int = 15,
) -> None:
    result = permutation_importance(
        model,
        x_test,
        y_test,
        scoring="f1_macro",
        n_repeats=8,
        random_state=random_state,
        n_jobs=1,
    )
    frame = (
        pd.DataFrame({"feature": x_test.columns, "importance": result.importances_mean})
        .sort_values("importance", ascending=False)
        .head(top_n)
    )
    plt.figure(figsize=(9, 6))
    ax = sns.barplot(data=frame, x="importance", y="feature", color="#59A14F")
    ax.set_title(f"Permutation Importance - {model_name}")
    ax.set_xlabel("Mean F1 Macro Decrease")
    ax.set_ylabel("Feature")
    _save_current(path)


def create_exploratory_figures(data: pd.DataFrame, target: pd.Series, config: TrainingConfig) -> None:
    figure_dir = config.output_dir / "figures"
    plot_target_distribution(target, figure_dir / "01_target_distribution.png")
    plot_score_distribution(data, config.target_column, figure_dir / "02_total_cup_points_distribution.png")
    plot_correlation_heatmap(data, config.numeric_features, figure_dir / "03_numeric_correlation_heatmap.png")
    plot_quality_boxplots(data, config.numeric_features, figure_dir / "04_sensoric_boxplots_by_quality.png")
