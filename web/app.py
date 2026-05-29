from __future__ import annotations

import json
import os
import subprocess
import sys
from functools import wraps
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from flask import Flask, jsonify, redirect, render_template, request, send_from_directory, session, url_for

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from coffee_quality.config import load_config


USERNAME = "Risyad"
PASSWORD = "Telkom321!"


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.secret_key = os.environ.get("COFFEE_APP_SECRET", "coffee-quality-local-dev-secret")

    @app.context_processor
    def inject_globals() -> dict[str, Any]:
        return {"app_title": "Arabica Quality Intelligence"}

    @app.route("/", methods=["GET", "POST"])
    def login():
        if session.get("authenticated"):
            return redirect(url_for("dashboard"))

        error = None
        if request.method == "POST":
            username = request.form.get("username", "").strip()
            password = request.form.get("password", "")
            if username == USERNAME and password == PASSWORD:
                session["authenticated"] = True
                session["username"] = username
                return redirect(url_for("dashboard"))
            error = "Username atau password tidak sesuai."

        return render_template("login.html", error=error)

    @app.route("/logout", methods=["POST"])
    @login_required
    def logout():
        session.clear()
        return redirect(url_for("login"))

    @app.route("/dashboard")
    @login_required
    def dashboard():
        config = load_config(ROOT / "configs" / "training_config.json")
        metrics = read_json(ROOT / "artifacts" / "metrics" / "training_summary.json")
        comparison = read_model_comparison(ROOT / "artifacts" / "metrics" / "model_comparison.csv")
        figures = list_figures(ROOT / "artifacts" / "figures")
        sample = read_json(ROOT / "examples" / "sample_input.json")
        eda = build_eda_summary(config, metrics)
        process_steps = build_process_steps(config, metrics)
        model_details = build_model_details(comparison, metrics.get("models", {}))

        dataset = metrics.get("dataset", {})
        best_model = max(
            comparison,
            key=lambda row: float(row.get("f1_macro", 0)),
            default={},
        )

        return render_template(
            "dashboard.html",
            config=config,
            dataset=dataset,
            eda=eda,
            process_steps=process_steps,
            model_details=model_details,
            comparison=comparison,
            best_model=best_model,
            figures=figures,
            sample=sample,
            total_slides=6,
        )

    @app.route("/artifacts/<path:filename>")
    @login_required
    def artifact_file(filename: str):
        return send_from_directory(ROOT / "artifacts", filename)

    @app.route("/api/predict", methods=["POST"])
    @login_required
    def predict():
        payload = request.get_json(force=True)
        model_name = payload.pop("model_name", "svm")
        model_path = ROOT / "artifacts" / "models" / f"{model_name}_model.joblib"
        if not model_path.exists():
            return jsonify({"error": f"Model {model_name} belum tersedia."}), 404

        config = load_config(ROOT / "configs" / "training_config.json")
        model = joblib.load(model_path)
        frame = pd.DataFrame([payload], columns=config.feature_columns)
        prediction = model.predict(frame)[0]
        probabilities = {}
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(frame)[0]
            probabilities = {
                label: round(float(value), 6)
                for label, value in zip(model.classes_, proba)
            }

        return jsonify(
            {
                "model": model_name,
                "prediction": prediction,
                "probabilities": probabilities,
            }
        )

    @app.route("/api/retrain", methods=["POST"])
    @login_required
    def retrain():
        mode = request.get_json(silent=True) or {}
        rf_iterations = int(mode.get("rf_iterations", 10))
        command = [
            sys.executable,
            str(ROOT / "scripts" / "train_models.py"),
            "--skip-permutation",
            "--rf-iterations",
            str(rf_iterations),
        ]
        completed = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=240,
            check=False,
        )
        return jsonify(
            {
                "ok": completed.returncode == 0,
                "return_code": completed.returncode,
                "stdout": completed.stdout[-5000:],
                "stderr": completed.stderr[-5000:],
            }
        ), 200 if completed.returncode == 0 else 500

    return app


def login_required(handler):
    @wraps(handler)
    def wrapper(*args, **kwargs):
        if not session.get("authenticated"):
            return redirect(url_for("login"))
        return handler(*args, **kwargs)

    return wrapper


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def read_model_comparison(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    return frame.to_dict(orient="records")


def build_eda_summary(config: Any, metrics: dict[str, Any]) -> dict[str, Any]:
    raw_path = ROOT / config.dataset_path
    cleaned_path = ROOT / "artifacts" / "processed" / "coffee_cleaned_with_quality_label.csv"
    if not raw_path.exists() or not cleaned_path.exists():
        return {}

    raw = pd.read_csv(
        raw_path,
        encoding="utf-8-sig",
        na_values=["", "NA", "NaN", "nan", "None"],
        keep_default_na=True,
        on_bad_lines="skip",
    )
    cleaned = pd.read_csv(cleaned_path)
    target_column = config.target_column

    missing = raw.isna().sum().sort_values(ascending=False).head(8)
    score = pd.to_numeric(cleaned[target_column], errors="coerce").dropna()
    score_stats = {
        "min": round(float(score.min()), 2),
        "q1": round(float(score.quantile(0.25)), 2),
        "median": round(float(score.median()), 2),
        "mean": round(float(score.mean()), 2),
        "q3": round(float(score.quantile(0.75)), 2),
        "max": round(float(score.max()), 2),
        "std": round(float(score.std()), 2),
    }

    sensoric_features = [
        "Aroma",
        "Flavor",
        "Aftertaste",
        "Acidity",
        "Body",
        "Balance",
        "Uniformity",
        "Clean.Cup",
        "Sweetness",
        "Cupper.Points",
    ]
    class_profile = (
        cleaned.groupby("quality_label")[sensoric_features]
        .mean(numeric_only=True)
        .round(2)
        .reset_index()
        .to_dict(orient="records")
    )
    numeric_summary = (
        cleaned[sensoric_features]
        .agg(["mean", "min", "max"])
        .T.round(2)
        .reset_index()
        .rename(columns={"index": "feature"})
        .to_dict(orient="records")
    )

    correlation_frame = cleaned[config.numeric_features + [target_column]].apply(pd.to_numeric, errors="coerce")
    correlations = (
        correlation_frame.corr(numeric_only=True)[target_column]
        .drop(labels=[target_column], errors="ignore")
        .dropna()
        .sort_values(key=lambda series: series.abs(), ascending=False)
        .head(8)
    )

    label_counts = metrics.get("dataset", {}).get("target_distribution", {})
    total_clean = max(int(metrics.get("dataset", {}).get("clean_rows", len(cleaned))), 1)
    label_distribution = [
        {
            "label": label,
            "count": int(count),
            "percent": round((int(count) / total_clean) * 100, 2),
        }
        for label, count in label_counts.items()
    ]

    return {
        "missing_top": [{"column": column, "missing": int(value)} for column, value in missing.items()],
        "score_stats": score_stats,
        "top_countries": [
            {"country": country, "count": int(count)}
            for country, count in cleaned["Country.of.Origin"].value_counts().head(8).items()
        ],
        "numeric_summary": numeric_summary,
        "class_profile": class_profile,
        "target_correlations": [
            {
                "feature": feature,
                "correlation": round(float(value), 4),
                "abs_percent": round(abs(float(value)) * 100, 1),
            }
            for feature, value in correlations.items()
        ],
        "label_distribution": label_distribution,
        "feature_count": len(config.feature_columns),
        "numeric_feature_count": len(config.numeric_features),
        "categorical_feature_count": len(config.categorical_features),
    }


def build_process_steps(config: Any, metrics: dict[str, Any]) -> list[dict[str, str]]:
    dataset = metrics.get("dataset", {})
    return [
        {
            "title": "Load Dataset",
            "value": f"{dataset.get('raw_rows', 0)} baris mentah",
            "description": "Membaca Coffee-modified.csv dengan encoding utf-8-sig dan menangani baris CSV tidak valid.",
        },
        {
            "title": "Data Cleaning",
            "value": f"{dataset.get('clean_rows', 0)} baris valid",
            "description": "Filter Arabica, skor target valid, dan nilai sensorik di rentang 0 sampai 10.",
        },
        {
            "title": "EDA",
            "value": f"{len(config.numeric_features)} numerik + {len(config.categorical_features)} kategorikal",
            "description": "Menghitung distribusi label, missing value, statistik skor, korelasi, dan profil kelas.",
        },
        {
            "title": "Labeling Mutu",
            "value": "3 kelas",
            "description": "Total Cup Points dipakai untuk membentuk Below Specialty, Very Good, dan Excellent.",
        },
        {
            "title": "Preprocessing",
            "value": f"{len(config.feature_columns)} fitur",
            "description": "Median imputation untuk numerik, most frequent imputation dan OneHotEncoder untuk kategorikal.",
        },
        {
            "title": "Training & CV",
            "value": f"{config.cv_folds}-fold CV",
            "description": "SVM memakai GridSearchCV, Random Forest memakai RandomizedSearchCV, scoring utama F1 Macro.",
        },
        {
            "title": "Evaluation",
            "value": f"Test size {int(config.test_size * 100)}%",
            "description": "Mengukur Accuracy, Balanced Accuracy, Precision, Recall, F1 Score, ROC AUC, dan Confusion Matrix.",
        },
        {
            "title": "Deployment",
            "value": "Flask dashboard",
            "description": "Model joblib, metrik JSON/CSV, dan visual PNG disajikan ke website untuk analisis dan prediksi.",
        },
    ]


def build_model_details(comparison: list[dict[str, Any]], models: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for row in comparison:
        model_key = row.get("model", "")
        report = models.get(model_key, {}).get("classification_report", {})
        class_rows = []
        for label, values in report.items():
            if label in {"accuracy", "macro avg", "weighted avg"}:
                continue
            class_rows.append(
                {
                    "label": label,
                    "precision": float(values.get("precision", 0)),
                    "recall": float(values.get("recall", 0)),
                    "f1_score": float(values.get("f1-score", 0)),
                    "support": int(values.get("support", 0)),
                }
            )

        output.append(
            {
                "name": str(model_key).replace("_", " ").title(),
                "key": model_key,
                "accuracy": float(row.get("accuracy", 0)),
                "balanced_accuracy": float(row.get("balanced_accuracy", 0)),
                "f1_macro": float(row.get("f1_macro", 0)),
                "f1_weighted": float(row.get("f1_weighted", 0)),
                "roc_auc": float(row.get("roc_auc_ovr_weighted", 0)),
                "best_cv_f1_macro": float(row.get("best_cv_f1_macro", 0)),
                "best_params": row.get("best_params", ""),
                "classes": class_rows,
            }
        )
    return output


def list_figures(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    labels = {
        "01_target_distribution.png": "Distribusi Kelas",
        "02_total_cup_points_distribution.png": "Sebaran Total Cup Points",
        "03_numeric_correlation_heatmap.png": "Korelasi Fitur Numerik",
        "04_sensoric_boxplots_by_quality.png": "Boxplot Fitur Sensorik",
        "05_confusion_matrix_svm.png": "Confusion Matrix SVM",
        "05_confusion_matrix_random_forest.png": "Confusion Matrix Random Forest",
        "06_roc_curve_svm.png": "ROC Curve SVM",
        "06_roc_curve_random_forest.png": "ROC Curve Random Forest",
        "07_feature_importance_random_forest.png": "Feature Importance Random Forest",
        "08_permutation_importance_svm.png": "Permutation Importance SVM",
        "08_permutation_importance_random_forest.png": "Permutation Importance Random Forest",
    }
    figures = []
    for figure in sorted(path.glob("*.png")):
        figures.append(
            {
                "title": labels.get(figure.name, figure.stem.replace("_", " ").title()),
                "path": f"figures/{figure.name}",
            }
        )
    return figures


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
