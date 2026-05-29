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

        dataset = metrics.get("dataset", {})
        models = metrics.get("models", {})
        best_model = max(
            comparison,
            key=lambda row: float(row.get("f1_macro", 0)),
            default={},
        )

        return render_template(
            "dashboard.html",
            config=config,
            dataset=dataset,
            models=models,
            comparison=comparison,
            best_model=best_model,
            figures=figures,
            sample=sample,
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
