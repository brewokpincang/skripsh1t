from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from coffee_quality.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict coffee quality from a JSON sample.")
    parser.add_argument("--model", default="artifacts/models/random_forest_model.joblib", help="Path to model joblib.")
    parser.add_argument("--config", default="configs/training_config.json", help="Path to training config JSON.")
    parser.add_argument("--sample", default="examples/sample_input.json", help="Path to sample JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    model = joblib.load(args.model)

    with Path(args.sample).open("r", encoding="utf-8") as file:
        payload = json.load(file)

    sample = pd.DataFrame([payload], columns=config.feature_columns)
    prediction = model.predict(sample)[0]
    probabilities = model.predict_proba(sample)[0] if hasattr(model, "predict_proba") else None

    print(f"Prediksi mutu kopi: {prediction}")
    if probabilities is not None:
        for label, probability in zip(model.classes_, probabilities):
            print(f"- {label}: {probability:.4f}")


if __name__ == "__main__":
    main()
