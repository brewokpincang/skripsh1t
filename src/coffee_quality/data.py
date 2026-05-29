from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from coffee_quality.config import TrainingConfig


@dataclass(frozen=True)
class PreparedData:
    raw: pd.DataFrame
    cleaned: pd.DataFrame
    features: pd.DataFrame
    target: pd.Series
    target_score: pd.Series
    dropped_rows: int


def load_raw_dataset(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        encoding="utf-8-sig",
        na_values=["", "NA", "NaN", "nan", "None"],
        keep_default_na=True,
        on_bad_lines="skip",
    )


def create_quality_label(scores: pd.Series, config: TrainingConfig) -> pd.Series:
    labels = pd.cut(
        scores,
        bins=config.quality_bins,
        labels=config.quality_labels,
        include_lowest=True,
        right=False,
    )
    return labels.astype("object")


def prepare_dataset(config: TrainingConfig) -> PreparedData:
    raw = load_raw_dataset(config.dataset_path)
    data = raw.copy()

    required_columns = set(config.feature_columns + [config.target_column, "Species"])
    missing_columns = sorted(required_columns.difference(data.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    for column in config.numeric_features + [config.target_column]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    score_columns = [
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

    valid_scores = data[score_columns].apply(lambda col: col.between(0, 10) | col.isna())
    data = data[
        (data["Species"].eq("Arabica"))
        & data[config.target_column].notna()
        & data[config.target_column].between(1, 100)
        & valid_scores.all(axis=1)
    ].copy()

    data["quality_label"] = create_quality_label(data[config.target_column], config)
    data = data[data["quality_label"].notna()].copy()

    for column in config.categorical_features:
        data[column] = (
            data[column]
            .astype("object")
            .where(data[column].notna(), "Unknown")
            .astype(str)
            .str.strip()
            .replace({"": "Unknown"})
        )

    data = data.replace([np.inf, -np.inf], np.nan)
    features = data[config.feature_columns].copy()
    target = data["quality_label"].copy()
    target_score = data[config.target_column].copy()

    return PreparedData(
        raw=raw,
        cleaned=data,
        features=features,
        target=target,
        target_score=target_score,
        dropped_rows=len(raw) - len(data),
    )
