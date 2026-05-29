from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrainingConfig:
    dataset_path: Path
    target_column: str
    test_size: float
    random_state: int
    cv_folds: int
    quality_bins: list[float]
    quality_labels: list[str]
    numeric_features: list[str]
    categorical_features: list[str]
    output_dir: Path

    @property
    def feature_columns(self) -> list[str]:
        return self.numeric_features + self.categorical_features


def load_config(path: str | Path = "configs/training_config.json") -> TrainingConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        raw: dict[str, Any] = json.load(file)

    return TrainingConfig(
        dataset_path=Path(raw["dataset_path"]),
        target_column=raw["target_column"],
        test_size=float(raw["test_size"]),
        random_state=int(raw["random_state"]),
        cv_folds=int(raw["cv_folds"]),
        quality_bins=[float(value) for value in raw["quality_bins"]],
        quality_labels=list(raw["quality_labels"]),
        numeric_features=list(raw["numeric_features"]),
        categorical_features=list(raw["categorical_features"]),
        output_dir=Path(raw["output_dir"]),
    )
