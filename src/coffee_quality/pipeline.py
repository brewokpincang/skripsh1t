from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from coffee_quality.config import TrainingConfig


def build_preprocessor(config: TrainingConfig, scale_numeric: bool) -> ColumnTransformer:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(numeric_steps)
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", min_frequency=3)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, config.numeric_features),
            ("categorical", categorical_pipeline, config.categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_svm_pipeline(config: TrainingConfig) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(config, scale_numeric=True)),
            (
                "model",
                SVC(
                    probability=True,
                    class_weight="balanced",
                    random_state=config.random_state,
                ),
            ),
        ]
    )


def build_random_forest_pipeline(config: TrainingConfig) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(config, scale_numeric=False)),
            (
                "model",
                RandomForestClassifier(
                    random_state=config.random_state,
                    n_jobs=1,
                    class_weight="balanced_subsample",
                ),
            ),
        ]
    )


def svm_search_space() -> list[dict[str, object]]:
    return [
        {
            "model__kernel": ["rbf"],
            "model__C": [0.1, 1, 10, 30],
            "model__gamma": ["scale", 0.01, 0.1],
        },
        {
            "model__kernel": ["linear"],
            "model__C": [0.1, 1, 10],
        },
    ]


def random_forest_search_space() -> dict[str, list[object]]:
    return {
        "model__n_estimators": [200, 400, 600],
        "model__max_depth": [None, 6, 10, 16, 24],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2"],
    }
