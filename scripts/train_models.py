from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
LOCAL_CACHE = ROOT / "artifacts" / ".cache"
LOCAL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(LOCAL_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(LOCAL_CACHE))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from coffee_quality.config import load_config
from coffee_quality.data import prepare_dataset
from coffee_quality.evaluation import evaluate_classifier, save_json, save_model, save_predictions
from coffee_quality.pipeline import (
    build_random_forest_pipeline,
    build_svm_pipeline,
    random_forest_search_space,
    svm_search_space,
)
from coffee_quality.visualization import (
    create_exploratory_figures,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_permutation_importance,
    plot_roc_curves,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SVM and Random Forest coffee quality classifiers.")
    parser.add_argument("--config", default="configs/training_config.json", help="Path to training config JSON.")
    parser.add_argument("--rf-iterations", type=int, default=35, help="Random search iterations for Random Forest.")
    parser.add_argument("--skip-permutation", action="store_true", help="Skip permutation importance plots.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    prepared = prepare_dataset(config)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "models").mkdir(parents=True, exist_ok=True)
    (config.output_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (config.output_dir / "figures").mkdir(parents=True, exist_ok=True)
    (config.output_dir / "processed").mkdir(parents=True, exist_ok=True)

    prepared.cleaned.to_csv(config.output_dir / "processed" / "coffee_cleaned_with_quality_label.csv", index=False)
    create_exploratory_figures(prepared.cleaned, prepared.target, config)

    x_train, x_test, y_train, y_test = train_test_split(
        prepared.features,
        prepared.target,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=prepared.target,
    )

    searches = {
        "svm": GridSearchCV(
            estimator=build_svm_pipeline(config),
            param_grid=svm_search_space(),
            scoring="f1_macro",
            cv=config.cv_folds,
            n_jobs=1,
            refit=True,
            verbose=1,
        ),
        "random_forest": RandomizedSearchCV(
            estimator=build_random_forest_pipeline(config),
            param_distributions=random_forest_search_space(),
            n_iter=args.rf_iterations,
            scoring="f1_macro",
            cv=config.cv_folds,
            random_state=config.random_state,
            n_jobs=1,
            refit=True,
            verbose=1,
        ),
    }

    summary_rows: list[dict[str, object]] = []
    all_metrics: dict[str, object] = {
        "dataset": {
            "raw_rows": len(prepared.raw),
            "clean_rows": len(prepared.cleaned),
            "dropped_rows": prepared.dropped_rows,
            "features": config.feature_columns,
            "target_distribution": prepared.target.value_counts().to_dict(),
            "quality_bins": config.quality_bins,
            "quality_labels": config.quality_labels,
        },
        "models": {},
    }

    for model_name, search in searches.items():
        print(f"\nTraining {model_name}...")
        search.fit(x_train, y_train)
        best_model = search.best_estimator_
        metrics = evaluate_classifier(best_model, x_test, y_test)
        metrics["best_params"] = search.best_params_
        metrics["best_cv_f1_macro"] = float(search.best_score_)
        all_metrics["models"][model_name] = metrics

        save_model(best_model, config.output_dir / "models" / f"{model_name}_model.joblib")
        save_json(metrics, config.output_dir / "metrics" / f"{model_name}_metrics.json")
        save_predictions(
            model_name,
            best_model,
            x_test,
            y_test,
            config.output_dir / "metrics" / f"{model_name}_test_predictions.csv",
        )

        plot_confusion_matrix(
            model_name,
            metrics["labels"],
            metrics["confusion_matrix"],
            config.output_dir / "figures" / f"05_confusion_matrix_{model_name}.png",
        )
        plot_roc_curves(
            model_name,
            best_model,
            x_test,
            y_test,
            config.output_dir / "figures" / f"06_roc_curve_{model_name}.png",
        )
        plot_feature_importance(
            model_name,
            best_model,
            config.output_dir / "figures" / f"07_feature_importance_{model_name}.png",
        )
        if not args.skip_permutation:
            plot_permutation_importance(
                model_name,
                best_model,
                x_test,
                y_test,
                config.output_dir / "figures" / f"08_permutation_importance_{model_name}.png",
                random_state=config.random_state,
            )

        summary_rows.append(
            {
                "model": model_name,
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "precision_macro": metrics["precision_macro"],
                "recall_macro": metrics["recall_macro"],
                "f1_macro": metrics["f1_macro"],
                "f1_weighted": metrics["f1_weighted"],
                "roc_auc_ovr_weighted": metrics.get("roc_auc_ovr_weighted"),
                "best_cv_f1_macro": metrics["best_cv_f1_macro"],
                "best_params": metrics["best_params"],
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values("f1_macro", ascending=False)
    summary.to_csv(config.output_dir / "metrics" / "model_comparison.csv", index=False)
    save_json(all_metrics, config.output_dir / "metrics" / "training_summary.json")

    best = summary.iloc[0]
    print("\nTraining selesai.")
    print(summary.to_string(index=False))
    print(f"\nModel terbaik berdasarkan F1 Macro: {best['model']} ({best['f1_macro']:.4f})")
    print(f"Artefak tersimpan di: {config.output_dir.resolve()}")


if __name__ == "__main__":
    main()
