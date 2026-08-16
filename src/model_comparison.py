"""
model_comparison.py
───────────────────
Compare multiple classifiers on the same feature matrix and test split used
by train_model.py.  Results are saved to experiments/model_comparison.csv.

Classifiers evaluated:
  - DummyClassifier (most-frequent baseline)
  - Logistic Regression
  - Random Forest
  - XGBoost  (reference model)
  - LightGBM (optional — skipped gracefully if not installed)

Usage:
    python -m src.model_comparison
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, cross_val_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from src.config import CONFIG, resolve_path
from src.evaluate import evaluate_model

# ── Optional LightGBM ─────────────────────────────────────────────────────────
try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    print("[INFO] LightGBM not installed. It will be excluded from comparison.")

# ── Config shortcuts ───────────────────────────────────────────────────────────
_CMP_CFG = CONFIG.get("comparison", {})
_MODEL_PARAMS = CONFIG["model_params"]
_SPLIT_CFG = CONFIG["train_test_split"]


# ── Model factory ─────────────────────────────────────────────────────────────

def get_models(config: dict | None = None) -> dict[str, Any]:
    """Return a dict of {name: unfitted classifier} to compare.

    Parameters
    ----------
    config : optional override of CONFIG (for testing / custom runs)
    """
    cfg = config or CONFIG
    rs = cfg.get("comparison", {}).get("random_state", 42)
    mp = cfg.get("model_params", _MODEL_PARAMS)

    # Build XGBoost params (drop keys XGBClassifier handles differently)
    xgb_params = dict(mp)
    xgb_params.pop("use_label_encoder", None)
    xgb_params.pop("eval_metric", None)
    random_state = xgb_params.pop("random_state", 42)

    models: dict[str, Any] = {
        "dummy_most_frequent": DummyClassifier(
            strategy="most_frequent", random_state=rs
        ),
        "logistic_regression": LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
            random_state=rs,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            random_state=rs,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            **xgb_params,
            eval_metric="mlogloss",
            random_state=random_state,
            verbosity=0,
        ),
    }

    if LGBM_AVAILABLE:
        # n_jobs=1 avoids OpenMP segfault when combined with PyTorch on macOS
        models["lightgbm"] = LGBMClassifier(random_state=rs, verbose=-1, n_jobs=1)

    return models


# ── Main comparison function ───────────────────────────────────────────────────

def compare_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
    cv_splits: int | None = None,
    scoring: str | None = None,
    random_state: int = 42,
    tracker=None,
) -> pd.DataFrame:
    """Train and evaluate every model; return a ranked comparison DataFrame.

    Parameters
    ----------
    X_train, y_train : training data
    X_test,  y_test  : held-out test data
    label_encoder    : fitted LabelEncoder (to decode class names)
    cv_splits        : number of CV folds (default from config)
    scoring          : sklearn scoring string (default from config)
    random_state     : for CV reproducibility
    tracker          : optional ExperimentTracker instance

    Returns
    -------
    pd.DataFrame sorted by f1_weighted DESC with one row per model
    """
    n_splits = cv_splits or _CMP_CFG.get("cv_splits", 5)
    score_fn = scoring or _CMP_CFG.get("scoring_metric", "f1_weighted")

    models = get_models()
    results: list[dict] = []

    print(f"\n{'Model':<30} {'CV F1':>8} {'±':>6} {'Acc':>7} {'F1-w':>8} {'F1-m':>8} {'Top3':>7}")
    print("-" * 80)

    for name, clf in models.items():
        print(f"  Training {name} …", end=" ", flush=True)

        # ── Cross-validation on training set ──────────────────────────────────
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        try:
            cv_scores = cross_val_score(
                clf, X_train, y_train, cv=cv, scoring=score_fn, n_jobs=-1
            )
            cv_mean = float(cv_scores.mean())
            cv_std = float(cv_scores.std())
        except Exception as e:
            print(f"[CV error: {e}]", end=" ")
            cv_mean = cv_std = float("nan")

        # ── Final fit ─────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        clf.fit(X_train, y_train)
        fit_time = time.perf_counter() - t0

        # ── Predict on test set ───────────────────────────────────────────────
        t1 = time.perf_counter()
        metrics = evaluate_model(clf, X_test, y_test, label_encoder, k=3)
        predict_time = time.perf_counter() - t1

        row = {
            "model_name": name,
            "cv_f1_mean": round(cv_mean, 4),
            "cv_f1_std": round(cv_std, 4),
            "accuracy": round(metrics["accuracy"], 4),
            "f1_macro": round(metrics["f1_macro"], 4),
            "f1_weighted": round(metrics["f1_weighted"], 4),
            "precision_macro": round(metrics["precision_macro"], 4),
            "precision_weighted": round(metrics["precision_weighted"], 4),
            "recall_macro": round(metrics["recall_macro"], 4),
            "recall_weighted": round(metrics["recall_weighted"], 4),
            "top3_accuracy": round(metrics.get("top3_accuracy", float("nan")), 4),
            "brier_score_avg": round(metrics.get("brier_score_avg", float("nan")), 4),
            "fit_time_s": round(fit_time, 2),
            "predict_time_s": round(predict_time, 3),
        }
        results.append(row)

        print(
            f"{cv_mean:8.4f} {cv_std:6.4f} "
            f"{row['accuracy']:7.4f} {row['f1_weighted']:8.4f} "
            f"{row['f1_macro']:8.4f} {row['top3_accuracy']:7.4f}"
        )

        # ── Log to tracker (if provided) ──────────────────────────────────────
        if tracker is not None:
            tracker.log_metrics({
                f"{name}_cv_f1_mean": cv_mean,
                f"{name}_f1_weighted": row["f1_weighted"],
                f"{name}_accuracy": row["accuracy"],
            })

    df = pd.DataFrame(results).sort_values("f1_weighted", ascending=False).reset_index(drop=True)
    return df


# ── Save results ──────────────────────────────────────────────────────────────

def save_comparison_results(
    df: pd.DataFrame,
    output_dir: Path | None = None,
    filename: str = "model_comparison.csv",
) -> Path:
    """Save the comparison DataFrame to experiments/model_comparison.csv."""
    if output_dir is None:
        from src.config import _PROJECT_ROOT  # type: ignore[attr-defined]
        experiments_dir = CONFIG.get("experiments_dir", "experiments")
        output_dir = _PROJECT_ROOT / experiments_dir
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    df.to_csv(path, index=False)
    print(f"\n  Comparison saved → {path}")
    return path


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main() -> None:
    from src.experiment_tracker import ExperimentTracker
    from src.feature_engineering import load_and_clean, generate_embeddings, build_feature_matrix

    print("=== MoveUp — Model Comparison ===\n")

    # 1. Data pipeline (same as train_model.py)
    print("[1/4] Loading data and building features …")
    df = load_and_clean()
    embeddings = generate_embeddings(df)
    X, y, scaler, cat_enc, label_encoder, feature_names = build_feature_matrix(df, embeddings)
    print(f"      X shape: {X.shape},  classes: {label_encoder.classes_.tolist()}")

    # 2. Same train/test split as train_model.py (StratifiedShuffleSplit, seed 42)
    print("\n[2/4] Splitting data …")
    min_count = np.bincount(y).min()
    if min_count >= 2:
        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=_SPLIT_CFG["test_size"],
            random_state=_SPLIT_CFG["random_state"],
        )
        train_idx, test_idx = next(sss.split(X, y))
    else:
        print("  ⚠️  Too few samples for stratified split — using full data.")
        train_idx = test_idx = np.arange(len(X))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    print(f"      Train: {len(X_train)}, Test: {len(X_test)}")

    # 3. Compare models with experiment tracking
    print("\n[3/4] Comparing models …")
    tracker = ExperimentTracker()
    run_id = tracker.start_run(
        "model_comparison",
        config={"test_size": _SPLIT_CFG["test_size"], "cv_splits": _CMP_CFG.get("cv_splits", 5)},
    )
    tracker.log_model_info("comparison", {"models": list(get_models().keys())})

    df_results = compare_models(X_train, y_train, X_test, y_test, label_encoder, tracker=tracker)
    tracker.log_metrics({"best_model": df_results.iloc[0]["model_name"],
                         "best_f1_weighted": df_results.iloc[0]["f1_weighted"]})
    tracker.end_run()

    # 4. Save
    print("\n[4/4] Saving results …")
    save_comparison_results(df_results)

    # Pretty-print table
    print("\n── Ranking (sorted by f1_weighted) ──")
    cols = ["model_name", "cv_f1_mean", "cv_f1_std", "accuracy",
            "f1_macro", "f1_weighted", "top3_accuracy", "brier_score_avg"]
    print(df_results[cols].to_string(index=False))
    print(f"\n→ Best model: {df_results.iloc[0]['model_name']}  "
          f"(f1_weighted = {df_results.iloc[0]['f1_weighted']:.4f})")


if __name__ == "__main__":
    main()
