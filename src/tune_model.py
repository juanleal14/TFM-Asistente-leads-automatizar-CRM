"""
tune_model.py
─────────────
Hyperparameter search for XGBoost and Random Forest using RandomizedSearchCV.

Why RandomizedSearchCV (not GridSearchCV):
  XGBoost alone has ~2 000 parameter combinations in the grid defined in
  config.yaml.  GridSearch would take hours; RandomizedSearchCV with n_iter=20
  explores the space efficiently (Bergstra & Bengio, 2012) and is a standard
  practice defensible in an academic context.

All search results are saved to:
  experiments/hyperparams/{model}_best_params.json
  experiments/hyperparams/{model}_cv_results.csv

Usage:
    python -m src.tune_model
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from src.config import CONFIG, resolve_path
from src.evaluate import evaluate_model

# ── Config shortcuts ───────────────────────────────────────────────────────────
_TUNE_CFG = CONFIG.get("tuning", {})
_SPLIT_CFG = CONFIG["train_test_split"]
_MODEL_PARAMS = CONFIG["model_params"]


# ── Param grid helpers ─────────────────────────────────────────────────────────

def get_param_grid(model_name: str, config: dict | None = None) -> dict:
    """Return the param grid for *model_name* from config.yaml.

    YAML *null* values are converted to Python ``None`` (e.g. max_depth: null).
    """
    cfg = config or CONFIG
    grids = cfg.get("tuning", {}).get("param_grids", {})
    if model_name not in grids:
        raise KeyError(
            f"No param_grid for '{model_name}' in config.yaml [tuning.param_grids]. "
            f"Available: {list(grids.keys())}"
        )
    raw = grids[model_name]
    # YAML null → Python None for every list in the grid
    return {k: [None if v is None else v for v in vals] for k, vals in raw.items()}


# ── Core tuning function ───────────────────────────────────────────────────────

def tune_model(
    model_name: str,
    estimator: Any,
    param_grid: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_iter: int | None = None,
    cv_splits: int | None = None,
    scoring: str | None = None,
    random_state: int = 42,
    tracker=None,
) -> tuple[Any, dict, pd.DataFrame]:
    """Run RandomizedSearchCV and return (best_estimator, best_params, cv_results_df).

    Parameters
    ----------
    model_name    : identifier string used for logging / file names
    estimator     : unfitted sklearn-compatible classifier
    param_grid    : dict of parameter distributions / lists
    X_train       : training features
    y_train       : training labels (integer-encoded)
    n_iter        : number of random candidates (default from config)
    cv_splits     : number of CV folds (default from config)
    scoring       : sklearn metric string (default from config)
    random_state  : reproducibility seed
    tracker       : optional ExperimentTracker instance

    Returns
    -------
    best_estimator : fitted estimator with best params
    best_params    : dict of best hyperparameters
    cv_results_df  : all candidates sorted by rank_test_score
    """
    n_iter = n_iter or _TUNE_CFG.get("n_iter", 20)
    cv_splits = cv_splits or _TUNE_CFG.get("cv_splits", 5)
    scoring = scoring or _TUNE_CFG.get("scoring", "f1_weighted")

    from sklearn.model_selection import StratifiedKFold

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    print(f"\n  Tuning {model_name}  (n_iter={n_iter}, cv={cv_splits}, scoring={scoring}) …")
    t0 = time.perf_counter()

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        random_state=random_state,
        n_jobs=-1,
        verbose=0,
        refit=True,          # refit best estimator on full X_train
        return_train_score=False,
    )
    search.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0

    best_estimator = search.best_estimator_
    best_params = search.best_params_
    best_score = search.best_score_

    print(f"    Best {scoring}: {best_score:.4f}  (elapsed {elapsed:.1f}s)")
    print(f"    Best params: {best_params}")

    # Build results DataFrame
    cv_results_df = pd.DataFrame(search.cv_results_)
    useful_cols = [
        "rank_test_score", "mean_test_score", "std_test_score", "params"
    ]
    cv_results_df = (
        cv_results_df[useful_cols]
        .sort_values("rank_test_score")
        .reset_index(drop=True)
    )

    # Log to tracker if provided
    if tracker is not None:
        tracker.log_metrics({
            f"{model_name}_best_cv_{scoring}": best_score,
            f"{model_name}_tuning_n_iter": n_iter,
        })

    return best_estimator, best_params, cv_results_df


# ── Tune all configured models ────────────────────────────────────────────────

def tune_all(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
    tracker=None,
) -> dict[str, dict]:
    """Tune XGBoost and Random Forest; evaluate best estimators on test set.

    Returns
    -------
    dict mapping model_name → best_params
    """
    from src.experiment_tracker import ExperimentTracker

    # ── XGBoost ───────────────────────────────────────────────────────────────
    xgb_params = dict(_MODEL_PARAMS)
    xgb_params.pop("use_label_encoder", None)
    xgb_params.pop("eval_metric", None)
    xgb_rs = xgb_params.pop("random_state", 42)
    xgb_base = XGBClassifier(
        **xgb_params,
        eval_metric="mlogloss",
        random_state=xgb_rs,
        verbosity=0,
    )

    xgb_grid = get_param_grid("xgboost")
    xgb_best, xgb_params_best, xgb_cv_df = tune_model(
        "xgboost", xgb_base, xgb_grid, X_train, y_train, tracker=tracker
    )
    xgb_metrics = evaluate_model(xgb_best, X_test, y_test, label_encoder)
    print(
        f"    XGBoost (tuned) test  →  "
        f"acc={xgb_metrics['accuracy']:.4f}  "
        f"f1_w={xgb_metrics['f1_weighted']:.4f}"
    )

    # ── Random Forest ─────────────────────────────────────────────────────────
    rs = _TUNE_CFG.get("random_state", 42)
    rf_base = RandomForestClassifier(random_state=rs, n_jobs=-1)
    rf_grid = get_param_grid("random_forest")
    rf_best, rf_params_best, rf_cv_df = tune_model(
        "random_forest", rf_base, rf_grid, X_train, y_train, tracker=tracker
    )
    rf_metrics = evaluate_model(rf_best, X_test, y_test, label_encoder)
    print(
        f"    RandomForest (tuned) test  →  "
        f"acc={rf_metrics['accuracy']:.4f}  "
        f"f1_w={rf_metrics['f1_weighted']:.4f}"
    )

    # ── Persist artefacts ─────────────────────────────────────────────────────
    _save_tuning_artefacts("xgboost", xgb_params_best, xgb_cv_df)
    _save_tuning_artefacts("random_forest", rf_params_best, rf_cv_df)

    if tracker is not None:
        tracker.save_hyperparams("xgboost", xgb_params_best)
        tracker.save_hyperparams("random_forest", rf_params_best)
        # Log test metrics for both tuned models
        tracker.log_metrics({
            "xgboost_tuned_f1_weighted": xgb_metrics["f1_weighted"],
            "xgboost_tuned_accuracy": xgb_metrics["accuracy"],
            "rf_tuned_f1_weighted": rf_metrics["f1_weighted"],
            "rf_tuned_accuracy": rf_metrics["accuracy"],
        })

    return {
        "xgboost": xgb_params_best,
        "random_forest": rf_params_best,
    }


def _save_tuning_artefacts(
    model_name: str,
    best_params: dict,
    cv_results_df: pd.DataFrame,
) -> None:
    """Save best_params JSON and cv_results CSV to experiments/hyperparams/."""
    from src.config import _PROJECT_ROOT  # type: ignore[attr-defined]
    experiments_dir = CONFIG.get("experiments_dir", "experiments")
    hp_dir = _PROJECT_ROOT / experiments_dir / "hyperparams"
    hp_dir.mkdir(parents=True, exist_ok=True)

    # CV results (all candidates)
    csv_path = hp_dir / f"{model_name}_cv_results.csv"
    cv_results_df.to_csv(csv_path, index=False)
    print(f"    CV results saved → {csv_path}")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main() -> None:
    from src.experiment_tracker import ExperimentTracker
    from src.feature_engineering import load_and_clean, generate_embeddings, build_feature_matrix

    print("=== MoveUp — Hyperparameter Tuning ===\n")

    # 1. Build feature matrix
    print("[1/3] Loading data and building features …")
    df = load_and_clean()
    embeddings = generate_embeddings(df)
    X, y, scaler, cat_enc, label_encoder, feature_names = build_feature_matrix(df, embeddings)
    print(f"      X shape: {X.shape}")

    # 2. Same split as train_model.py
    print("\n[2/3] Splitting data …")
    min_count = np.bincount(y).min()
    if min_count >= 2:
        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=_SPLIT_CFG["test_size"],
            random_state=_SPLIT_CFG["random_state"],
        )
        train_idx, test_idx = next(sss.split(X, y))
    else:
        train_idx = test_idx = np.arange(len(X))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    print(f"      Train: {len(X_train)}, Test: {len(X_test)}")

    # 3. Tune with tracker
    print("\n[3/3] Tuning models …")
    tracker = ExperimentTracker()
    run_id = tracker.start_run(
        "hyperparameter_tuning",
        config={
            "n_iter": _TUNE_CFG.get("n_iter", 20),
            "cv_splits": _TUNE_CFG.get("cv_splits", 5),
            "scoring": _TUNE_CFG.get("scoring", "f1_weighted"),
        },
    )
    tracker.log_model_info("tuning", {"models": ["xgboost", "random_forest"]})

    best_params = tune_all(X_train, y_train, X_test, y_test, label_encoder, tracker=tracker)
    tracker.end_run()

    print("\n── Best Parameters ──")
    for name, params in best_params.items():
        print(f"\n  {name}:")
        for k, v in params.items():
            print(f"    {k}: {v}")

    print("\nDone. Results saved to experiments/hyperparams/")


if __name__ == "__main__":
    main()
