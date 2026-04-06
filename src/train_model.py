"""
train_model.py
──────────────
Orchestrates the full training pipeline:
  1. Load & clean data
  2. Generate / load cached embeddings
  3. Build feature matrix
  4. Train XGBoost with cross-validation
  5. Evaluate on held-out test set
  6. Save model + all artefacts to a single .joblib file
  7. Generate evaluation plots

Usage:
    python -m src.train_model
"""
from __future__ import annotations

import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, cross_val_score
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

from src.config import CONFIG, resolve_path
from src.feature_engineering import load_and_clean, generate_embeddings, build_feature_matrix

# ── Config shortcuts ──────────────────────────────────────────────────────────
MODEL_PARAMS: dict = CONFIG["model_params"]
SPLIT_CFG: dict = CONFIG["train_test_split"]
CV_CFG: dict = CONFIG["cross_validation"]


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    X: np.ndarray,
    y: np.ndarray,
    num_classes: int,
) -> tuple[XGBClassifier, np.ndarray, np.ndarray]:
    """Train an XGBoost classifier.

    Returns
    -------
    model, X_test, y_test
    """
    # Train / test split — stratified when possible, plain otherwise
    min_class_count = np.bincount(y).min()
    if min_class_count >= 2:
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=SPLIT_CFG["test_size"],
            random_state=SPLIT_CFG["random_state"],
        )
        train_idx, test_idx = next(sss.split(X, y))
    else:
        print("  ⚠️  Dataset too small for stratified split — using all data for train/test.")
        train_idx = test_idx = np.arange(len(X))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Build classifier
    params = dict(MODEL_PARAMS)
    params["num_class"] = num_classes
    # Remove keys that XGBClassifier handles differently
    params.pop("use_label_encoder", None)
    params.pop("eval_metric", None)
    params.pop("random_state", None)

    model = XGBClassifier(
        **params,
        eval_metric="mlogloss",
        random_state=MODEL_PARAMS.get("random_state", 42),
        verbosity=0,
    )

    # 5-fold cross-validation on training set (skipped if too few samples)
    if len(X_train) >= CV_CFG["n_splits"] * 2 and min_class_count >= 2:
        cv = StratifiedKFold(
            n_splits=CV_CFG["n_splits"],
            shuffle=True,
            random_state=CV_CFG["random_state"],
        )
        cv_scores = cross_val_score(model, X_train, y_train,
                                    cv=cv, scoring="f1_weighted", n_jobs=-1)
        print(f"\nCross-validation F1 (weighted) — {CV_CFG['n_splits']} folds:")
        print(f"  {cv_scores.round(4).tolist()}")
        print(f"  Mean: {cv_scores.mean():.4f}  ±  {cv_scores.std():.4f}")
    else:
        print("\n  ⚠️  Dataset too small for cross-validation — skipped.")

    # Final fit on full training set
    model.fit(X_train, y_train)

    return model, X_test, y_test


# ── Artefact persistence ──────────────────────────────────────────────────────

def save_model(
    model: XGBClassifier,
    scaler,
    cat_encoder,
    label_encoder,
    feature_names: list[str],
    model_path=None,
) -> None:
    """Persist model + all encoders to a single .joblib file."""
    if model_path is None:
        model_path = resolve_path("model")
    model_path.parent.mkdir(parents=True, exist_ok=True)

    artefacts = {
        "model": model,
        "scaler": scaler,
        "cat_encoder": cat_encoder,
        "label_encoder": label_encoder,
        "feature_names": feature_names,
        "embedding_model": CONFIG["embedding_model"],
        "categorical_features": CONFIG["categorical_features"],
        "numeric_features": CONFIG["numeric_features"],
        "null_fill_value": CONFIG["null_fill_value"],
    }
    joblib.dump(artefacts, model_path)
    print(f"\nArtefacts saved to {model_path}")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main() -> None:
    from src.evaluate import plot_results  # local import to avoid circular dep

    print("=== MoveUp Next-Action Predictor — Training Pipeline ===\n")

    # 1. Load data
    print("[1/6] Loading and cleaning data …")
    df = load_and_clean()
    print(f"      {len(df)} interactions, {df['lead_id'].nunique()} leads")
    print(f"      Class distribution:\n{df['next_step'].value_counts().to_string()}")

    # 2. Embeddings
    print("\n[2/6] Generating / loading embeddings …")
    embeddings = generate_embeddings(df)
    print(f"      Shape: {embeddings.shape}")

    # 3. Feature matrix
    print("\n[3/6] Building feature matrix …")
    X, y, scaler, cat_encoder, label_encoder, feature_names = build_feature_matrix(
        df, embeddings
    )
    print(f"      X shape: {X.shape}, classes: {label_encoder.classes_.tolist()}")

    # 4. Train
    print("\n[4/6] Training XGBoost classifier …")
    model, X_test, y_test = train(X, y, num_classes=len(label_encoder.classes_))

    # 5. Evaluate
    print("\n[5/6] Evaluating on test set …")
    y_pred = np.argmax(model.predict_proba(X_test), axis=1)
    present = np.union1d(y_test, y_pred)
    print(classification_report(
        y_test, y_pred,
        labels=present,
        target_names=label_encoder.inverse_transform(present),
        zero_division=0,
    ))

    # 6. Save
    print("[6/6] Saving artefacts …")
    save_model(model, scaler, cat_encoder, label_encoder, feature_names)

    # 7. Plots
    print("\nGenerating evaluation plots …")
    plots_dir = resolve_path("plots")
    plot_results(model, X_test, y_test, label_encoder, feature_names, plots_dir)
    print("Done.")


if __name__ == "__main__":
    main()
