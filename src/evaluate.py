"""
evaluate.py
───────────
Generates evaluation plots from a trained model.

Plots saved:
  - confusion_matrix.png
  - feature_importance.png  (top 30 features)
  - distribution_comparison.png  (real vs predicted)

Usage:
    python -m src.evaluate
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from src.config import resolve_path


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved → {path}")


def _plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    output_dir: Path,
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_norm,
        annot=cm,          # show raw counts in cells
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title("Confusion Matrix", fontsize=14, pad=12)
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_ylabel("True label", fontsize=11)
    plt.xticks(rotation=35, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    _save(fig, output_dir / "confusion_matrix.png")


def _plot_feature_importance(
    model: XGBClassifier,
    feature_names: list[str],
    output_dir: Path,
    top_n: int = 30,
) -> None:
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    top_names = [feature_names[i] for i in indices]
    top_vals = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#2196F3" if not n.startswith("emb_") else "#90CAF9" for n in top_names]
    ax.barh(range(top_n), top_vals[::-1], color=colors[::-1])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_names[::-1], fontsize=8)
    ax.set_xlabel("Feature Importance (gain)", fontsize=11)
    ax.set_title(f"Top {top_n} Feature Importances", fontsize=14)
    ax.invert_xaxis()
    _save(fig, output_dir / "feature_importance.png")


def _plot_distribution_comparison(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    output_dir: Path,
) -> None:
    n_classes = len(class_names)
    x = np.arange(n_classes)
    width = 0.35

    real_counts = np.bincount(y_true, minlength=n_classes)
    pred_counts = np.bincount(y_pred, minlength=n_classes)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width / 2, real_counts, width, label="Real", color="#42A5F5")
    ax.bar(x + width / 2, pred_counts, width, label="Predicted", color="#EF5350")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Real vs Predicted Label Distribution", fontsize=14)
    ax.legend()
    _save(fig, output_dir / "distribution_comparison.png")


def plot_results(
    model: XGBClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
    feature_names: list[str],
    output_dir: Path | None = None,
) -> None:
    """Generate and save all three evaluation plots."""
    if output_dir is None:
        output_dir = resolve_path("plots")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_pred = model.predict(X_test)
    class_names = label_encoder.classes_.tolist()

    print("  Plotting confusion matrix …")
    _plot_confusion_matrix(y_test, y_pred, class_names, output_dir)

    print("  Plotting feature importances …")
    _plot_feature_importance(model, feature_names, output_dir)

    print("  Plotting distribution comparison …")
    _plot_distribution_comparison(y_test, y_pred, class_names, output_dir)


# ── Standalone (requires trained model) ──────────────────────────────────────

if __name__ == "__main__":
    import joblib
    from src.feature_engineering import load_and_clean, generate_embeddings, build_feature_matrix

    model_path = resolve_path("model")
    if not model_path.exists():
        raise FileNotFoundError(
            f"No model found at {model_path}. Run `python -m src.train_model` first."
        )

    arts = joblib.load(model_path)
    df = load_and_clean()
    emb = generate_embeddings(df)
    X, y, *_ = build_feature_matrix(
        df, emb,
        scaler=arts["scaler"],
        cat_encoder=arts["cat_encoder"],
        label_encoder=arts["label_encoder"],
        fit=False,
    )
    plot_results(
        arts["model"], X, y,
        arts["label_encoder"], arts["feature_names"],
    )
    print("All plots generated.")
