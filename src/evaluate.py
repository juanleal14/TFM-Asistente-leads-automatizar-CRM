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

    y_pred = np.argmax(model.predict_proba(X_test), axis=1)
    class_names = label_encoder.classes_.tolist()

    print("  Plotting confusion matrix …")
    _plot_confusion_matrix(y_test, y_pred, class_names, output_dir)

    print("  Plotting feature importances …")
    _plot_feature_importance(model, feature_names, output_dir)

    print("  Plotting distribution comparison …")
    _plot_distribution_comparison(y_test, y_pred, class_names, output_dir)


# ── Extended evaluation functions ────────────────────────────────────────────

def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_encoder,
    k: int = 3,
) -> dict:
    """Compute a comprehensive set of evaluation metrics.

    Works with any sklearn-compatible classifier that exposes predict_proba().

    Parameters
    ----------
    model        : fitted classifier
    X_test       : feature matrix (n_samples, n_features)
    y_test       : integer-encoded true labels (n_samples,)
    label_encoder: fitted LabelEncoder used to decode class indices
    k            : top-k for top-k accuracy (default 3)

    Returns
    -------
    dict with keys:
        accuracy, f1_macro, f1_weighted,
        precision_macro, precision_weighted,
        recall_macro, recall_weighted,
        classification_report (dict),
        top3_accuracy, calibration_metrics,
        confusion_matrix (list[list[int]]),
        per_class_metrics
    """
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        classification_report as _cr,
        confusion_matrix as _cm,
        brier_score_loss,
    )

    proba = model.predict_proba(X_test)
    y_pred = np.argmax(proba, axis=1)
    class_names = label_encoder.classes_.tolist()

    # ── Standard metrics ──────────────────────────────────────────────────────
    metrics: dict = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall_weighted": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
    }

    # ── Classification report (dict form) ─────────────────────────────────────
    # Use only labels present in y_test or y_pred to avoid size mismatch when
    # a class has 0 test samples (e.g. "Escalar a manager" with 2 total rows).
    present_labels = sorted(set(y_test) | set(y_pred))
    present_names = [class_names[i] for i in present_labels]
    metrics["classification_report"] = _cr(
        y_test, y_pred,
        labels=present_labels,
        target_names=present_names,
        output_dict=True,
        zero_division=0,
    )

    # ── Top-k accuracy ────────────────────────────────────────────────────────
    # For each sample, check whether the true label is among the top-k predicted
    top_k_indices = np.argsort(proba, axis=1)[:, -k:]  # shape (N, k)
    top_k_correct = np.any(top_k_indices == y_test[:, None], axis=1)
    metrics[f"top{k}_accuracy"] = float(np.mean(top_k_correct))

    # ── Calibration — Brier score per class (one-vs-rest) ─────────────────────
    brier_per_class: dict[str, float] = {}
    for i, label in enumerate(class_names):
        y_bin = (y_test == i).astype(int)
        brier_per_class[label] = float(brier_score_loss(y_bin, proba[:, i]))
    metrics["calibration_metrics"] = {
        "brier_score_per_class": brier_per_class,
        "brier_score_avg": float(np.mean(list(brier_per_class.values()))),
    }
    # Also expose brier_score_avg at top level for easy CSV logging
    metrics["brier_score_avg"] = metrics["calibration_metrics"]["brier_score_avg"]

    # ── Confusion matrix ──────────────────────────────────────────────────────
    metrics["confusion_matrix"] = _cm(y_test, y_pred).tolist()

    # ── Per-class breakdown ───────────────────────────────────────────────────
    cr_dict = metrics["classification_report"]
    per_class: dict[str, dict] = {}
    for label in class_names:
        entry = cr_dict.get(label, {})
        per_class[label] = {
            "precision": entry.get("precision", 0.0),
            "recall": entry.get("recall", 0.0),
            "f1": entry.get("f1-score", 0.0),
            "support": int(entry.get("support", 0)),
        }
    metrics["per_class_metrics"] = per_class

    return metrics


def plot_calibration(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_encoder,
    output_dir: "Path | None" = None,
    n_bins: int = 10,
) -> None:
    """Plot reliability diagrams (one per class) and save calibration_plot.png.

    Uses sklearn.calibration.calibration_curve with strategy='uniform'.
    Subplot layout: 2 rows × 4 cols for 7 classes (last subplot left blank).
    """
    from sklearn.calibration import calibration_curve

    if output_dir is None:
        output_dir = resolve_path("plots")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    class_names = label_encoder.classes_.tolist()
    proba = model.predict_proba(X_test)
    n_classes = len(class_names)

    ncols = 4
    nrows = (n_classes + ncols - 1) // ncols  # ceil division
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes_flat = axes.flatten()

    for i, label in enumerate(class_names):
        ax = axes_flat[i]
        y_bin = (y_test == i).astype(int)
        if y_bin.sum() == 0:
            ax.set_title(f"{label}\n(no samples)", fontsize=8)
            ax.axis("off")
            continue
        try:
            fraction_pos, mean_pred = calibration_curve(
                y_bin, proba[:, i], n_bins=n_bins, strategy="uniform"
            )
            ax.plot(mean_pred, fraction_pos, "s-", label="Model")
            ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect")
            ax.set_xlabel("Mean predicted probability", fontsize=7)
            ax.set_ylabel("Fraction positive", fontsize=7)
            ax.set_title(label, fontsize=8)
            ax.legend(fontsize=6)
        except Exception:
            ax.set_title(f"{label}\n(calibration error)", fontsize=8)
            ax.axis("off")

    # Hide unused subplots
    for j in range(n_classes, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle("Calibration Reliability Diagrams", fontsize=13, y=1.01)
    plt.tight_layout()
    _save(fig, output_dir / "calibration_plot.png")


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
