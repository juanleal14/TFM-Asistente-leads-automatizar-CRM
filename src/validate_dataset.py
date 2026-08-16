"""
validate_dataset.py
───────────────────
Analyse the quality of the synthetic CRM dataset for TFM validation.

Checks performed:
  1. Class distribution — imbalance ratio + chi-square test
  2. Feature correlations — Pearson (numeric) + Kruskal-Wallis
  3. Duplicate detection — exact and near-duplicate transcripts
  4. Temporal coherence — call-sequence integrity within each lead
  5. Artificial patterns — sector uniformity + PCA embedding scatter

Results are saved to experiments/validation/.

Usage:
    python -m src.validate_dataset
    python -m src.validate_dataset --skip-embeddings   # skip slow PCA step
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import CONFIG, resolve_path
from src.feature_engineering import load_and_clean
from src.utils import save_json


# ── Constants ─────────────────────────────────────────────────────────────────
_RARE_CLASS_THRESHOLD = 0.05   # classes with < 5 % of samples → "rare"
_NUMERIC_FEATURES = CONFIG["numeric_features"]


# ── Helper: standard result container ─────────────────────────────────────────

def _result(check_name: str, passed: bool, stats: dict,
             warnings: list | None = None, errors: list | None = None) -> dict:
    return {
        "check_name": check_name,
        "passed": passed,
        "warnings": warnings or [],
        "errors": errors or [],
        "stats": stats,
    }


# ── 1. Class distribution ─────────────────────────────────────────────────────

def analyze_class_distribution(
    df: pd.DataFrame, output_dir: Path | None = None
) -> dict:
    """Count and percentage per class, imbalance ratio, chi-square uniformity test."""
    from scipy.stats import chisquare

    counts = df["next_step"].value_counts()
    total = counts.sum()
    percentages = (counts / total * 100).round(2)
    n_classes = len(counts)

    imbalance_ratio = float(counts.max() / max(counts.min(), 1))
    rare_classes = percentages[percentages < _RARE_CLASS_THRESHOLD * 100].index.tolist()

    # Chi-square test against uniform distribution
    expected = np.full(n_classes, total / n_classes)
    chi2_stat, chi2_pvalue = chisquare(counts.values, f_exp=expected)

    warnings = []
    errors = []

    if imbalance_ratio > 10:
        warnings.append(
            f"High imbalance ratio ({imbalance_ratio:.1f}×). "
            "Minority classes may not be learned reliably."
        )
    if rare_classes:
        warnings.append(f"Rare classes (< {_RARE_CLASS_THRESHOLD*100:.0f}%): {rare_classes}")
    if chi2_pvalue < 0.05:
        warnings.append(
            f"Class distribution is NOT uniform (chi2={chi2_stat:.1f}, p={chi2_pvalue:.4f}). "
            "This is expected for a realistic CRM dataset."
        )

    stats = {
        "n_classes": n_classes,
        "total_samples": int(total),
        "counts": counts.to_dict(),
        "percentages": percentages.to_dict(),
        "imbalance_ratio": round(imbalance_ratio, 2),
        "rare_classes": rare_classes,
        "chisquare_stat": round(float(chi2_stat), 3),
        "chisquare_pvalue": round(float(chi2_pvalue), 6),
    }

    # Plot
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(counts.index, counts.values, color="#42A5F5", edgecolor="white")
        ax.bar_label(bars, fmt="%d", padding=3, fontsize=8)
        ax.set_xticklabels(counts.index, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title("Class Distribution — next_step", fontsize=13)
        fig.tight_layout()
        fig.savefig(output_dir / "class_distribution.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved → {output_dir / 'class_distribution.png'}")

    return _result("class_distribution", not warnings and not errors, stats, warnings, errors)


# ── 2. Feature correlations ───────────────────────────────────────────────────

def analyze_feature_correlations(
    df: pd.DataFrame, output_dir: Path | None = None
) -> dict:
    """Pearson correlation matrix of numeric features + Kruskal-Wallis test."""
    from scipy.stats import kruskal

    num_cols = [c for c in _NUMERIC_FEATURES if c in df.columns]
    corr = df[num_cols].corr().round(4)

    # Kruskal-Wallis: does company_num_employees differ across next_step groups?
    groups = [
        grp["company_num_employees"].dropna().values
        for _, grp in df.groupby("next_step")
        if len(grp) > 0
    ]
    try:
        kw_stat, kw_pvalue = kruskal(*groups)
    except Exception:
        kw_stat, kw_pvalue = float("nan"), float("nan")

    warnings = []
    # High employees↔revenue correlation is expected (revenue generated from employees)
    if "company_num_employees" in corr and "company_annual_revenue_eur" in corr:
        r = corr.loc["company_num_employees", "company_annual_revenue_eur"]
        if abs(r) > 0.8:
            warnings.append(
                f"High correlation num_employees ↔ revenue (r={r:.2f}). "
                "Expected artefact of synthetic data generation."
            )

    stats = {
        "numeric_features_used": num_cols,
        "correlation_matrix": corr.to_dict(),
        "kruskal_wallis": {
            "feature": "company_num_employees",
            "target": "next_step",
            "stat": round(float(kw_stat), 3),
            "pvalue": round(float(kw_pvalue), 6),
            "significant": bool(kw_pvalue < 0.05) if not np.isnan(kw_pvalue) else False,
        },
    }

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7, 5))
        import seaborn as sns
        sns.heatmap(
            corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8}
        )
        ax.set_title("Numeric Feature Correlation Matrix", fontsize=12)
        fig.tight_layout()
        fig.savefig(output_dir / "correlations.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved → {output_dir / 'correlations.png'}")

    return _result("feature_correlations", True, stats, warnings)


# ── 3. Duplicate detection ────────────────────────────────────────────────────

def detect_duplicates(df: pd.DataFrame) -> dict:
    """Count exact duplicates and near-duplicate transcripts."""
    key_cols = ["company_name", "contact_name", "call_number"]
    exact_dups = int(df.duplicated(subset=key_cols).sum())

    # Near-duplicate transcripts: MD5 of first 200 characters
    transcript_hashes = (
        df["current_transcript"]
        .fillna("")
        .str[:200]
        .apply(lambda t: hashlib.md5(t.encode()).hexdigest())
    )
    hash_counts = transcript_hashes.value_counts()
    near_dup_hashes = hash_counts[hash_counts > 1]
    near_dup_count = int(near_dup_hashes.sum() - len(near_dup_hashes))

    warnings = []
    if exact_dups > 0:
        warnings.append(f"{exact_dups} exact duplicate rows found (same company+contact+call_number).")
    if near_dup_count > 0:
        warnings.append(
            f"{near_dup_count} near-duplicate transcripts (identical first 200 chars). "
            "GPT may have reused phrasing."
        )

    stats = {
        "exact_duplicates": exact_dups,
        "near_duplicate_transcripts": near_dup_count,
        "total_rows": len(df),
        "unique_transcripts": int(transcript_hashes.nunique()),
    }

    return _result("duplicates", exact_dups == 0 and near_dup_count == 0, stats, warnings)


# ── 4. Temporal coherence ─────────────────────────────────────────────────────

def analyze_temporal_coherence(df: pd.DataFrame) -> dict:
    """Verify call-sequence integrity within each lead."""
    null_fill = CONFIG.get("null_fill_value", "PRIMERA_LLAMADA")
    inconsistencies: list[dict] = []

    # Check 1: call_number==1 rows should have null_fill in prev fields and 0 days_since_last_call
    first_calls = df[df["call_number"] == 1]
    bad_prev_outcome = first_calls[
        first_calls["prev_outcome"].notna()
        & (first_calls["prev_outcome"] != null_fill)
        & (first_calls["prev_outcome"] != "")
    ]
    for _, row in bad_prev_outcome.iterrows():
        inconsistencies.append({
            "lead_id": row.get("lead_id", "?"),
            "call_number": 1,
            "type": "wrong_prev_outcome_first_call",
            "detail": f"prev_outcome='{row['prev_outcome']}' (expected '{null_fill}')",
        })

    bad_days = first_calls[
        first_calls["days_since_last_call"].notna()
        & (first_calls["days_since_last_call"] != 0)
    ]
    for _, row in bad_days.iterrows():
        inconsistencies.append({
            "lead_id": row.get("lead_id", "?"),
            "call_number": 1,
            "type": "nonzero_days_first_call",
            "detail": f"days_since_last_call={row['days_since_last_call']}",
        })

    # Check 2: next_step[n] == prev_next_step[n+1] for the same lead
    if "lead_id" in df.columns and "next_step" in df.columns and "prev_next_step" in df.columns:
        for lead_id, grp in df.groupby("lead_id"):
            grp_sorted = grp.sort_values("call_number").reset_index(drop=True)
            for i in range(len(grp_sorted) - 1):
                cur_next = grp_sorted.loc[i, "next_step"]
                nxt_prev = grp_sorted.loc[i + 1, "prev_next_step"]
                if pd.notna(cur_next) and pd.notna(nxt_prev) and cur_next != nxt_prev:
                    inconsistencies.append({
                        "lead_id": lead_id,
                        "call_number": int(grp_sorted.loc[i + 1, "call_number"]),
                        "type": "next_step_prev_mismatch",
                        "detail": (
                            f"call {int(grp_sorted.loc[i,'call_number'])} next_step='{cur_next}' "
                            f"≠ call {int(grp_sorted.loc[i+1,'call_number'])} prev_next_step='{nxt_prev}'"
                        ),
                    })

    # Check 3: days_since_entry >= days_since_last_call
    if "days_since_entry" in df.columns and "days_since_last_call" in df.columns:
        invalid = df[
            df["days_since_entry"].notna()
            & df["days_since_last_call"].notna()
            & (df["days_since_entry"] < df["days_since_last_call"])
        ]
        for _, row in invalid.iterrows():
            inconsistencies.append({
                "lead_id": row.get("lead_id", "?"),
                "call_number": row.get("call_number", "?"),
                "type": "days_since_entry_lt_last_call",
                "detail": (
                    f"days_since_entry={row['days_since_entry']} < "
                    f"days_since_last_call={row['days_since_last_call']}"
                ),
            })

    n_inc = len(inconsistencies)
    coherence_rate = round(1.0 - n_inc / max(len(df), 1), 4)

    warnings = []
    if n_inc > 0:
        warnings.append(
            f"{n_inc} temporal inconsistencies found "
            f"({n_inc / len(df) * 100:.1f}% of rows)."
        )

    stats = {
        "n_inconsistencies": n_inc,
        "coherence_rate": coherence_rate,
        "inconsistency_examples": inconsistencies[:10],  # cap for JSON readability
    }

    return _result("temporal_coherence", n_inc == 0, stats, warnings)


# ── 5. Artificial patterns ────────────────────────────────────────────────────

def detect_artificial_patterns(
    df: pd.DataFrame,
    output_dir: Path | None = None,
    skip_embeddings: bool = False,
) -> dict:
    """Detect signs of over-uniformity: sector bias and embedding clustering."""
    from scipy.stats import kruskal

    stats: dict[str, Any] = {}
    warnings: list[str] = []

    # Sector representation
    sector_counts = df["company_sector"].value_counts()
    mean_count = sector_counts.mean()
    over_rep = sector_counts[sector_counts > 3 * mean_count].index.tolist()
    stats["sector_counts"] = sector_counts.to_dict()
    stats["over_represented_sectors"] = over_rep
    if over_rep:
        warnings.append(f"Over-represented sectors (>3× mean): {over_rep}")

    # Distribution of next_step by sector — is it suspiciously uniform?
    sector_entropy: dict[str, float] = {}
    for sector, grp in df.groupby("company_sector"):
        dist = grp["next_step"].value_counts(normalize=True)
        entropy = float(-(dist * np.log(dist + 1e-9)).sum())
        sector_entropy[sector] = round(entropy, 4)
    stats["sector_entropy"] = sector_entropy
    entropy_std = round(float(np.std(list(sector_entropy.values()))), 4)
    stats["sector_entropy_std"] = entropy_std
    if entropy_std < 0.1:
        warnings.append(
            f"Low variance in per-sector entropy ({entropy_std:.3f}). "
            "All sectors show very similar next_step distributions — possible dataset uniformity."
        )

    # PCA of embeddings (optional — requires SentenceTransformer)
    pca_done = False
    if not skip_embeddings and output_dir is not None:
        try:
            from sentence_transformers import SentenceTransformer
            from sklearn.decomposition import PCA

            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            sample = df.sample(min(200, len(df)), random_state=42)
            model_name = CONFIG.get("embedding_model", "paraphrase-multilingual-MiniLM-L12-v2")
            st_model = SentenceTransformer(model_name)
            emb = st_model.encode(
                sample["current_transcript"].fillna("").tolist(),
                show_progress_bar=False,
            )
            pca = PCA(n_components=2, random_state=42)
            coords = pca.fit_transform(emb)
            explained = pca.explained_variance_ratio_.sum()

            labels = sample["next_step"].values
            unique_labels = list(set(labels))
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))  # type: ignore[attr-defined]
            label_to_color = dict(zip(unique_labels, colors))

            fig, ax = plt.subplots(figsize=(9, 6))
            for label in unique_labels:
                mask = labels == label
                ax.scatter(
                    coords[mask, 0], coords[mask, 1],
                    c=[label_to_color[label]], label=label, alpha=0.6, s=25, edgecolors="none"
                )
            ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=9)
            ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=9)
            ax.set_title(f"Transcript Embeddings — PCA (total var={explained*100:.1f}%)", fontsize=12)
            ax.legend(fontsize=6, loc="best", framealpha=0.7)
            fig.tight_layout()
            fig.savefig(output_dir / "embedding_pca.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"    Saved → {output_dir / 'embedding_pca.png'}")

            stats["pca_explained_variance"] = round(float(explained), 4)
            pca_done = True
        except Exception as e:
            warnings.append(f"PCA embedding plot skipped: {e}")

    stats["pca_computed"] = pca_done

    return _result("artificial_patterns", len(warnings) == 0, stats, warnings)


# ── Master report ─────────────────────────────────────────────────────────────

def generate_validation_report(
    df: pd.DataFrame | None = None,
    csv_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    skip_embeddings: bool = False,
) -> dict:
    """Run all validation checks and save results to experiments/validation/.

    Parameters
    ----------
    df            : pre-loaded DataFrame (if None, loads from csv_path or config)
    csv_path      : path to raw CSV (used if df is None)
    output_dir    : output directory (default: experiments/validation/)
    skip_embeddings: skip the PCA embedding visualisation (faster)

    Returns
    -------
    dict with one entry per check plus a top-level summary
    """
    # Resolve output directory
    if output_dir is None:
        from src.config import _PROJECT_ROOT  # type: ignore[attr-defined]
        experiments_dir = CONFIG.get("experiments_dir", "experiments")
        output_dir = _PROJECT_ROOT / experiments_dir / "validation"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data if needed
    if df is None:
        if csv_path is not None:
            df = load_and_clean(csv_path)
        else:
            df = load_and_clean()

    print("=== Dataset Validation Report ===\n")
    print(f"  Dataset: {len(df)} rows, {df['lead_id'].nunique()} leads\n")

    checks = {}

    print("[1/5] Class distribution …")
    checks["class_distribution"] = analyze_class_distribution(df, output_dir)

    print("[2/5] Feature correlations …")
    checks["feature_correlations"] = analyze_feature_correlations(df, output_dir)

    print("[3/5] Duplicate detection …")
    checks["duplicates"] = detect_duplicates(df)

    print("[4/5] Temporal coherence …")
    checks["temporal_coherence"] = analyze_temporal_coherence(df)

    print("[5/5] Artificial patterns …")
    checks["artificial_patterns"] = detect_artificial_patterns(
        df, output_dir, skip_embeddings=skip_embeddings
    )

    # Build summary
    n_passed = sum(1 for c in checks.values() if c["passed"])
    report = {
        "summary": {
            "checks_passed": n_passed,
            "checks_total": len(checks),
            "all_passed": n_passed == len(checks),
        },
        "checks": checks,
    }

    # Save JSON report
    report_path = output_dir / "validation_report.json"
    save_json(report, report_path)
    print(f"\n  Report saved → {report_path}")

    # Console summary
    print("\n── Validation Summary ──")
    icons = {True: "[OK]  ", False: "[WARN]"}
    for name, check in checks.items():
        status = icons[check["passed"]]
        msg = f"{status} {name}"
        if check["warnings"]:
            msg += f": {check['warnings'][0]}"
        print(f"  {msg}")

    print(f"\n  {n_passed}/{len(checks)} checks passed.")
    return report


# ── Standalone ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    skip_emb = "--skip-embeddings" in sys.argv
    generate_validation_report(skip_embeddings=skip_emb)
