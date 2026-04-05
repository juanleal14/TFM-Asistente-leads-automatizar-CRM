"""
feature_engineering.py
───────────────────────
Loads the raw CSV, generates sentence embeddings, and builds the final
feature matrix used for training and prediction.

Usage (standalone check):
    python -m src.feature_engineering
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from src.config import CONFIG, resolve_path

# ── Config shortcuts ──────────────────────────────────────────────────────────
CAT_FEATURES: list[str] = CONFIG["categorical_features"]
NUM_FEATURES: list[str] = CONFIG["numeric_features"]
NULL_FILL: str = CONFIG["null_fill_value"]
EMBEDDING_MODEL: str = CONFIG["embedding_model"]
EMBEDDING_DIM: int = CONFIG["embedding_dim"]


# ── Step 1 — Load & clean ─────────────────────────────────────────────────────

def load_and_clean(csv_path: str | Path | None = None) -> pd.DataFrame:
    """Load the raw CSV and apply basic cleaning.

    Fills NaN in *prev_outcome* and *prev_next_step* with NULL_FILL so that
    first-call rows are properly represented in categorical encoding.
    """
    if csv_path is None:
        csv_path = resolve_path("raw_data")
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    for col in ("prev_outcome", "prev_next_step"):
        df[col] = df[col].fillna(NULL_FILL).replace("", NULL_FILL)

    # Ensure numeric columns are numeric (coerce bad values to NaN → fill 0)
    for col in NUM_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


# ── Step 2 — Embeddings ───────────────────────────────────────────────────────

def generate_embeddings(
    df: pd.DataFrame,
    model_name: str | None = None,
    cache_path: str | Path | None = None,
) -> np.ndarray:
    """Return a (N, 2*EMBEDDING_DIM) matrix of sentence embeddings.

    Two embeddings are computed per row:
    - transcript embedding  : ``current_transcript``
    - context embedding     : ``initial_interest_notes`` + " | " + ``prev_outcome``

    Results are cached to *cache_path*.npz to avoid re-computation on reruns.
    """
    if model_name is None:
        model_name = EMBEDDING_MODEL
    if cache_path is None:
        cache_path = resolve_path("processed_data")

    cache_file = Path(str(cache_path) + ".npz")

    if cache_file.exists():
        print(f"  Loading cached embeddings from {cache_file}")
        data = np.load(cache_file)
        return data["embeddings"]

    print(f"  Computing embeddings with model '{model_name}' …")
    model = SentenceTransformer(model_name)

    transcripts = df["current_transcript"].fillna("").tolist()
    contexts = (
        df["initial_interest_notes"].fillna("") + " | " + df["prev_outcome"].fillna("")
    ).tolist()

    emb_transcripts = model.encode(transcripts, show_progress_bar=True,
                                   batch_size=64, convert_to_numpy=True)
    emb_contexts = model.encode(contexts, show_progress_bar=True,
                                batch_size=64, convert_to_numpy=True)

    embeddings = np.hstack([emb_transcripts, emb_contexts])

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_file, embeddings=embeddings)
    print(f"  Embeddings cached to {cache_file}")

    return embeddings


# ── Step 3 — Feature matrix ───────────────────────────────────────────────────

def build_feature_matrix(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    scaler: StandardScaler | None = None,
    cat_encoder: OneHotEncoder | None = None,
    label_encoder: LabelEncoder | None = None,
    fit: bool = True,
) -> tuple[np.ndarray, np.ndarray | None, StandardScaler,
           OneHotEncoder, LabelEncoder, list[str]]:
    """Combine numeric, categorical, and embedding features into *X*.

    Parameters
    ----------
    df          : cleaned DataFrame from ``load_and_clean``
    embeddings  : output of ``generate_embeddings``
    scaler      : pre-fitted StandardScaler (required when fit=False)
    cat_encoder : pre-fitted OneHotEncoder (required when fit=False)
    label_encoder: pre-fitted LabelEncoder (required when fit=False)
    fit         : when True, fit new encoders; when False, use provided ones

    Returns
    -------
    X, y, scaler, cat_encoder, label_encoder, feature_names
    y is None when 'next_step' column is absent (prediction mode).
    """
    # ── Numeric ──────────────────────────────────────────────
    X_num = df[NUM_FEATURES].values.astype(float)
    if fit:
        scaler = StandardScaler()
        X_num_scaled = scaler.fit_transform(X_num)
    else:
        X_num_scaled = scaler.transform(X_num)

    # ── Categorical ───────────────────────────────────────────
    X_cat_raw = df[CAT_FEATURES].fillna(NULL_FILL).astype(str)
    if fit:
        cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        X_cat_ohe = cat_encoder.fit_transform(X_cat_raw)
    else:
        X_cat_ohe = cat_encoder.transform(X_cat_raw)

    # ── Stack ─────────────────────────────────────────────────
    X = np.hstack([X_num_scaled, X_cat_ohe, embeddings])

    # ── Target ────────────────────────────────────────────────
    y = None
    if "next_step" in df.columns:
        if fit:
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(df["next_step"])
        else:
            y = label_encoder.transform(df["next_step"])

    # ── Feature names (for interpretability) ─────────────────
    ohe_feature_names: list[str] = cat_encoder.get_feature_names_out(CAT_FEATURES).tolist()
    emb_feature_names = [f"emb_{i}" for i in range(embeddings.shape[1])]
    feature_names: list[str] = NUM_FEATURES + ohe_feature_names + emb_feature_names

    return X, y, scaler, cat_encoder, label_encoder, feature_names


# ── Quick sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading data …")
    df = load_and_clean()
    print(f"  {len(df)} rows, {df['lead_id'].nunique()} leads")
    emb = generate_embeddings(df)
    print(f"  Embeddings shape: {emb.shape}")
    X, y, sc, ce, le, names = build_feature_matrix(df, emb)
    print(f"  Feature matrix shape: {X.shape}")
    print(f"  Classes: {le.classes_.tolist()}")
