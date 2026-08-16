"""
tests/test_feature_engineering.py
──────────────────────────────────
Unit tests for src/feature_engineering.py

All tests run without GPT, without the full dataset, and without the trained
model.  Embedding tests are conditionally skipped when sentence_transformers is
not importable or when the model is not cached locally.

Run with:
    pytest tests/test_feature_engineering.py -v
"""
from __future__ import annotations

import importlib
import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import CONFIG

# ── Import helpers ─────────────────────────────────────────────────────────────

def _can_import(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


_HAS_ST = _can_import("sentence_transformers")


# ── Minimal CSV used by load_and_clean tests (written to tmp_path) ─────────────

_CSV_ROWS = (
    "interaction_id,lead_id,contact_name,contact_role,company_name,"
    "company_sector,company_country,company_city,company_num_employees,"
    "company_annual_revenue_eur,lead_source,lead_entry_date,initial_interest_notes,"
    "call_number,call_timestamp,days_since_entry,days_since_last_call,"
    "prev_outcome,prev_next_step,current_transcript,current_outcome,next_step,final_status\n"
    "r1,l1,Ana G.,CEO,Corp A,Tecnología,España,Madrid,100,10000000,LinkedIn,"
    "2025-01-01,Notas.,1,2025-01-06,5,0,,,Agente: Hi. Contacto: Hola.,OK,"
    "Enviar documentación,Converted\n"
    "r2,l2,Bob L.,CFO,Corp B,Logística,España,Barcelona,200,20000000,LinkedIn,"
    "2025-01-02,Notas B.,1,2025-01-07,5,0,,,Agente: Hi. Contacto: Buenos días.,OK2,"
    "Recontactar en X días,Lost\n"
    "r3,l3,Clara R.,COO,Corp C,Banca,España,Valencia,50,5000000,Formulario web,"
    "2025-01-03,,2,2025-01-10,7,7,Primera toma de contacto.,Enviar documentación,"
    "Agente: Follow-up. Contacto: Sí.,Interés mantenido,"
    "Agendar demo/reunión con especialista,Nurturing\n"
)

NULL_FILL = CONFIG["null_fill_value"]
NUM_FEATURES = CONFIG["numeric_features"]


def _write_csv(tmp_path: Path, content: str = _CSV_ROWS) -> Path:
    p = tmp_path / "test_dataset.csv"
    p.write_text(content, encoding="utf-8")
    return p


# ── load_and_clean ─────────────────────────────────────────────────────────────

class TestLoadAndClean:
    """Tests for load_and_clean()."""

    def test_returns_dataframe(self, tmp_path):
        from src.feature_engineering import load_and_clean

        csv_path = _write_csv(tmp_path)
        df = load_and_clean(csv_path)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    def test_null_fill_applied(self, tmp_path):
        """Rows with empty prev_outcome / prev_next_step get NULL_FILL."""
        from src.feature_engineering import load_and_clean

        csv_path = _write_csv(tmp_path)
        df = load_and_clean(csv_path)

        # Row 1 has empty prev_outcome and prev_next_step
        assert df.loc[0, "prev_outcome"] == NULL_FILL
        assert df.loc[0, "prev_next_step"] == NULL_FILL

        # Row 3 has prev_outcome set — should be preserved, not overwritten
        assert df.loc[2, "prev_outcome"] == "Primera toma de contacto."

    def test_numeric_columns_are_numeric(self, tmp_path):
        """All numeric features must be float/int after load_and_clean."""
        from src.feature_engineering import load_and_clean

        csv_path = _write_csv(tmp_path)
        df = load_and_clean(csv_path)

        for col in NUM_FEATURES:
            assert col in df.columns, f"Missing numeric column: {col}"
            assert pd.api.types.is_numeric_dtype(df[col]), (
                f"Column '{col}' is not numeric after load_and_clean"
            )

    def test_empty_string_prev_fields_are_filled(self, tmp_path):
        """NaN (blank cell) in prev_outcome/prev_next_step is replaced with NULL_FILL."""
        from src.feature_engineering import load_and_clean

        # Use a blank CSV cell (no quotes) which pandas reads as NaN
        csv_with_nan = (
            "interaction_id,lead_id,contact_name,contact_role,company_name,"
            "company_sector,company_country,company_city,company_num_employees,"
            "company_annual_revenue_eur,lead_source,lead_entry_date,initial_interest_notes,"
            "call_number,call_timestamp,days_since_entry,days_since_last_call,"
            "prev_outcome,prev_next_step,current_transcript,current_outcome,next_step,final_status\n"
            "r1,l1,X,CEO,C,Tecnología,España,Madrid,10,1000000,LinkedIn,"
            "2025-01-01,,1,2025-01-06,5,0,,,Transcript,Out,Enviar documentación,Converted\n"
        )
        csv_path = _write_csv(tmp_path, csv_with_nan)
        df = load_and_clean(csv_path)
        # Blank cells become NaN → should be replaced with NULL_FILL
        assert df.loc[0, "prev_outcome"] == NULL_FILL
        assert df.loc[0, "prev_next_step"] == NULL_FILL


# ── generate_embeddings ────────────────────────────────────────────────────────

@pytest.mark.skipif(not _HAS_ST, reason="sentence_transformers not installed")
class TestGenerateEmbeddings:
    """Tests for generate_embeddings().  Require sentence_transformers."""

    def test_output_shape(self, minimal_df, tmp_path):
        """Embeddings must have shape (N, 768) = (N, 2 × 384)."""
        from src.feature_engineering import generate_embeddings

        emb = generate_embeddings(minimal_df, cache_path=tmp_path / "cache")
        assert emb.shape == (len(minimal_df), 768), (
            f"Expected (N, 768), got {emb.shape}"
        )

    def test_cache_hit(self, minimal_df, tmp_path):
        """Second call with same cache_path returns identical results."""
        from src.feature_engineering import generate_embeddings

        cache = tmp_path / "emb_cache"
        emb1 = generate_embeddings(minimal_df, cache_path=cache)
        emb2 = generate_embeddings(minimal_df, cache_path=cache)
        np.testing.assert_array_equal(emb1, emb2)

    def test_cache_file_created(self, minimal_df, tmp_path):
        """Cache .npz file is created after first call."""
        from src.feature_engineering import generate_embeddings

        cache = tmp_path / "my_cache"
        generate_embeddings(minimal_df, cache_path=cache)
        assert (tmp_path / "my_cache.npz").exists()


# ── build_feature_matrix ───────────────────────────────────────────────────────

class TestBuildFeatureMatrix:
    """Tests for build_feature_matrix() — use dummy zero embeddings."""

    def test_feature_count_matches_names(self, minimal_df, dummy_embeddings):
        """X.shape[1] must equal len(feature_names)."""
        from src.feature_engineering import build_feature_matrix

        X, y, sc, ce, le, names = build_feature_matrix(minimal_df, dummy_embeddings)
        assert X.shape[1] == len(names), (
            f"Feature count mismatch: X has {X.shape[1]} cols, "
            f"but feature_names has {len(names)} entries"
        )

    def test_rows_preserved(self, minimal_df, dummy_embeddings):
        """X.shape[0] must equal len(df)."""
        from src.feature_engineering import build_feature_matrix

        X, y, *_ = build_feature_matrix(minimal_df, dummy_embeddings)
        assert X.shape[0] == len(minimal_df)

    def test_label_encoding_roundtrip(self, minimal_df, dummy_embeddings):
        """inverse_transform(y) must recover the original next_step strings."""
        from src.feature_engineering import build_feature_matrix

        X, y, sc, ce, le, names = build_feature_matrix(minimal_df, dummy_embeddings)
        assert y is not None
        reconstructed = le.inverse_transform(y).tolist()
        original = minimal_df["next_step"].tolist()
        assert reconstructed == original

    def test_fit_false_does_not_refit_scaler(self, minimal_df, dummy_embeddings):
        """Calling with fit=False must reuse the scaler without modifying it."""
        from src.feature_engineering import build_feature_matrix

        # First call: fit=True
        X, y, sc, ce, le, names = build_feature_matrix(minimal_df, dummy_embeddings)
        original_mean = sc.mean_.copy()

        # Second call: fit=False on a 2-row subset
        sub_df = minimal_df.iloc[:2].copy().reset_index(drop=True)
        sub_emb = np.zeros((2, 768))
        build_feature_matrix(sub_df, sub_emb, scaler=sc, cat_encoder=ce,
                             label_encoder=le, fit=False)

        # Scaler parameters must not have changed
        np.testing.assert_array_equal(sc.mean_, original_mean)

    def test_no_target_column(self, minimal_df, dummy_embeddings):
        """When next_step is absent, y must be None."""
        from src.feature_engineering import build_feature_matrix

        df_no_target = minimal_df.drop(columns=["next_step"])
        # Need to fit first so encoders exist
        _, _, sc, ce, le, _ = build_feature_matrix(minimal_df, dummy_embeddings)
        X, y, *_ = build_feature_matrix(df_no_target, dummy_embeddings,
                                         scaler=sc, cat_encoder=ce,
                                         label_encoder=le, fit=False)
        assert y is None
