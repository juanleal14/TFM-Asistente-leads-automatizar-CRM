"""
tests/test_integration.py
─────────────────────────
End-to-end integration tests that exercise the full pipeline:
  load_and_clean → build_feature_matrix → predict_next_step

These tests require:
  - tests/fixtures/sample_data.csv (50 rows of real data)
  - models/moveup_nextstep_model.joblib (trained model)

Skip gracefully when either is absent.

Run only integration tests:
    pytest tests/test_integration.py -v

Run all tests except integration:
    pytest tests/ -m "not integration" -v
"""
from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import MODEL_PATH, SAMPLE_CSV

pytestmark = pytest.mark.integration

VALID_NEXT_STEPS = [
    "Recontactar en X días",
    "Enviar documentación",
    "Agendar demo/reunión con especialista",
    "Escalar a manager del lead",
    "Cerrar lead - no interesado",
    "Cerrar lead - nurturing",
    "Esperar confirmación cliente",
]


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def sample_df():
    """Load 50-row sample from tests/fixtures/sample_data.csv."""
    if not SAMPLE_CSV.exists():
        pytest.skip(f"Sample data not found at {SAMPLE_CSV}")
    from src.feature_engineering import load_and_clean
    return load_and_clean(SAMPLE_CSV)


@pytest.fixture(scope="module")
def first_row(sample_df):
    """Return the first row of sample_df as a dict."""
    return sample_df.iloc[0].to_dict()


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestFeatureMatrixPipeline:
    """Integration tests: load_and_clean → build_feature_matrix."""

    def test_sample_data_loads_correctly(self, sample_df):
        """Sample CSV must load into a non-empty DataFrame with required columns."""
        required_cols = [
            "lead_id", "company_sector", "next_step",
            "prev_outcome", "prev_next_step", "current_transcript",
        ]
        for col in required_cols:
            assert col in sample_df.columns, f"Missing column: {col}"
        assert len(sample_df) > 0

    def test_null_fill_applied_to_sample(self, sample_df):
        """No NaN in prev_outcome or prev_next_step after load_and_clean."""
        from src.config import CONFIG
        null_fill = CONFIG["null_fill_value"]
        assert not sample_df["prev_outcome"].isna().any()
        assert not sample_df["prev_next_step"].isna().any()

    def test_feature_matrix_shape(self, sample_df):
        """Feature matrix shape must be (n_rows, n_features) with n_features > 1000."""
        from src.feature_engineering import build_feature_matrix
        dummy_emb = np.zeros((len(sample_df), 768))
        X, y, sc, ce, le, names = build_feature_matrix(sample_df, dummy_emb)
        assert X.shape[0] == len(sample_df)
        assert X.shape[1] == len(names)
        # The feature matrix should have numeric + OHE + embeddings > 1000 cols
        assert X.shape[1] > 1000, f"Suspiciously few features: {X.shape[1]}"

    def test_all_next_step_classes_are_valid(self, sample_df):
        """All next_step values in the sample must be from the known categories."""
        from src.config import CONFIG
        valid = set(CONFIG["next_step_categories"])
        found = set(sample_df["next_step"].unique())
        unknown = found - valid
        assert not unknown, f"Unknown next_step values: {unknown}"


class TestEndToEndPrediction:
    """Integration tests: feature_engineering → predict_next_step."""

    @pytest.mark.skipif(
        not MODEL_PATH.exists(),
        reason="Trained model not found. Run `python -m src.train_model` first.",
    )
    def test_predict_on_first_sample_row(self, first_row):
        """predict_next_step on the first sample row must return valid output."""
        from src.predict import predict_next_step

        result = predict_next_step(
            lead_id=str(first_row.get("lead_id", "test")),
            contact_name=str(first_row.get("contact_name", "Test")),
            contact_role=str(first_row.get("contact_role", "CEO")),
            company_name=str(first_row.get("company_name", "Corp")),
            company_sector=str(first_row.get("company_sector", "Tecnología")),
            company_country=str(first_row.get("company_country", "España")),
            company_city=str(first_row.get("company_city", "Madrid")),
            company_num_employees=int(first_row.get("company_num_employees", 100)),
            company_annual_revenue_eur=float(first_row.get("company_annual_revenue_eur", 1e7)),
            lead_source=str(first_row.get("lead_source", "LinkedIn")),
            call_number=int(first_row.get("call_number", 1)),
            days_since_entry=int(first_row.get("days_since_entry", 0)),
            days_since_last_call=int(first_row.get("days_since_last_call", 0)),
            prev_outcome=str(first_row.get("prev_outcome", "")),
            prev_next_step=str(first_row.get("prev_next_step", "")),
            current_transcript=str(first_row.get("current_transcript", "Agente: Hola.")),
            initial_interest_notes=str(first_row.get("initial_interest_notes", "")),
        )

        # Schema checks
        assert "predicted_next_step" in result
        assert "confidence" in result
        assert "probabilities" in result

        # Validity checks
        assert result["predicted_next_step"] in VALID_NEXT_STEPS
        assert 0.0 <= result["confidence"] <= 1.0
        assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-3

    @pytest.mark.skipif(
        not MODEL_PATH.exists(),
        reason="Trained model not found. Run `python -m src.train_model` first.",
    )
    def test_predictions_are_deterministic(self, first_row):
        """Same input must produce identical predictions (model is deterministic)."""
        from src.predict import predict_next_step

        kwargs = dict(
            lead_id=str(first_row.get("lead_id", "test")),
            contact_name=str(first_row.get("contact_name", "Test")),
            contact_role=str(first_row.get("contact_role", "CEO")),
            company_name=str(first_row.get("company_name", "Corp")),
            company_sector=str(first_row.get("company_sector", "Tecnología")),
            company_country=str(first_row.get("company_country", "España")),
            company_city=str(first_row.get("company_city", "Madrid")),
            company_num_employees=int(first_row.get("company_num_employees", 100)),
            company_annual_revenue_eur=float(first_row.get("company_annual_revenue_eur", 1e7)),
            lead_source=str(first_row.get("lead_source", "LinkedIn")),
            call_number=int(first_row.get("call_number", 1)),
            days_since_entry=int(first_row.get("days_since_entry", 0)),
            days_since_last_call=int(first_row.get("days_since_last_call", 0)),
            prev_outcome=str(first_row.get("prev_outcome", "")),
            prev_next_step=str(first_row.get("prev_next_step", "")),
            current_transcript=str(first_row.get("current_transcript", "Agente: Hola.")),
            initial_interest_notes=str(first_row.get("initial_interest_notes", "")),
        )

        result1 = predict_next_step(**kwargs)
        result2 = predict_next_step(**kwargs)

        assert result1["predicted_next_step"] == result2["predicted_next_step"]
        assert result1["confidence"] == result2["confidence"]
