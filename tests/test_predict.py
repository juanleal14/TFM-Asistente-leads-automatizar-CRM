"""
tests/test_predict.py
─────────────────────
Unit tests for src/predict.py

Design decisions:
  - Tests run against the REAL trained model when it exists at
    models/moveup_nextstep_model.joblib.  We do NOT mock joblib.load() —
    mocking XGBoost internals is fragile and proves nothing about the actual
    model behaviour.
  - All tests in TestPredictNextStep are decorated with @MODEL_EXISTS to skip
    gracefully when the model has not been trained yet.
  - test_custom_model_path copies the real model to a tmp_path to verify
    the model_path override mechanism.

Run with:
    pytest tests/test_predict.py -v
    pytest tests/test_predict.py -v -k "not custom_model_path"   # skip copy test
"""
from __future__ import annotations

import pathlib

import pytest

from tests.conftest import MODEL_PATH

VALID_NEXT_STEPS = [
    "Recontactar en X días",
    "Enviar documentación",
    "Agendar demo/reunión con especialista",
    "Escalar a manager del lead",
    "Cerrar lead - no interesado",
    "Cerrar lead - nurturing",
    "Esperar confirmación cliente",
]

EXAMPLE_INPUT = dict(
    lead_id="test-0001",
    contact_name="Ana López",
    contact_role="Director General",
    company_name="Test Corp",
    company_sector="Tecnología",
    company_country="España",
    company_city="Barcelona",
    company_num_employees=300,
    company_annual_revenue_eur=25_000_000,
    lead_source="LinkedIn",
    call_number=1,
    days_since_entry=5,
    days_since_last_call=0,
    prev_outcome="",
    prev_next_step="",
    current_transcript=(
        "Agente: Buenos días, soy Carlos de MoveUp.\n"
        "Contacto: Hola, sí, vi su mensaje en LinkedIn.\n"
        "Agente: ¿Podría contarme cómo gestionan los desplazamientos corporativos?\n"
        "Contacto: Actualmente usamos taxis. Tenemos unos 150 viajes al mes."
    ),
    initial_interest_notes="Interés inicial vía LinkedIn en solución corporativa.",
)

# Decorator — skip entire class if model file is absent
MODEL_EXISTS = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason=f"Trained model not found at {MODEL_PATH}. Run `python -m src.train_model` first.",
)


@MODEL_EXISTS
class TestPredictNextStep:
    """Tests for predict_next_step() against the real trained model."""

    def test_output_schema(self):
        """Result must have exactly the three expected keys."""
        from src.predict import predict_next_step

        result = predict_next_step(**EXAMPLE_INPUT)
        assert set(result.keys()) == {"predicted_next_step", "confidence", "probabilities"}

    def test_predicted_label_is_valid(self):
        """Predicted label must be one of the seven known action classes."""
        from src.predict import predict_next_step

        result = predict_next_step(**EXAMPLE_INPUT)
        assert result["predicted_next_step"] in VALID_NEXT_STEPS, (
            f"Unexpected label: {result['predicted_next_step']}"
        )

    def test_confidence_in_range(self):
        """Confidence must be a probability in [0, 1]."""
        from src.predict import predict_next_step

        result = predict_next_step(**EXAMPLE_INPUT)
        assert isinstance(result["confidence"], float)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_probabilities_sum_to_one(self):
        """Probabilities must form a valid distribution (sum ≈ 1)."""
        from src.predict import predict_next_step

        result = predict_next_step(**EXAMPLE_INPUT)
        total = sum(result["probabilities"].values())
        assert abs(total - 1.0) < 1e-3, f"Probabilities sum to {total}, not 1.0"

    def test_probabilities_keys_match_categories(self):
        """All seven action classes must appear in probabilities dict."""
        from src.predict import predict_next_step
        from src.config import CONFIG

        result = predict_next_step(**EXAMPLE_INPUT)
        expected = set(CONFIG["next_step_categories"])
        assert set(result["probabilities"].keys()) == expected

    def test_null_prev_fields_handled(self):
        """Empty prev_outcome and prev_next_step must not raise exceptions."""
        from src.predict import predict_next_step

        inp = dict(EXAMPLE_INPUT, prev_outcome="", prev_next_step="")
        result = predict_next_step(**inp)
        assert result["predicted_next_step"] in VALID_NEXT_STEPS

    def test_first_call_input(self):
        """First call (call_number=1, no prev context) must work correctly."""
        from src.predict import predict_next_step

        first_call = dict(
            EXAMPLE_INPUT,
            call_number=1,
            days_since_entry=0,
            days_since_last_call=0,
            prev_outcome="",
            prev_next_step="",
        )
        result = predict_next_step(**first_call)
        assert result["predicted_next_step"] in VALID_NEXT_STEPS

    def test_custom_model_path(self, tmp_path):
        """model_path override must load the model from the specified path."""
        import shutil
        from src.predict import predict_next_step

        tmp_model = tmp_path / "model_copy.joblib"
        shutil.copy(MODEL_PATH, tmp_model)

        result = predict_next_step(**EXAMPLE_INPUT, model_path=tmp_model)
        assert result["predicted_next_step"] in VALID_NEXT_STEPS

    def test_confidence_equals_max_probability(self):
        """confidence must equal the highest value in probabilities."""
        from src.predict import predict_next_step

        result = predict_next_step(**EXAMPLE_INPUT)
        max_prob = max(result["probabilities"].values())
        assert abs(result["confidence"] - max_prob) < 1e-4
