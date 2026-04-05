"""
tests/test_predict.py
─────────────────────
Unit tests for src/predict.py

Run with:
    pytest tests/test_predict.py -v
"""
import pytest

# TODO: implement full test suite once a trained model exists at
# models/moveup_nextstep_model.joblib.


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


class TestPredictNextStep:
    """Tests for predict_next_step()."""

    def test_output_schema(self):
        # TODO: call predict_next_step(**EXAMPLE_INPUT) with a mocked model and
        # assert the result has keys: predicted_next_step, confidence, probabilities.
        pytest.skip("Trained model not yet available")

    def test_predicted_label_is_valid(self):
        # TODO: assert result["predicted_next_step"] is one of VALID_NEXT_STEPS.
        pytest.skip("Trained model not yet available")

    def test_confidence_in_range(self):
        # TODO: assert 0.0 <= result["confidence"] <= 1.0.
        pytest.skip("Trained model not yet available")

    def test_probabilities_sum_to_one(self):
        # TODO: assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-4.
        pytest.skip("Trained model not yet available")

    def test_probabilities_keys_match_categories(self):
        # TODO: assert set(result["probabilities"].keys()) == set(VALID_NEXT_STEPS).
        pytest.skip("Trained model not yet available")

    def test_null_prev_fields_handled(self):
        # TODO: pass prev_outcome="" and prev_next_step="" and assert no exception
        # is raised (they should be filled with null_fill_value internally).
        pytest.skip("Trained model not yet available")

    def test_custom_model_path(self, tmp_path):
        # TODO: save a minimal mock model to tmp_path / "model.joblib" and pass
        # it as model_path to predict_next_step, asserting correct load.
        pytest.skip("Trained model not yet available")
