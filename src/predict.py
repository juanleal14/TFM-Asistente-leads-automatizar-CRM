"""
predict.py
──────────
Standalone prediction module.  Loads saved artefacts and returns the
predicted next action for a new lead interaction.

Usage:
    python -m src.predict
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import CONFIG, resolve_path


def predict_next_step(
    # ── Static lead fields ────────────────────────────────
    lead_id: str,
    contact_name: str,
    contact_role: str,
    company_name: str,
    company_sector: str,
    company_country: str,
    company_city: str,
    company_num_employees: int,
    company_annual_revenue_eur: float,
    lead_source: str,
    # ── Per-call fields ───────────────────────────────────
    call_number: int,
    days_since_entry: int,
    days_since_last_call: int,
    prev_outcome: str,
    prev_next_step: str,
    current_transcript: str,
    initial_interest_notes: str = "",
    # ── Optional override ─────────────────────────────────
    model_path: str | Path | None = None,
) -> dict:
    """Predict the next best action for a lead interaction.

    Returns
    -------
    {
        "predicted_next_step": str,
        "confidence": float,
        "probabilities": {label: prob, ...}
    }
    """
    # ── Load artefacts ────────────────────────────────────────────────────────
    if model_path is None:
        model_path = resolve_path("model")
    arts = joblib.load(model_path)

    model = arts["model"]
    scaler = arts["scaler"]
    cat_encoder = arts["cat_encoder"]
    label_encoder = arts["label_encoder"]
    embedding_model_name: str = arts.get("embedding_model", CONFIG["embedding_model"])
    null_fill: str = arts.get("null_fill_value", CONFIG["null_fill_value"])
    cat_features: list[str] = arts.get("categorical_features", CONFIG["categorical_features"])
    num_features: list[str] = arts.get("numeric_features", CONFIG["numeric_features"])

    # ── Build single-row DataFrame ────────────────────────────────────────────
    prev_outcome_val = prev_outcome if prev_outcome else null_fill
    prev_next_step_val = prev_next_step if prev_next_step else null_fill

    row = {
        "lead_id": lead_id,
        "contact_name": contact_name,
        "contact_role": contact_role,
        "company_name": company_name,
        "company_sector": company_sector,
        "company_country": company_country,
        "company_city": company_city,
        "company_num_employees": company_num_employees,
        "company_annual_revenue_eur": company_annual_revenue_eur,
        "lead_source": lead_source,
        "call_number": call_number,
        "days_since_entry": days_since_entry,
        "days_since_last_call": days_since_last_call,
        "prev_outcome": prev_outcome_val,
        "prev_next_step": prev_next_step_val,
        "current_transcript": current_transcript,
        "initial_interest_notes": initial_interest_notes,
    }
    df = pd.DataFrame([row])

    # ── Generate embeddings (no cache for single predictions) ─────────────────
    # Lazy import: must happen AFTER joblib.load to avoid OpenMP/PyTorch deadlock on macOS
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415
    emb_model = SentenceTransformer(embedding_model_name)
    transcript_emb = emb_model.encode(
        [current_transcript], convert_to_numpy=True
    )
    context_text = f"{initial_interest_notes} | {prev_outcome_val}"
    context_emb = emb_model.encode(
        [context_text], convert_to_numpy=True
    )
    embeddings = np.hstack([transcript_emb, context_emb])  # shape (1, 768)

    # ── Numeric features ──────────────────────────────────────────────────────
    X_num = df[num_features].values.astype(float)
    X_num_scaled = scaler.transform(X_num)

    # ── Categorical features ──────────────────────────────────────────────────
    X_cat_raw = df[cat_features].fillna(null_fill).astype(str)
    X_cat_ohe = cat_encoder.transform(X_cat_raw)

    # ── Stack ─────────────────────────────────────────────────────────────────
    X = np.hstack([X_num_scaled, X_cat_ohe, embeddings])

    # ── Predict ───────────────────────────────────────────────────────────────
    proba = model.predict_proba(X)[0]
    pred_idx = int(np.argmax(proba))
    predicted_label: str = label_encoder.inverse_transform([pred_idx])[0]
    confidence: float = round(float(proba[pred_idx]), 4)

    probabilities = {
        label_encoder.inverse_transform([i])[0]: round(float(p), 4)
        for i, p in enumerate(proba)
    }

    return {
        "predicted_next_step": predicted_label,
        "confidence": confidence,
        "probabilities": probabilities,
    }


def main() -> None:
    """Run a hardcoded example prediction for quick testing."""
    example = dict(
        lead_id="demo-0001",
        contact_name="Alejandro García",
        contact_role="Director de Operaciones",
        company_name="Iberia Logistics",
        company_sector="Logística",
        company_country="España",
        company_city="Madrid",
        company_num_employees=850,
        company_annual_revenue_eur=95_000_000,
        lead_source="LinkedIn",
        call_number=2,
        days_since_entry=18,
        days_since_last_call=7,
        prev_outcome=(
            "Primera llamada positiva. El contacto mostró interés en reducir costes de "
            "flota y solicitó documentación sobre tarifas corporativas."
        ),
        prev_next_step="Enviar documentación",
        current_transcript=(
            "Agente: Buenos días, Alejandro. Le llamo como habíamos quedado para "
            "hacer seguimiento de la documentación que le enviamos la semana pasada.\n"
            "Contacto: Sí, la revisé con el equipo. Nos pareció muy interesante la "
            "tarifa corporativa. Tenemos más de 200 desplazamientos al mes entre "
            "Madrid y Barcelona.\n"
            "Agente: Perfecto, con ese volumen podríamos ofrecerle condiciones "
            "preferentes. ¿Estarían abiertos a una demo con nuestro especialista "
            "para ver la plataforma de gestión?\n"
            "Contacto: Claro, cuanto antes mejor. Queremos resolver esto este "
            "trimestre porque tenemos un evento de empresa en junio.\n"
            "Agente: Entendido. ¿La semana que viene les viene bien para la demo?\n"
            "Contacto: El miércoles por la mañana sería perfecto.\n"
            "Agente: Perfecto, le confirmo la invitación al calendario."
        ),
        initial_interest_notes=(
            "Empresa logística con alta frecuencia de viajes B2B. Interés en "
            "centralizar gastos de movilidad y mejorar visibilidad de flota."
        ),
    )

    print("Running example prediction …\n")
    result = predict_next_step(**example)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
