"""
pipeline_demo.py
────────────────
Demo visual del pipeline completo de un lead.

Muestra paso a paso:
  1. Contexto del lead
  2. Transcript de la llamada actual
  3. Predicción del modelo (siguiente acción + confianza)
  4. Ejecución de esa acción
  5. Vuelta al paso 2 hasta estado terminal

Ideal para defensa TFM o presentaciones.

Uso:
    python -m src.pipeline_demo                 # lead de ejemplo hardcodeado
    python -m src.pipeline_demo --random        # lead generado aleatoriamente
    python -m src.pipeline_demo --seed 7        # semilla concreta para reproducibilidad
    python -m src.pipeline_demo --no-llm        # usar plantillas en vez de GPT-4o-mini
"""
from __future__ import annotations

import os
import random
import sys
import time
from pathlib import Path

import joblib

from src.config import CONFIG, resolve_path
from src.predict import _predict_from_artifacts
from src.simulate import (
    DEFAULT_ACTION_OUTCOMES,
    LeadState,
    generate_synthetic_transcript,
    _generate_synthetic_lead,
)


# ── Lead de ejemplo para la demo ──────────────────────────────────────────────

DEMO_LEAD = {
    "lead_id": "demo-001",
    "contact_name": "Alejandro Vega",
    "contact_role": "Director de Operaciones",
    "company_name": "Iberia Logistics",
    "company_sector": "Logística",
    "company_country": "España",
    "company_city": "Madrid",
    "company_num_employees": 420,
    "company_annual_revenue_eur": 58_000_000.0,
    "lead_source": "LinkedIn",
    "initial_interest_notes": (
        "Empresa de logística con alta frecuencia de desplazamientos B2B. "
        "Interés en centralizar la gestión de movilidad corporativa y reducir costes de flota."
    ),
}


# ── Helpers de formato ─────────────────────────────────────────────────────────

def _sep(char: str = "─", width: int = 70) -> str:
    return char * width


def _bar(label: str, value: float, width: int = 30) -> str:
    filled = int(value * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"{bar}  {value*100:5.1f}%  {label}"


STATUS_ICONS = {
    "converted":          "✓  LEAD CONVERTIDO",
    "lost":               "✗  LEAD PERDIDO",
    "nurturing":          "⏸  LEAD EN NURTURING",
    "max_steps_reached":  "⏹  MÁXIMO DE PASOS ALCANZADO",
}

STATUS_COLORS = {
    "converted": "\033[92m",   # green
    "lost":      "\033[91m",   # red
    "nurturing": "\033[93m",   # yellow
    "max_steps_reached": "\033[90m",  # grey
}
RESET = "\033[0m"


# ── Demo principal ─────────────────────────────────────────────────────────────

def run_demo(
    lead_data: dict | None = None,
    max_steps: int = 8,
    seed: int = 42,
    pause: float = 0.0,
    model_path: Path | None = None,
    use_llm_summary: bool = True,
) -> dict:
    """Ejecuta el pipeline demo paso a paso con salida visual en consola.

    Parameters
    ----------
    lead_data       : dict con los campos del lead (usa DEMO_LEAD si None)
    max_steps       : número máximo de iteraciones
    seed            : semilla para transiciones estocásticas
    pause           : segundos entre pasos (0 = sin pausa, útil para demos en vivo)
    model_path      : ruta al .joblib (usa la de config si None)
    use_llm_summary : si True, usa GPT-4o-mini para generar prev_outcome
                      tras cada llamada (cierre del loop de producción).
                      Requiere OPENAI_API_KEY. Si no disponible, fallback a plantillas.

    Returns
    -------
    dict con trajectory, final_status, total_steps, total_days
    """
    rng = random.Random(seed)
    lead = lead_data or DEMO_LEAD
    ao = DEFAULT_ACTION_OUTCOMES

    # ── Configurar resumidor LLM ──────────────────────────────────────────────
    summarizer = None
    if use_llm_summary:
        if os.environ.get("OPENAI_API_KEY"):
            from src.summarize import summarize_call
            summarizer = summarize_call
        else:
            print("  [WARN] OPENAI_API_KEY no encontrada — fallback a plantillas.")
            print("         Exporta tu key para activar el resumen LLM en producción.\n")

    # ── Cargar modelo ──────────────────────────────────────────────────────────
    if model_path is None:
        model_path = resolve_path("model")
    if not model_path.exists():
        print(f"  [ERROR] Modelo no encontrado en {model_path}")
        print("  Ejecuta primero: python -m src.train_model")
        sys.exit(1)

    arts = joblib.load(model_path)
    from sentence_transformers import SentenceTransformer
    emb_name = arts.get("embedding_model", CONFIG["embedding_model"])
    emb_model = SentenceTransformer(emb_name)

    agent_names = CONFIG.get("agent_names", ["Agente MoveUp"])
    agent_name = rng.choice(agent_names)

    # ── Inicializar estado ─────────────────────────────────────────────────────
    state = LeadState(**{k: v for k, v in lead.items() if k in LeadState.__dataclass_fields__})

    trajectory = []

    # ── Cabecera ───────────────────────────────────────────────────────────────
    print()
    print(_sep("═"))
    print("  PIPELINE DEMO — MoveUp Next-Action Predictor")
    print(_sep("═"))
    print()
    print(f"  Lead        : {state.contact_name}  ({state.contact_role})")
    print(f"  Empresa     : {state.company_name}  ({state.company_sector})")
    print(f"  Ciudad      : {state.company_city}, {state.company_country}")
    print(f"  Empleados   : {state.company_num_employees:,}   |   Facturación: {state.company_annual_revenue_eur/1e6:.1f}M €")
    print(f"  Fuente lead : {state.lead_source}")
    print(f"  Agente      : {agent_name}")
    print()
    print(f"  Notas iniciales:")
    print(f"  \"{state.initial_interest_notes}\"")
    print()

    # ── Loop principal ─────────────────────────────────────────────────────────
    for step in range(max_steps):

        if pause:
            time.sleep(pause)

        print(_sep())
        print(f"  LLAMADA {state.call_number}   |   Día {state.days_since_entry} desde entrada")
        if state.prev_next_step not in ("PRIMERA_LLAMADA", ""):
            print(f"  Contexto anterior: {state.prev_next_step}")
        print(_sep())

        # Generar transcript
        state.current_transcript = generate_synthetic_transcript(state, agent_name)

        # Mostrar transcript
        print()
        print("  [ TRANSCRIPCIÓN ]")
        # Dividir en turnos para mejor legibilidad
        for line in state.current_transcript.split(". "):
            line = line.strip()
            if line:
                if line.startswith("Agente"):
                    print(f"    {line}.")
                elif line.startswith("Contacto"):
                    print(f"    {line}.")
        print()

        if pause:
            time.sleep(pause)

        # Predecir
        pred = _predict_from_artifacts(
            arts, emb_model,
            lead_id=state.lead_id,
            contact_name=state.contact_name,
            contact_role=state.contact_role,
            company_name=state.company_name,
            company_sector=state.company_sector,
            company_country=state.company_country,
            company_city=state.company_city,
            company_num_employees=state.company_num_employees,
            company_annual_revenue_eur=state.company_annual_revenue_eur,
            lead_source=state.lead_source,
            call_number=state.call_number,
            days_since_entry=state.days_since_entry,
            days_since_last_call=state.days_since_last_call,
            prev_outcome=state.prev_outcome,
            prev_next_step=state.prev_next_step,
            current_transcript=state.current_transcript,
            initial_interest_notes=state.initial_interest_notes,
        )

        predicted_action = pred["predicted_next_step"]
        confidence = pred["confidence"]
        probs = pred["probabilities"]

        # Mostrar predicción
        print("  [ PREDICCIÓN DEL MODELO ]")
        print()
        print(f"  Siguiente acción recomendada:")
        print(f"  >>> {predicted_action}  (confianza: {confidence*100:.1f}%)")
        print()
        print("  Distribución completa de probabilidades:")

        # Ordenar por probabilidad descendente
        for label, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
            marker = "►" if label == predicted_action else " "
            print(f"  {marker}  {_bar(label, prob)}")
        print()

        if pause:
            time.sleep(pause)

        # Registrar en trayectoria
        trajectory.append({
            "step": step + 1,
            "call_number": state.call_number,
            "day": state.days_since_entry,
            "predicted_next_step": predicted_action,
            "confidence": confidence,
        })

        # Resolver transición
        outcome_cfg = ao.get(predicted_action, {
            "terminal": False, "prob_terminal": 0.0, "days_increment": (7, 14),
            "outcome_summary": predicted_action,
        })
        is_terminal = outcome_cfg.get("terminal", False)
        terminal_status = outcome_cfg.get("terminal_status", "lost")

        if not is_terminal and outcome_cfg.get("prob_terminal", 0) > 0:
            if rng.random() < outcome_cfg["prob_terminal"]:
                is_terminal = True

        # Mostrar acción ejecutada
        print("  [ ACCIÓN EJECUTADA ]")
        print()
        print(f"  {outcome_cfg.get('outcome_summary', predicted_action)}")

        if is_terminal:
            print()
            final_status = terminal_status if outcome_cfg.get("terminal") else outcome_cfg.get("terminal_status", "converted")
            state.status = final_status

            color = STATUS_COLORS.get(final_status, "")
            icon = STATUS_ICONS.get(final_status, final_status.upper())
            print()
            print(_sep("═"))
            print(f"  {color}{icon}{RESET}")
            print(_sep("═"))
            break

        # ── Generar prev_outcome para la siguiente llamada ────────────────────
        # En producción, este resumen se genera con un LLM tras cada llamada y
        # se guarda en el CRM para alimentar la siguiente predicción. Cierra el
        # loop de producción que en el dataset original generaba GPT-4o.
        if summarizer is not None:
            print("  [ RESUMEN LLM (prev_outcome para próxima llamada) ]")
            try:
                next_prev_outcome = summarizer(
                    state.current_transcript, predicted_action
                )
                print(f"  > {next_prev_outcome}")
            except Exception as e:
                print(f"  [WARN] LLM falló ({e}); usando plantilla.")
                next_prev_outcome = outcome_cfg.get("outcome_summary", predicted_action)
        else:
            next_prev_outcome = outcome_cfg.get("outcome_summary", predicted_action)

        # ── Actualizar estado para el siguiente paso ──────────────────────────
        days_lo, days_hi = outcome_cfg.get("days_increment", (7, 14))
        days_gap = rng.randint(days_lo, max(days_lo, days_hi))
        state.days_since_entry += days_gap
        state.days_since_last_call = days_gap
        state.call_number += 1
        state.prev_next_step = predicted_action
        state.prev_outcome = next_prev_outcome

        print()
        print(f"  Próxima llamada en {days_gap} días  (día {state.days_since_entry} desde entrada)")

    else:
        state.status = "max_steps_reached"
        print()
        print(_sep("═"))
        print(f"  {STATUS_ICONS['max_steps_reached']}")
        print(_sep("═"))

    # ── Resumen final ──────────────────────────────────────────────────────────
    print()
    print("  RESUMEN DE LA TRAYECTORIA")
    print(_sep())
    print(f"  {'Llamada':<10} {'Día':<8} {'Acción predicha':<45} {'Confianza':>10}")
    print(_sep())
    for t in trajectory:
        print(
            f"  {t['call_number']:<10} "
            f"{t['day']:<8} "
            f"{t['predicted_next_step']:<45} "
            f"{t['confidence']*100:>9.1f}%"
        )
    print(_sep())
    print(f"  Total: {len(trajectory)} llamada(s)   |   "
          f"Duración: {state.days_since_entry} días   |   "
          f"Estado final: {state.status}")
    print()

    return {
        "lead_id": state.lead_id,
        "trajectory": trajectory,
        "final_status": state.status,
        "total_steps": len(trajectory),
        "total_days": state.days_since_entry,
    }


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    use_random = "--random" in sys.argv
    use_llm = "--no-llm" not in sys.argv
    seed = 42
    pause = 0.0

    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--seed" and i < len(sys.argv):
            seed = int(sys.argv[i + 1])
        if arg == "--pause" and i < len(sys.argv):
            pause = float(sys.argv[i + 1])

    if use_random:
        rng = random.Random(seed)
        lead = _generate_synthetic_lead(rng)
    else:
        lead = DEMO_LEAD

    run_demo(lead_data=lead, seed=seed, pause=pause, use_llm_summary=use_llm)
