"""
simulate.py
───────────
Sequential lead simulation for TFM strategy analysis.

The simulator iterates a lead through predict → act → update cycles until
reaching a terminal state or exhausting max_steps.  Transcripts are generated
from templates (no GPT calls) — fast enough for batch analysis.

Methodological note (for TFM defence):
  Template transcripts produce more homogeneous embeddings than GPT-generated
  ones, which may bias predictions toward more "generic" actions.  This is
  documented as a limitation of the simulation approach.

Usage:
    python -m src.simulate                        # 50-lead batch demo
    python -m src.simulate --n-leads 200          # custom batch size
    python -m src.simulate --strategy aggressive  # strategy comparison
"""
from __future__ import annotations

import random
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import CONFIG
from src.predict import _predict_from_artifacts
from src.utils import save_json


# ── Transcript templates ───────────────────────────────────────────────────────
# 15 templates total (2 per continuing action + 3 terminal)
# Variable slots: {contact_name}, {company_name}, {company_sector}, {agent_name}

TRANSCRIPT_TEMPLATES: dict[str, list[str]] = {
    "PRIMERA_LLAMADA": [
        (
            "Agente: Buenos días {contact_name}, soy {agent_name} de MoveUp. Le llamo porque "
            "detectamos que {company_name} podría beneficiarse de nuestra plataforma de movilidad "
            "corporativa para el sector {company_sector}. "
            "Contacto: Hola, sí, cuénteme. Estamos evaluando opciones para nuestros desplazamientos."
        ),
        (
            "Agente: Hola {contact_name}, le llamo de MoveUp. Veo que {company_name} trabaja en "
            "el sector {company_sector}. ¿Gestionan actualmente los desplazamientos de manera "
            "centralizada? "
            "Contacto: No, todavía lo hacemos de forma manual. Es un problema porque perdemos "
            "mucho tiempo en la gestión."
        ),
    ],
    "Enviar documentación": [
        (
            "Agente: Buenos días {contact_name}, le llamo para hacer seguimiento de la "
            "documentación que enviamos sobre nuestras tarifas corporativas. "
            "¿Ha tenido ocasión de revisarla? "
            "Contacto: Sí, la revisé con el equipo y tenemos algunas preguntas sobre los "
            "niveles de servicio."
        ),
        (
            "Agente: Hola {contact_name}, ¿recibió bien el material que le enviamos? "
            "Contacto: Sí, aunque todavía no hemos tenido tiempo de analizarlo en detalle. "
            "Esta semana hemos tenido mucho trabajo."
        ),
    ],
    "Recontactar en X días": [
        (
            "Agente: Buenos días {contact_name}, quedamos en hablar hoy. "
            "¿Cómo está el proyecto internamente? "
            "Contacto: Seguimos evaluando opciones, tenemos reunión de dirección la próxima "
            "semana donde lo presentaremos."
        ),
        (
            "Agente: Hola {contact_name}, le llamo como acordamos. "
            "¿Ha podido avanzar en la evaluación interna? "
            "Contacto: Todavía no, ha habido cambios en el equipo directivo que han retrasado "
            "la toma de decisiones."
        ),
    ],
    "Agendar demo/reunión con especialista": [
        (
            "Agente: Buenos días {contact_name}, le llamo para confirmar la demo con nuestro "
            "especialista. ¿Todo sigue según lo previsto? "
            "Contacto: Sí, lo tenemos en el calendario. El equipo está bastante interesado "
            "en ver la plataforma en detalle."
        ),
        (
            "Agente: Hola {contact_name}, hemos preparado la presentación técnica para la demo. "
            "¿Hay algo específico que quiera que cubramos? "
            "Contacto: Nos interesa especialmente la integración con nuestros sistemas actuales "
            "y los informes de gasto."
        ),
    ],
    "Escalar a manager del lead": [
        (
            "Agente: Buenos días {contact_name}, le llamo para coordinar la reunión con la "
            "dirección que habíamos mencionado. "
            "Contacto: Sí, el director tiene disponibilidad la próxima semana. "
            "Le confirmo el horario por correo."
        ),
        (
            "Agente: Hola {contact_name}, ¿ha podido hablar con su responsable sobre nuestra "
            "propuesta? "
            "Contacto: Sí, está interesado pero necesita ver los números concretos y el "
            "retorno esperado."
        ),
    ],
    "Esperar confirmación cliente": [
        (
            "Agente: Buenos días {contact_name}, le llamo para ver si han podido tomar una "
            "decisión sobre nuestra propuesta. "
            "Contacto: Seguimos en proceso interno, espero tener respuesta definitiva esta semana."
        ),
        (
            "Agente: Hola {contact_name}, ¿hay novedades sobre la aprobación interna? "
            "Contacto: Aún no, hay un proceso de validación presupuestaria. "
            "Creo que en unos días tendremos respuesta."
        ),
    ],
    "_terminal_lost": [
        (
            "Agente: Buenos días {contact_name}, le llamo para hacer seguimiento. "
            "Contacto: Mire, finalmente hemos decidido no seguir adelante con MoveUp. "
            "No encaja con nuestra situación actual y hemos elegido otra solución."
        ),
    ],
    "_terminal_nurturing": [
        (
            "Agente: Hola {contact_name}, ¿hay alguna novedad sobre el proyecto? "
            "Contacto: Por ahora no vamos a avanzar. El presupuesto está congelado hasta "
            "el próximo año, pero lo retomamos entonces."
        ),
    ],
    "_terminal_converted": [
        (
            "Agente: Buenos días {contact_name}, llamo para confirmar los detalles finales. "
            "Contacto: Perfecto, ya tenemos la aprobación interna. Queremos arrancar cuanto "
            "antes. ¿Cuáles son los próximos pasos para firmar el contrato?"
        ),
    ],
}

# ── Transition map ─────────────────────────────────────────────────────────────
# Each action defines whether it's terminal and the probability of reaching a
# terminal state (prob_terminal > 0 means stochastic resolution).

DEFAULT_ACTION_OUTCOMES: dict[str, dict] = {
    "Recontactar en X días": {
        "terminal": False,
        "prob_terminal": 0.0,
        "days_increment": (7, 21),
        "outcome_summary": "El contacto indica que necesita más tiempo. Se agenda recontacto.",
    },
    "Enviar documentación": {
        "terminal": False,
        "prob_terminal": 0.0,
        "days_increment": (5, 14),
        "outcome_summary": "El contacto aceptó recibir documentación comercial.",
    },
    "Agendar demo/reunión con especialista": {
        "terminal": False,
        "prob_terminal": 0.30,
        "terminal_status": "converted",
        "days_increment": (5, 10),
        "outcome_summary": "El contacto aceptó la demo. Se confirma fecha.",
    },
    "Escalar a manager del lead": {
        "terminal": False,
        "prob_terminal": 0.20,
        "terminal_status": "converted",
        "days_increment": (7, 14),
        "outcome_summary": "Se coordina reunión con dirección del cliente.",
    },
    "Esperar confirmación cliente": {
        "terminal": False,
        "prob_terminal": 0.15,
        "terminal_status": "converted",
        "days_increment": (7, 21),
        "outcome_summary": "El contacto está en proceso de aprobación interna.",
    },
    "Cerrar lead - no interesado": {
        "terminal": True,
        "terminal_status": "lost",
        "prob_terminal": 1.0,
        "days_increment": (0, 0),
        "outcome_summary": "El contacto declina continuar. Lead cerrado como perdido.",
    },
    "Cerrar lead - nurturing": {
        "terminal": True,
        "terminal_status": "nurturing",
        "prob_terminal": 1.0,
        "days_increment": (0, 0),
        "outcome_summary": "El contacto pospone la decisión. Lead en nurturing a largo plazo.",
    },
}

# ── LeadState ──────────────────────────────────────────────────────────────────

@dataclass
class LeadState:
    """Mutable state of a lead across simulation steps."""

    # Static fields
    lead_id: str
    contact_name: str
    contact_role: str
    company_name: str
    company_sector: str
    company_country: str
    company_city: str
    company_num_employees: int
    company_annual_revenue_eur: float
    lead_source: str
    initial_interest_notes: str

    # Mutable per-call state
    call_number: int = 1
    days_since_entry: int = 0
    days_since_last_call: int = 0
    prev_outcome: str = "PRIMERA_LLAMADA"
    prev_next_step: str = "PRIMERA_LLAMADA"
    current_transcript: str = ""

    # History and status
    trajectory: list = field(default_factory=list)
    status: str = "active"   # active | converted | lost | nurturing | max_steps_reached


# ── Template engine ────────────────────────────────────────────────────────────

def generate_synthetic_transcript(state: LeadState, agent_name: str) -> str:
    """Select and fill a transcript template based on the current state.

    Uses state.prev_next_step to pick the appropriate template bucket.
    Fills {contact_name}, {company_name}, {company_sector}, {agent_name}.
    """
    key = state.prev_next_step
    if key not in TRANSCRIPT_TEMPLATES:
        # Fallback to primera llamada templates
        key = "PRIMERA_LLAMADA"

    template = random.choice(TRANSCRIPT_TEMPLATES[key])
    return template.format(
        contact_name=state.contact_name,
        company_name=state.company_name,
        company_sector=state.company_sector,
        agent_name=agent_name,
    )


# ── Core simulation ────────────────────────────────────────────────────────────

def simulate_lead(
    lead_data: dict,
    model_path: str | Path | None = None,
    max_steps: int = 10,
    action_outcomes: dict | None = None,
    arts: dict | None = None,
    emb_model=None,
    seed: int = 42,
) -> dict:
    """Simulate a single lead's journey through the sales pipeline.

    Parameters
    ----------
    lead_data      : dict with static lead fields (same keys as LeadState fields)
    model_path     : path to model .joblib (used if arts is None)
    max_steps      : maximum number of prediction steps
    action_outcomes: transition map (default: DEFAULT_ACTION_OUTCOMES)
    arts           : pre-loaded artefacts dict (avoids joblib.load in batch mode)
    emb_model      : pre-loaded SentenceTransformer (avoids re-init in batch mode)
    seed           : random seed for stochastic transitions

    Returns
    -------
    dict with keys:
        lead_id, trajectory, final_status, total_steps, confidence_avg, total_days
    """
    rng = random.Random(seed)
    ao = action_outcomes or DEFAULT_ACTION_OUTCOMES

    # Load artefacts if not provided (single-lead convenience)
    if arts is None or emb_model is None:
        if model_path is None:
            from src.config import resolve_path
            model_path = resolve_path("model")
        if arts is None:
            arts = joblib.load(model_path)
        if emb_model is None:
            from sentence_transformers import SentenceTransformer
            emb_name = arts.get("embedding_model", CONFIG["embedding_model"])
            emb_model = SentenceTransformer(emb_name)

    # Agent name (random, reproducible)
    agent_names = CONFIG.get("agent_names", ["Agente MoveUp"])
    agent_name = rng.choice(agent_names)

    # Initialise state
    state = LeadState(**{k: v for k, v in lead_data.items() if k in LeadState.__dataclass_fields__})

    for step in range(max_steps):
        # Generate transcript
        state.current_transcript = generate_synthetic_transcript(state, agent_name)

        # Predict
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

        # Record step
        state.trajectory.append({
            "step": step + 1,
            "call_number": state.call_number,
            "days_since_entry": state.days_since_entry,
            "prev_next_step": state.prev_next_step,
            "predicted_next_step": predicted_action,
            "confidence": confidence,
        })

        # Resolve action outcome
        outcome_cfg = ao.get(predicted_action, {"terminal": False, "prob_terminal": 0.0, "days_increment": (7, 14)})
        is_terminal = outcome_cfg.get("terminal", False)

        if not is_terminal and outcome_cfg.get("prob_terminal", 0) > 0:
            is_terminal = rng.random() < outcome_cfg["prob_terminal"]
            if is_terminal:
                state.status = outcome_cfg.get("terminal_status", "converted")
        elif is_terminal:
            state.status = outcome_cfg.get("terminal_status", "lost")

        if is_terminal:
            break

        # Update state for next step
        days_lo, days_hi = outcome_cfg.get("days_increment", (7, 14))
        days_gap = rng.randint(days_lo, max(days_lo, days_hi))
        state.days_since_entry += days_gap
        state.days_since_last_call = days_gap
        state.call_number += 1
        state.prev_next_step = predicted_action
        state.prev_outcome = outcome_cfg.get("outcome_summary", predicted_action)
    else:
        # Loop exhausted without terminal state
        state.status = "max_steps_reached"

    total_steps = len(state.trajectory)
    confidences = [s["confidence"] for s in state.trajectory]
    confidence_avg = round(float(np.mean(confidences)) if confidences else 0.0, 4)

    return {
        "lead_id": state.lead_id,
        "company_name": state.company_name,
        "company_sector": state.company_sector,
        "trajectory": state.trajectory,
        "final_status": state.status,
        "total_steps": total_steps,
        "confidence_avg": confidence_avg,
        "total_days": state.days_since_entry,
    }


# ── Batch simulation ───────────────────────────────────────────────────────────

def _generate_synthetic_lead(rng: random.Random) -> dict:
    """Create a random lead dict from CONFIG pools (no GPT)."""
    sectors = CONFIG.get("sectors", ["Consultoría IT"])
    lead_sources = CONFIG.get("lead_sources", ["LinkedIn"])
    contact_roles = CONFIG.get("contact_roles", ["CEO"])
    cities_data = CONFIG.get("cities", [{"city": "Madrid", "country": "España"}])
    city_entry = rng.choice(cities_data)

    employees = rng.choice([
        rng.randint(10, 50),
        rng.randint(51, 250),
        rng.randint(251, 1000),
        rng.randint(1001, 5000),
    ])
    revenue = employees * rng.uniform(80_000, 200_000)

    first_names = ["Laura", "Carlos", "María", "Javier", "Ana", "Pablo", "Sofía", "Diego"]
    last_names = ["García", "Martínez", "López", "Fernández", "Ruiz", "González", "Torres"]

    contact_name = f"{rng.choice(first_names)} {rng.choice(last_names)}"
    company_name = f"{rng.choice(['Global', 'Tech', 'Iberia', 'Euro', 'Digital'])} " \
                   f"{rng.choice(['Solutions', 'Group', 'Corp', 'Services', 'Partners'])}"

    return {
        "lead_id": str(uuid.uuid4())[:8],
        "contact_name": contact_name,
        "contact_role": rng.choice(contact_roles),
        "company_name": company_name,
        "company_sector": rng.choice(sectors),
        "company_country": city_entry["country"],
        "company_city": city_entry["city"],
        "company_num_employees": employees,
        "company_annual_revenue_eur": round(revenue, 0),
        "lead_source": rng.choice(lead_sources),
        "initial_interest_notes": (
            f"Empresa del sector {rng.choice(sectors)} con interés en movilidad corporativa."
        ),
    }


def run_simulation_batch(
    n_leads: int = 50,
    model_path: str | Path | None = None,
    max_steps: int = 10,
    seed: int = 42,
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Simulate n_leads synthetic leads and return a summary DataFrame.

    Artefacts and SentenceTransformer are loaded ONCE for efficiency.

    Parameters
    ----------
    n_leads    : number of leads to simulate
    model_path : path to .joblib (default from config)
    max_steps  : max steps per lead
    seed       : base seed (each lead gets seed + i for reproducibility)
    output_dir : where to save results (default: experiments/simulations/)

    Returns
    -------
    pd.DataFrame with one row per simulated lead
    """
    # Resolve output
    if output_dir is None:
        from src.config import _PROJECT_ROOT  # type: ignore[attr-defined]
        experiments_dir = CONFIG.get("experiments_dir", "experiments")
        output_dir = _PROJECT_ROOT / experiments_dir / "simulations"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model path
    if model_path is None:
        from src.config import resolve_path
        model_path = resolve_path("model")

    print(f"  Loading artefacts from {model_path} …")
    arts = joblib.load(model_path)
    from sentence_transformers import SentenceTransformer
    emb_name = arts.get("embedding_model", CONFIG["embedding_model"])
    emb_model = SentenceTransformer(emb_name)

    rng_global = random.Random(seed)
    results = []

    print(f"  Simulating {n_leads} leads (max_steps={max_steps}) …")
    for i in range(n_leads):
        lead_data = _generate_synthetic_lead(rng_global)
        result = simulate_lead(
            lead_data,
            arts=arts,
            emb_model=emb_model,
            max_steps=max_steps,
            seed=seed + i,
        )
        results.append({
            "lead_id": result["lead_id"],
            "company_name": result["company_name"],
            "company_sector": result["company_sector"],
            "final_status": result["final_status"],
            "total_steps": result["total_steps"],
            "total_days": result["total_days"],
            "confidence_avg": result["confidence_avg"],
        })

    df = pd.DataFrame(results)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"results_{timestamp}.json"
    save_json(results, json_path)

    stats = {
        "n_leads": n_leads,
        "timestamp": timestamp,
        "max_steps": max_steps,
        "status_distribution": df["final_status"].value_counts().to_dict(),
        "avg_steps": round(float(df["total_steps"].mean()), 2),
        "avg_days": round(float(df["total_days"].mean()), 1),
        "avg_confidence": round(float(df["confidence_avg"].mean()), 4),
        "conversion_rate": round(float((df["final_status"] == "converted").mean()), 4),
        "loss_rate": round(float((df["final_status"] == "lost").mean()), 4),
    }
    save_json(stats, output_dir / f"stats_{timestamp}.json")

    print(f"  Results saved → {json_path}")
    return df


# ── Strategy comparison ────────────────────────────────────────────────────────

def compare_strategies(
    leads_batch: list[dict],
    strategies: list[dict],
    model_path: str | Path | None = None,
) -> pd.DataFrame:
    """Compare different action-outcome strategies on the same set of leads.

    Parameters
    ----------
    leads_batch : list of lead dicts (same format as _generate_synthetic_lead)
    strategies  : list of {"name": str, "action_outcomes": dict}
    model_path  : path to .joblib

    Returns
    -------
    pd.DataFrame with one row per strategy and summary metrics
    """
    if model_path is None:
        from src.config import resolve_path
        model_path = resolve_path("model")

    arts = joblib.load(model_path)
    from sentence_transformers import SentenceTransformer
    emb_name = arts.get("embedding_model", CONFIG["embedding_model"])
    emb_model = SentenceTransformer(emb_name)

    strategy_rows = []
    for strategy in strategies:
        name = strategy["name"]
        ao = strategy.get("action_outcomes", DEFAULT_ACTION_OUTCOMES)
        results = []

        print(f"  Strategy '{name}': simulating {len(leads_batch)} leads …")
        for i, lead_data in enumerate(leads_batch):
            res = simulate_lead(
                lead_data,
                arts=arts,
                emb_model=emb_model,
                action_outcomes=ao,
                seed=42 + i,
            )
            results.append(res)

        statuses = [r["final_status"] for r in results]
        steps = [r["total_steps"] for r in results]
        days = [r["total_days"] for r in results]
        confs = [r["confidence_avg"] for r in results]

        strategy_rows.append({
            "strategy_name": name,
            "n_leads": len(leads_batch),
            "conversion_rate": round(sum(s == "converted" for s in statuses) / len(statuses), 4),
            "loss_rate": round(sum(s == "lost" for s in statuses) / len(statuses), 4),
            "nurturing_rate": round(sum(s == "nurturing" for s in statuses) / len(statuses), 4),
            "max_steps_rate": round(sum(s == "max_steps_reached" for s in statuses) / len(statuses), 4),
            "avg_steps_to_terminal": round(float(np.mean(steps)), 2),
            "avg_total_days": round(float(np.mean(days)), 1),
            "avg_confidence": round(float(np.mean(confs)), 4),
        })

    return pd.DataFrame(strategy_rows)


# ── Predefined strategies for TFM comparison ──────────────────────────────────

def _aggressive_strategy() -> dict:
    """Strategy with higher demo/conversion probability and shorter recontact delays."""
    ao = {k: dict(v) for k, v in DEFAULT_ACTION_OUTCOMES.items()}
    ao["Agendar demo/reunión con especialista"]["prob_terminal"] = 0.50
    ao["Escalar a manager del lead"]["prob_terminal"] = 0.35
    ao["Esperar confirmación cliente"]["prob_terminal"] = 0.25
    ao["Recontactar en X días"]["days_increment"] = (3, 10)
    return {"name": "aggressive", "action_outcomes": ao}


def _conservative_strategy() -> dict:
    """Strategy with lower probabilities and longer delays between actions."""
    ao = {k: dict(v) for k, v in DEFAULT_ACTION_OUTCOMES.items()}
    ao["Agendar demo/reunión con especialista"]["prob_terminal"] = 0.15
    ao["Escalar a manager del lead"]["prob_terminal"] = 0.10
    ao["Esperar confirmación cliente"]["prob_terminal"] = 0.08
    ao["Recontactar en X días"]["days_increment"] = (14, 30)
    return {"name": "conservative", "action_outcomes": ao}


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    n_leads = 50
    for arg in sys.argv[1:]:
        if arg.startswith("--n-leads="):
            n_leads = int(arg.split("=")[1])
        elif arg.startswith("--n-leads"):
            idx = sys.argv.index(arg)
            if idx + 1 < len(sys.argv):
                n_leads = int(sys.argv[idx + 1])

    do_comparison = "--strategy" in sys.argv or "--compare" in sys.argv

    print("=== MoveUp — Lead Simulation ===\n")

    if do_comparison:
        print(f"[Mode] Strategy comparison on {n_leads} leads\n")
        rng = random.Random(42)
        leads = [_generate_synthetic_lead(rng) for _ in range(n_leads)]
        strategies = [
            {"name": "default", "action_outcomes": DEFAULT_ACTION_OUTCOMES},
            _aggressive_strategy(),
            _conservative_strategy(),
        ]
        df_cmp = compare_strategies(leads, strategies)
        print("\n── Strategy Comparison ──")
        print(df_cmp.to_string(index=False))

        from src.config import _PROJECT_ROOT  # type: ignore[attr-defined]
        out = _PROJECT_ROOT / CONFIG.get("experiments_dir", "experiments") / "simulations"
        out.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        df_cmp.to_csv(out / f"strategy_comparison_{ts}.csv", index=False)
        print(f"\nSaved → {out / f'strategy_comparison_{ts}.csv'}")

    else:
        print(f"[Mode] Batch simulation ({n_leads} leads)\n")
        df = run_simulation_batch(n_leads=n_leads, max_steps=10)

        print("\n── Simulation Summary ──")
        status_dist = df["final_status"].value_counts()
        print(f"  Status distribution:\n{status_dist.to_string()}")
        print(f"  Avg steps per lead  : {df['total_steps'].mean():.2f}")
        print(f"  Avg days per lead   : {df['total_days'].mean():.1f}")
        print(f"  Avg confidence      : {df['confidence_avg'].mean():.4f}")
        print(f"  Conversion rate     : {(df['final_status']=='converted').mean()*100:.1f}%")
        print(f"  Loss rate           : {(df['final_status']=='lost').mean()*100:.1f}%")

    print("\nDone.")


if __name__ == "__main__":
    main()
