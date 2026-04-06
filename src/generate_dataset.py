"""
generate_dataset.py
───────────────────
Generates a synthetic CRM dataset for MoveUp using the OpenAI API (GPT-4o).
Each lead gets 1-3 call interactions.  One CSV row is written per call.

Usage:
    python -m src.generate_dataset
"""
import json
import os
import random
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from openai import OpenAI

from src.config import CONFIG, resolve_path

# ── Constants pulled from config ──────────────────────────────────────────────
NEXT_STEP_CATEGORIES: list[str] = CONFIG["next_step_categories"]
NUM_LEADS: int = CONFIG["num_leads"]
OPENAI_MODEL: str = CONFIG["openai_model"]
MAX_RETRIES: int = CONFIG["max_retries"]
RATE_LIMIT_N: int = CONFIG["rate_limit_every_n_leads"]
PARTIAL_SAVE_N: int = CONFIG["partial_save_every_n_leads"]
NULL_FILL: str = CONFIG["null_fill_value"]

SECTORS: list[str] = CONFIG["sectors"]
LEAD_SOURCES: list[str] = CONFIG["lead_sources"]
CONTACT_ROLES: list[str] = CONFIG["contact_roles"]
CITIES: list[dict] = CONFIG["cities"]
AGENT_NAMES: list[str] = CONFIG["agent_names"]
SPECIALIST_NAMES: list[str] = CONFIG["specialist_names"]

_STATUS_PROBS = CONFIG["final_status_probs"]
FINAL_STATUSES = list(_STATUS_PROBS.keys())
FINAL_STATUS_WEIGHTS = list(_STATUS_PROBS.values())

# ── Helpers ───────────────────────────────────────────────────────────────────

def _rand_company_name() -> str:
    prefixes = ["Tech", "Global", "Smart", "Pro", "Iberia", "Euro", "Nova",
                "Alpha", "Prime", "Grupo", "Corp", "Digi", "Flex", "Max", "Omega"]
    suffixes = ["Solutions", "Services", "Group", "Partners", "Consulting",
                "Logistics", "Systems", "Innovations", "Dynamics", "Networks"]
    return f"{random.choice(prefixes)} {random.choice(suffixes)}"


def _rand_person_name() -> str:
    first_names = [
        "Carlos", "Laura", "Miguel", "Sofía", "Alejandro", "Isabel", "David",
        "Marta", "Javier", "Ana", "Roberto", "Elena", "Fernando", "Cristina",
        "Alberto", "Lucía", "Rafael", "Patricia", "Jorge", "María",
    ]
    last_names = [
        "García", "Martínez", "López", "Sánchez", "Fernández", "González",
        "Rodríguez", "Pérez", "Gómez", "Díaz", "Torres", "Ruiz", "Navarro",
        "Moreno", "Jiménez", "Álvarez", "Romero", "Iglesias", "Serrano", "Vega",
    ]
    return f"{random.choice(first_names)} {random.choice(last_names)}"


def generate_lead_metadata() -> dict:
    """Return a dict with all static fields for a new lead."""
    city_entry = random.choice(CITIES)
    employees = random.choice(
        [random.randint(10, 50), random.randint(50, 250),
         random.randint(250, 1000), random.randint(1000, 10000)]
    )
    revenue = round(employees * random.uniform(60_000, 300_000) / 1000) * 1000
    lead_entry_date = datetime.today() - timedelta(days=random.randint(7, 180))

    return {
        "lead_id": str(uuid.uuid4())[:8],
        "contact_name": _rand_person_name(),
        "contact_role": random.choice(CONTACT_ROLES),
        "company_name": _rand_company_name(),
        "company_sector": random.choice(SECTORS),
        "company_country": city_entry["country"],
        "company_city": city_entry["city"],
        "company_num_employees": employees,
        "company_annual_revenue_eur": revenue,
        "lead_source": random.choice(LEAD_SOURCES),
        "lead_entry_date": lead_entry_date.strftime("%Y-%m-%d"),
        "agent_name": random.choice(AGENT_NAMES),
    }


def assign_lead_journey(meta: dict) -> tuple[str, int]:
    """Return (final_status, num_calls) with realistic probability biases."""
    employees = meta["company_num_employees"]
    source = meta["lead_source"]

    # Adjust weights based on company size and lead source
    weights = list(FINAL_STATUS_WEIGHTS)  # [Converted, Lost, Nurturing, In Progress]
    if employees >= 500:
        weights[0] *= 1.5   # larger companies convert more
    if source in ("Formulario web", "Referencia de cliente", "Inbound / Blog"):
        weights[0] *= 1.4   # inbound converts more
    if source == "Llamada en frío":
        weights[1] *= 1.3   # cold calls lose more
    if employees < 50:
        weights[0] *= 0.7   # small companies convert less

    # Normalise
    total = sum(weights)
    weights = [w / total for w in weights]

    final_status = random.choices(FINAL_STATUSES, weights=weights, k=1)[0]

    call_weights_map = CONFIG["calls_per_status"]
    call_w = call_weights_map.get(final_status, [1, 1, 1])
    num_calls = random.choices([1, 2, 3], weights=call_w[:3], k=1)[0]

    return final_status, num_calls


def build_generation_prompt(meta: dict, final_status: str, num_calls: int) -> str:
    """Build a GPT-4o prompt that produces a realistic multi-call journey."""
    categories_str = "\n".join(f"  - \"{c}\"" for c in NEXT_STEP_CATEGORIES)
    specialist = random.choice(SPECIALIST_NAMES)

    status_guidance = {
        "Converted": (
            "La empresa está muy interesada en MoveUp. El contacto menciona crecimiento, "
            "alto volumen de desplazamientos, urgencia por solucionar costes de movilidad. "
            "El tono es positivo y proactivo. Acaban firmando o confirmando."
        ),
        "Lost": (
            "La empresa tiene objeciones claras: presupuesto insuficiente, ya tienen proveedor, "
            "mala coyuntura, reorganización interna. El contacto es escéptico o poco comprometido. "
            "El lead acaba cerrándose como no interesado."
        ),
        "Nurturing": (
            "La empresa muestra interés moderado pero no hay urgencia. Quieren información, "
            "comparar opciones, o esperan aprobación presupuestaria. El proceso queda en pausa."
        ),
        "In Progress": (
            "El proceso está activo. Hay interés real pero aún no se ha concluido. "
            "El contacto pide más detalles, demos o reuniones con decisores."
        ),
    }

    guidance = status_guidance.get(final_status, "")

    return f"""Eres un generador de datos sintéticos para un CRM de ventas B2B.
La empresa es MoveUp, un servicio de movilidad corporativa (tipo Uber for Business).

Genera el recorrido de venta de un lead con las siguientes características:
- Empresa: {meta['company_name']} ({meta['company_sector']}, {meta['company_city']}, {meta['company_country']})
- Empleados: {meta['company_num_employees']}, Facturación: {meta['company_annual_revenue_eur']:,} €
- Contacto: {meta['contact_name']}, {meta['contact_role']}
- Fuente del lead: {meta['lead_source']}
- Agente de ventas: {meta['agent_name']}
- Especialista disponible: {specialist}
- Número de llamadas: {num_calls}
- Estado final: {final_status}

Contexto narrativo:
{guidance}

INSTRUCCIONES:
1. Genera diálogos realistas en ESPAÑOL entre el agente y el contacto.
2. Cada transcripción debe tener entre 8 y 20 turnos de diálogo (Agente: ... / Contacto: ...).
3. Los transcripts deben contener señales aprendibles: menciones a volumen de viajes, presupuesto, urgencia, objeciones, etc.
4. El campo "next_step" de cada llamada DEBE ser exactamente uno de:
{categories_str}
5. Para la última llamada, "next_step" debe ser coherente con el estado final {final_status}.
6. "days_until_next_call": número de días hasta la siguiente acción (0 si es la última llamada).

Devuelve ÚNICAMENTE JSON válido con este esquema exacto (sin texto adicional, sin markdown):
{{
  "contact_name": "...",
  "company_name": "...",
  "initial_interest_notes": "Breve nota sobre el interés inicial del lead (1-2 frases)",
  "calls": [
    {{
      "transcript": "Agente: ...\\nContacto: ...\\nAgente: ...",
      "outcome": "Resumen breve de lo que ocurrió en la llamada",
      "next_step": "exactamente uno de los 7 valores",
      "days_until_next_call": 0
    }}
  ]
}}
Genera exactamente {num_calls} elemento(s) en el array "calls"."""


def generate_lead_with_llm(client: OpenAI, meta: dict,
                            final_status: str, num_calls: int) -> dict | None:
    """Call GPT-4o and return parsed JSON, retrying up to MAX_RETRIES times."""
    prompt = build_generation_prompt(meta, final_status, num_calls)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.85,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content
            data = json.loads(raw)

            # Validate structure
            if "calls" not in data or not isinstance(data["calls"], list):
                raise ValueError("Missing 'calls' array in LLM response")
            if len(data["calls"]) != num_calls:
                raise ValueError(
                    f"Expected {num_calls} calls, got {len(data['calls'])}"
                )
            for call in data["calls"]:
                ns = call.get("next_step", "")
                if ns not in NEXT_STEP_CATEGORIES:
                    raise ValueError(f"Invalid next_step: '{ns}'")

            return data

        except (json.JSONDecodeError, ValueError, KeyError) as exc:
            print(f"    [attempt {attempt}/{MAX_RETRIES}] Validation error: {exc}")
            if attempt == MAX_RETRIES:
                print(f"    Skipping lead after {MAX_RETRIES} failed attempts.")
                return None
        except Exception as exc:  # network / API errors
            print(f"    [attempt {attempt}/{MAX_RETRIES}] API error: {exc}")
            time.sleep(5)
            if attempt == MAX_RETRIES:
                return None

    return None


def lead_to_rows(meta: dict, llm_data: dict, final_status: str) -> list[dict]:
    """Convert one lead's LLM output into a list of interaction rows."""
    calls = llm_data["calls"]
    rows = []
    entry_date = datetime.strptime(meta["lead_entry_date"], "%Y-%m-%d")
    call_timestamp = entry_date + timedelta(days=random.randint(1, 7))
    prev_outcome = ""
    prev_next_step = ""

    for i, call in enumerate(calls):
        call_number = i + 1
        days_since_entry = (call_timestamp - entry_date).days
        days_since_last_call = 0 if i == 0 else (
            call_timestamp - datetime.strptime(
                rows[-1]["call_timestamp"], "%Y-%m-%d"
            )
        ).days

        row = {
            "interaction_id": str(uuid.uuid4())[:12],
            "lead_id": meta["lead_id"],
            "contact_name": llm_data.get("contact_name", meta["contact_name"]),
            "contact_role": meta["contact_role"],
            "company_name": llm_data.get("company_name", meta["company_name"]),
            "company_sector": meta["company_sector"],
            "company_country": meta["company_country"],
            "company_city": meta["company_city"],
            "company_num_employees": meta["company_num_employees"],
            "company_annual_revenue_eur": meta["company_annual_revenue_eur"],
            "lead_source": meta["lead_source"],
            "lead_entry_date": meta["lead_entry_date"],
            "initial_interest_notes": llm_data.get("initial_interest_notes", ""),
            "call_number": call_number,
            "call_timestamp": call_timestamp.strftime("%Y-%m-%d"),
            "days_since_entry": days_since_entry,
            "days_since_last_call": days_since_last_call,
            "prev_outcome": prev_outcome if prev_outcome else NULL_FILL,
            "prev_next_step": prev_next_step if prev_next_step else NULL_FILL,
            "current_transcript": call["transcript"],
            "current_outcome": call["outcome"],
            "next_step": call["next_step"],
            "final_status": final_status,
        }
        rows.append(row)

        prev_outcome = call["outcome"]
        prev_next_step = call["next_step"]
        days_gap = call.get("days_until_next_call", random.randint(3, 14))
        call_timestamp += timedelta(days=max(1, days_gap))

    return rows


def _find_latest_partial(output_dir: Path) -> tuple[list[dict], int]:
    """Return (rows_so_far, last_lead_idx) from the most advanced partial CSV."""
    partials = sorted(output_dir.glob("partial_*.csv"))
    if not partials:
        return [], 0
    latest = partials[-1]
    last_idx = int(latest.stem.split("_")[1])
    rows = pd.read_csv(latest, encoding="utf-8-sig").to_dict(orient="records")
    print(f"  Resuming from partial save: {latest.name} ({len(rows)} rows, {last_idx} leads done)")
    return rows, last_idx


def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY not set. Export it before running this script."
        )

    client = OpenAI(api_key=api_key)
    output_path = resolve_path("raw_data")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows, start_idx = _find_latest_partial(output_path.parent)
    skipped = 0

    print(f"Generating {NUM_LEADS} leads … (starting from lead {start_idx + 1})")

    for idx in range(start_idx + 1, NUM_LEADS + 1):
        print(f"  Lead {idx}/{NUM_LEADS}", end="", flush=True)

        meta = generate_lead_metadata()
        final_status, num_calls = assign_lead_journey(meta)
        print(f" [{final_status}, {num_calls} calls]", end="", flush=True)

        llm_data = generate_lead_with_llm(client, meta, final_status, num_calls)
        if llm_data is None:
            skipped += 1
            print(" — SKIPPED")
            continue

        rows = lead_to_rows(meta, llm_data, final_status)
        all_rows.extend(rows)
        print(f" → {len(rows)} rows")

        # Rate limiting
        if idx % RATE_LIMIT_N == 0:
            time.sleep(1)

        # Partial save
        if idx % PARTIAL_SAVE_N == 0 and all_rows:
            partial_path = output_path.parent / f"partial_{idx}.csv"
            pd.DataFrame(all_rows).to_csv(
                partial_path, index=False, encoding="utf-8-sig"
            )
            print(f"  [partial save → {partial_path}]")

    if not all_rows:
        print("No rows generated. Check OpenAI API key and quota.")
        return

    df = pd.DataFrame(all_rows)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\nDone. {len(df)} rows saved to {output_path}  (skipped {skipped} leads)")


if __name__ == "__main__":
    main()
