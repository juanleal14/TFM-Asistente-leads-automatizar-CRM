"""
MoveUp CRM - Generador de Dataset Sintético
=============================================
Genera 500 leads sintéticos con transcripts realistas usando OpenAI GPT-4o.
Cada lead tiene entre 1 y 3 llamadas, generando una fila por interacción.

Requisitos:
    pip install openai pandas

Uso:
    export OPENAI_API_KEY="tu-api-key"
    python generate_dataset.py
"""

import openai
import pandas as pd
import json
import random
import time
import os
from datetime import datetime, timedelta

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────
NUM_LEADS = 500
OUTPUT_FILE = "moveup_crm_dataset.csv"
MODEL = "gpt-4o"

# Categorías cerradas de next_step (target variable)
NEXT_STEPS = [
    "Recontactar en X días",
    "Enviar documentación",
    "Agendar demo/reunión con especialista",
    "Escalar a manager del lead",
    "Cerrar lead - no interesado",
    "Cerrar lead - nurturing",
    "Esperar confirmación cliente"
]

# Distribución realista de resultados finales
FINAL_STATUS_WEIGHTS = {
    "Converted": 0.30,
    "Lost": 0.40,
    "Nurturing": 0.20,
    "In Progress": 0.10
}

# Número de llamadas según el final_status
CALLS_BY_STATUS = {
    "Converted": [2, 3, 3, 3],
    "Lost": [1, 1, 2, 2, 3],
    "Nurturing": [1, 2, 2],
    "In Progress": [1, 2]
}

# Pools de datos para variabilidad
SECTORS = [
    "Consultoría IT", "Logística", "Distribución alimentaria", "Retail",
    "Farmacéutica", "Telecomunicaciones", "Seguros", "Banca",
    "Construcción", "Energía", "Automoción", "Turismo y hostelería",
    "Educación", "Sanidad privada", "Servicios legales", "Inmobiliaria",
    "Marketing y publicidad", "Recursos humanos", "Transporte",
    "Manufactura industrial"
]

LEAD_SOURCES = [
    "Formulario web", "Llamada en frío (outbound)", "LinkedIn",
    "Referencia de cliente", "Evento/Feria", "Webinar MoveUp",
    "Google Ads", "Partner/Revendedor", "Email marketing"
]

CONTACT_ROLES = [
    "Office Manager", "Head of People", "Director de Operaciones",
    "CFO", "CEO", "Responsable de Compras", "Facility Manager",
    "Director Financiero", "HR Manager", "COO",
    "Responsable de Flota", "Travel Manager", "Director General",
    "Responsable de Logística", "Admin Manager"
]

CITIES = [
    ("Madrid", "España"), ("Barcelona", "España"), ("Valencia", "España"),
    ("Sevilla", "España"), ("Bilbao", "España"), ("Málaga", "España"),
    ("Zaragoza", "España"), ("Murcia", "España"), ("Palma", "España"),
    ("A Coruña", "España"), ("Alicante", "España"), ("Valladolid", "España"),
    ("Vigo", "España"), ("San Sebastián", "España"), ("Pamplona", "España"),
    ("Lisboa", "Portugal"), ("Oporto", "Portugal"),
    ("Ciudad de México", "México"), ("Bogotá", "Colombia"),
    ("Buenos Aires", "Argentina")
]

AGENT_NAMES = ["Carlos", "Laura", "Ana", "Miguel", "Sofía", "Javier", "Elena", "Pablo"]
SPECIALIST_NAMES = ["Ana", "Roberto", "Lucía", "Marcos"]


# ─────────────────────────────────────────────
# GENERACIÓN DE METADATOS DEL LEAD
# ─────────────────────────────────────────────
def generate_lead_metadata(lead_num: int) -> dict:
    """Genera los datos estáticos de un lead (empresa + contacto)."""
    city, country = random.choice(CITIES)
    num_employees = random.choice(
        [random.randint(5, 30)] * 3 +
        [random.randint(31, 150)] * 4 +
        [random.randint(151, 500)] * 2 +
        [random.randint(501, 5000)] * 1
    )
    revenue = int(num_employees * random.uniform(40000, 120000))
    sector = random.choice(SECTORS)
    lead_source = random.choice(LEAD_SOURCES)

    entry_date = datetime(2025, 1, 1) + timedelta(
        days=random.randint(0, 348)
    )

    return {
        "lead_id": f"L-{lead_num:04d}",
        "contact_role": random.choice(CONTACT_ROLES),
        "company_sector": sector,
        "company_country": country,
        "company_city": city,
        "company_num_employees": num_employees,
        "company_annual_revenue_eur": revenue,
        "lead_source": lead_source,
        "lead_entry_date": entry_date.strftime("%Y-%m-%d"),
        "entry_date_obj": entry_date
    }


def assign_lead_journey(metadata: dict) -> dict:
    """Asigna final_status y número de llamadas basado en perfil de empresa."""
    num_emp = metadata["company_num_employees"]
    source = metadata["lead_source"]

    weights = dict(FINAL_STATUS_WEIGHTS)
    if num_emp > 100:
        weights["Converted"] += 0.10
        weights["Lost"] -= 0.10
    elif num_emp < 20:
        weights["Converted"] -= 0.10
        weights["Lost"] += 0.10

    if source in ["Formulario web", "Referencia de cliente", "Webinar MoveUp"]:
        weights["Converted"] += 0.05
        weights["Lost"] -= 0.05
    elif source == "Llamada en frío (outbound)":
        weights["Converted"] -= 0.05
        weights["Lost"] += 0.05

    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}

    final_status = random.choices(
        list(weights.keys()),
        weights=list(weights.values()),
        k=1
    )[0]

    num_calls = random.choice(CALLS_BY_STATUS[final_status])

    return {
        "final_status": final_status,
        "num_calls": num_calls
    }


# ─────────────────────────────────────────────
# GENERACIÓN DE TRANSCRIPTS CON GPT-4o
# ─────────────────────────────────────────────
def build_generation_prompt(metadata: dict, journey: dict) -> str:
    """Construye el prompt para que GPT-4o genere todo el ciclo de llamadas."""

    next_steps_list = "\n".join(f"  - {s}" for s in NEXT_STEPS)

    return f"""Genera datos sintéticos para un lead de CRM de la empresa MoveUp (servicio de movilidad corporativa estilo Uber for Business, para empresas que necesitan gestionar desplazamientos de empleados).

DATOS DEL LEAD:
- Lead ID: {metadata['lead_id']}
- Rol del contacto: {metadata['contact_role']}
- Sector de la empresa: {metadata['company_sector']}
- Ciudad: {metadata['company_city']}, {metadata['company_country']}
- Nº empleados: {metadata['company_num_employees']}
- Revenue anual: {metadata['company_annual_revenue_eur']}€
- Fuente de captación: {metadata['lead_source']}
- Fecha de entrada: {metadata['lead_entry_date']}
- Resultado final del lead: {journey['final_status']}
- Número total de llamadas: {journey['num_calls']}

CATEGORÍAS VÁLIDAS PARA next_step (USA EXACTAMENTE UNA DE ESTAS):
{next_steps_list}

INSTRUCCIONES:
1. Inventa un nombre realista para el contacto (nombre español/latino) y un nombre de empresa ficticio pero creíble para el sector.
2. Escribe unas "initial_interest_notes" breves (1-2 frases) describiendo por qué entró al CRM.
3. Para CADA llamada (de 1 a {journey['num_calls']}), genera:
   - "transcript": Un diálogo realista entre el agente de MoveUp y el contacto (8-15 líneas de diálogo). El agente se llama {random.choice(AGENT_NAMES)}. Si es la llamada final de un lead Converted y es llamada 3, el agente especialista se llama {random.choice(SPECIALIST_NAMES)}.
   - "outcome": Resumen breve del resultado de la llamada (1 frase).
   - "next_step": EXACTAMENTE una de las categorías válidas listadas arriba.
   - "days_until_next_call": Días hasta la siguiente llamada (solo si hay más llamadas después, si no, null).

4. La narrativa debe ser COHERENTE con el resultado final ({journey['final_status']}):
   - Si es "Converted": progresión positiva, el lead se va cualificando. La última llamada debería terminar con next_step "Esperar confirmación cliente" o "Agendar demo/reunión con especialista".
   - Si es "Lost": el lead pierde interés, hay objeciones insalvables (precio, timing, no encaja). Última llamada con "Cerrar lead - no interesado".
   - Si es "Nurturing": hay interés pero no es el momento. Última llamada con "Cerrar lead - nurturing".
   - Si es "In Progress": el lead aún está activo. Última llamada con "Recontactar en X días", "Enviar documentación" o "Agendar demo/reunión con especialista".

5. Haz que cada transcript tenga señales textuales que un modelo de ML pueda captar:
   - Leads que convierten: mencionan crecimiento, volumen alto, problemas actuales claros, urgencia.
   - Leads que se pierden: mencionan poco volumen, falta de presupuesto, "ya tenemos solución", poca necesidad.
   - Varía el estilo: algunos contactos son directos, otros divagan, algunos son muy formales, otros coloquiales.

Responde SOLO con un JSON válido con esta estructura (sin markdown, sin backticks, sin texto adicional):
{{
  "contact_name": "Nombre Apellido",
  "company_name": "Nombre Empresa S.L.",
  "initial_interest_notes": "...",
  "calls": [
    {{
      "transcript": "Agente: ...\\nContacto: ...\\nAgente: ...",
      "outcome": "...",
      "next_step": "...",
      "days_until_next_call": 14
    }}
  ]
}}"""


def generate_lead_with_llm(client: openai.OpenAI, metadata: dict, journey: dict, max_retries: int = 3) -> dict | None:
    """Llama a GPT-4o para generar los transcripts de un lead completo."""
    prompt = build_generation_prompt(metadata, journey)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                temperature=0.9,
                max_tokens=4000,
                messages=[
                    {
                        "role": "system",
                        "content": "Eres un generador de datos sintéticos para CRM. Responde SOLO con JSON válido, sin backticks, sin markdown, sin texto adicional."
                    },
                    {"role": "user", "content": prompt}
                ]
            )

            raw = response.choices[0].message.content.strip()
            # Limpiar posibles backticks
            raw = raw.replace("```json", "").replace("```", "").strip()
            data = json.loads(raw)

            # Validar estructura mínima
            if "calls" not in data or len(data["calls"]) != journey["num_calls"]:
                print(f"  ⚠️  Estructura inválida (intento {attempt + 1})")
                continue

            # Validar que next_step sea una categoría válida
            valid = True
            for call in data["calls"]:
                if call.get("next_step") not in NEXT_STEPS:
                    print(f"  ⚠️  next_step inválido: '{call.get('next_step')}' (intento {attempt + 1})")
                    valid = False
                    break
            if not valid:
                continue

            return data

        except json.JSONDecodeError as e:
            print(f"  ⚠️  Error JSON (intento {attempt + 1}): {e}")
        except openai.APIError as e:
            print(f"  ⚠️  Error API (intento {attempt + 1}): {e}")
            time.sleep(2)

    return None


# ─────────────────────────────────────────────
# TRANSFORMAR A FILAS (UNA POR INTERACCIÓN)
# ─────────────────────────────────────────────
def lead_to_rows(metadata: dict, journey: dict, llm_data: dict) -> list[dict]:
    """Convierte un lead generado en filas del dataset (una por llamada)."""
    rows = []
    calls = llm_data["calls"]
    entry_date = metadata["entry_date_obj"]

    call_date = entry_date + timedelta(days=random.randint(0, 2))

    prev_outcome = None
    prev_next_step = None
    prev_call_date = None

    for i, call in enumerate(calls):
        if i > 0:
            days_gap = calls[i - 1].get("days_until_next_call", random.randint(3, 14))
            if days_gap is None:
                days_gap = random.randint(3, 14)
            call_date = prev_call_date + timedelta(days=int(days_gap))

        days_since_entry = (call_date - entry_date).days
        days_since_last = (call_date - prev_call_date).days if prev_call_date else 0

        row = {
            "interaction_id": f"INT-{metadata['lead_id']}-C{i+1}",
            "lead_id": metadata["lead_id"],
            "contact_name": llm_data["contact_name"],
            "contact_role": metadata["contact_role"],
            "company_name": llm_data["company_name"],
            "company_sector": metadata["company_sector"],
            "company_country": metadata["company_country"],
            "company_city": metadata["company_city"],
            "company_num_employees": metadata["company_num_employees"],
            "company_annual_revenue_eur": metadata["company_annual_revenue_eur"],
            "lead_source": metadata["lead_source"],
            "lead_entry_date": metadata["lead_entry_date"],
            "initial_interest_notes": llm_data["initial_interest_notes"],
            "call_number": i + 1,
            "call_timestamp": call_date.strftime("%Y-%m-%d %H:%M:%S"),
            "days_since_entry": days_since_entry,
            "days_since_last_call": days_since_last,
            "prev_outcome": prev_outcome if prev_outcome else "",
            "prev_next_step": prev_next_step if prev_next_step else "",
            "current_transcript": call["transcript"],
            "current_outcome": call["outcome"],
            "next_step": call["next_step"],
            "final_status": journey["final_status"]
        }
        rows.append(row)

        prev_outcome = call["outcome"]
        prev_next_step = call["next_step"]
        prev_call_date = call_date

    return rows


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("MoveUp CRM - Generador de Dataset Sintético (GPT-4o)")
    print("=" * 60)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: Define OPENAI_API_KEY como variable de entorno.")
        print("   export OPENAI_API_KEY='tu-api-key'")
        return

    client = openai.OpenAI(api_key=api_key)

    all_rows = []
    failed_leads = []

    print(f"\n🚀 Generando {NUM_LEADS} leads...\n")

    for i in range(1, NUM_LEADS + 1):
        metadata = generate_lead_metadata(i)
        journey = assign_lead_journey(metadata)

        print(f"[{i:3d}/{NUM_LEADS}] {metadata['lead_id']} | "
              f"{metadata['company_sector']:25s} | "
              f"{metadata['company_num_employees']:5d} emp | "
              f"{journey['num_calls']} calls | "
              f"→ {journey['final_status']}", end=" ... ")

        llm_data = generate_lead_with_llm(client, metadata, journey)

        if llm_data:
            rows = lead_to_rows(metadata, journey, llm_data)
            all_rows.extend(rows)
            print(f"✅ ({len(rows)} filas)")
        else:
            failed_leads.append(metadata['lead_id'])
            print("❌ FALLO (3 intentos)")

        # Rate limiting
        if i % 10 == 0:
            time.sleep(1)

        # Guardado parcial cada 50 leads (por si se corta)
        if i % 50 == 0 and all_rows:
            partial_df = pd.DataFrame(all_rows)
            partial_df.to_csv(f"moveup_partial_{i}.csv", index=False, encoding="utf-8-sig")
            print(f"   💾 Guardado parcial: moveup_partial_{i}.csv ({len(all_rows)} filas)")

    # Guardar CSV final
    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 60)
    print(f"✅ Dataset generado: {OUTPUT_FILE}")
    print(f"   Total leads: {NUM_LEADS - len(failed_leads)}")
    print(f"   Total filas (interacciones): {len(all_rows)}")
    print(f"   Leads fallidos: {len(failed_leads)}")

    if failed_leads:
        print(f"   IDs fallidos: {', '.join(failed_leads)}")

    # Stats
    print(f"\n📊 Distribución de final_status:")
    for status, count in df["final_status"].value_counts().items():
        pct = count / len(df) * 100
        print(f"   {status}: {count} interacciones ({pct:.1f}%)")

    print(f"\n📊 Distribución de next_step (TARGET):")
    for step, count in df["next_step"].value_counts().items():
        pct = count / len(df) * 100
        print(f"   {step}: {count} ({pct:.1f}%)")

    print(f"\n📊 Distribución de call_number:")
    for cn, count in sorted(df["call_number"].value_counts().items()):
        print(f"   Llamada {cn}: {count} filas")

    # Limpiar parciales
    for i in range(50, NUM_LEADS + 1, 50):
        partial = f"moveup_partial_{i}.csv"
        if os.path.exists(partial):
            os.remove(partial)
            print(f"   🗑️  Eliminado parcial: {partial}")


if __name__ == "__main__":
    main()
