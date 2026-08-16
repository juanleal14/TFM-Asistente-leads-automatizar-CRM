"""
augment_minority.py
───────────────────
Genera leads sintéticos sesgados hacia una clase minoritaria del target,
y los APPENDIA al CSV principal del dataset.

Resuelve el problema de desbalanceo extremo (clase con <5 ejemplos no permite
StratifiedKFold con 5 folds y produce warnings en model_comparison.py).

Uso:
    python -m src.augment_minority "Escalar a manager del lead" --n 30
    python -m src.augment_minority "Recontactar en X días" --n 20

Argumentos:
    target_class : nombre exacto de la clase a boostar (entre comillas)
    --n          : número de leads a generar (default 30)

Comportamiento:
- Cada lead generado contiene la clase target en alguna de sus llamadas
- Se appendea al CSV definido en config.yaml (paths.raw_data)
- NO sobrescribe filas existentes; añade al final
- Tras correr este script:
    1. Borra data/processed/embeddings_cache.npz
    2. Reentrena: python -m src.train_model
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import uuid

import pandas as pd
from openai import OpenAI

from src.config import CONFIG, resolve_path
from src.generate_dataset import (
    NEXT_STEP_CATEGORIES,
    OPENAI_MODEL,
    RATE_LIMIT_N,
    assign_lead_journey,
    generate_lead_metadata,
    generate_lead_with_llm,
    lead_to_rows,
)


def augment_minority_class(
    target_class: str,
    n_leads: int,
    output_path=None,
) -> int:
    """Genera n_leads sesgados hacia target_class y appendea al CSV existente.

    Returns
    -------
    int : número de filas añadidas (varía porque cada lead da 1-3 llamadas).
    """
    if target_class not in NEXT_STEP_CATEGORIES:
        raise ValueError(
            f"Clase '{target_class}' no está en NEXT_STEP_CATEGORIES. "
            f"Valores válidos: {NEXT_STEP_CATEGORIES}"
        )

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY no configurada.")

    client = OpenAI(api_key=api_key)

    if output_path is None:
        output_path = resolve_path("raw_data")

    if not output_path.exists():
        raise FileNotFoundError(
            f"CSV principal no encontrado en {output_path}. "
            f"Genera primero el dataset base con `python -m src.generate_dataset`."
        )

    df_existing = pd.read_csv(output_path, encoding="utf-8-sig")
    initial_rows = len(df_existing)
    initial_class_count = (df_existing["next_step"] == target_class).sum()

    print(f"Dataset actual: {initial_rows} filas")
    print(f"Clase '{target_class}' antes: {initial_class_count} ejemplos")
    print(f"Generando {n_leads} leads sesgados hacia esa clase…\n")

    new_rows: list[dict] = []
    skipped = 0

    for idx in range(1, n_leads + 1):
        print(f"  Lead {idx}/{n_leads}", end="", flush=True)

        meta = generate_lead_metadata()
        # Asignamos un final_status compatible con la acción forzada.
        # Para "Escalar a manager", el lead suele estar In Progress o Converted
        # (no tiene sentido escalar y luego cerrar como perdido inmediatamente).
        if "Escalar" in target_class or "Agendar" in target_class:
            final_status = "In Progress"
        else:
            _, _ = assign_lead_journey(meta)  # consumir aleatoriedad
            final_status, _ = assign_lead_journey(meta)

        # Forzamos al menos 2 llamadas (la forzada + al menos una más para realismo)
        num_calls = 2 if idx % 2 == 0 else 3
        print(f" [{final_status}, {num_calls} calls]", end="", flush=True)

        llm_data = generate_lead_with_llm(
            client, meta, final_status, num_calls,
            forced_next_step=target_class,
        )
        if llm_data is None:
            skipped += 1
            print(" — SKIPPED")
            continue

        # Validar: el target_class realmente aparece en alguna llamada
        steps_generated = [c["next_step"] for c in llm_data["calls"]]
        if target_class not in steps_generated:
            skipped += 1
            print(f" — SKIPPED (target no apareció: {steps_generated})")
            continue

        rows = lead_to_rows(meta, llm_data, final_status)
        new_rows.extend(rows)
        print(f" → {len(rows)} filas (next_steps: {steps_generated})")

        if idx % RATE_LIMIT_N == 0:
            time.sleep(1)

    if not new_rows:
        print("\nNo se generaron filas. Aborta sin escribir.")
        return 0

    # ── Append al CSV ─────────────────────────────────────────────────────────
    df_new = pd.DataFrame(new_rows)
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    df_combined.to_csv(output_path, index=False, encoding="utf-8-sig")

    final_class_count = (df_combined["next_step"] == target_class).sum()

    print()
    print("=" * 70)
    print(f"  Filas añadidas: {len(new_rows)}  (skipped: {skipped} leads)")
    print(f"  Total filas en CSV: {initial_rows} → {len(df_combined)}")
    print(f"  Clase '{target_class}': {initial_class_count} → {final_class_count}")
    print("=" * 70)
    print("\nSiguientes pasos:")
    print("  1. rm data/processed/embeddings_cache.npz   # invalidar cache")
    print("  2. python -m src.train_model                # reentrenar")
    print("  3. python -m src.model_comparison           # verificar mejora")

    return len(new_rows)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Augment dataset by generating leads of a minority class."
    )
    parser.add_argument(
        "target_class",
        help='Nombre exacto de la clase a boostar (ej: "Escalar a manager del lead")',
    )
    parser.add_argument(
        "--n", type=int, default=30,
        help="Número de leads a generar (default: 30)",
    )
    args = parser.parse_args()

    augment_minority_class(target_class=args.target_class, n_leads=args.n)


if __name__ == "__main__":
    main()
