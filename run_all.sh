#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
#  run_all.sh
#  Ejecuta todo el pipeline TFM de principio a fin.
#
#  Uso:
#    bash run_all.sh             # pipeline completo sin tuning (~3 min)
#    bash run_all.sh --tune      # incluye RandomizedSearchCV (~10 min)
#    bash run_all.sh --no-llm    # demo final con templates en vez de GPT
# ─────────────────────────────────────────────────────────────────────────
set -e  # parar si algo falla

# ── Config ────────────────────────────────────────────────────────────────
INCLUDE_TUNING=false
DEMO_LLM_FLAG=""
for arg in "$@"; do
    case $arg in
        --tune)   INCLUDE_TUNING=true ;;
        --no-llm) DEMO_LLM_FLAG="--no-llm" ;;
    esac
done

# Cargar API key si existe .env.local
if [ -f .env.local ]; then
    set -a && source .env.local && set +a
fi

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
PY=.venv/bin/python

# ── Helper visual ─────────────────────────────────────────────────────────
section() {
    echo
    echo "═══════════════════════════════════════════════════════════════════"
    echo "  $1"
    echo "═══════════════════════════════════════════════════════════════════"
}

# ── 1. Tests ──────────────────────────────────────────────────────────────
section "1/9 — Tests unitarios + integración (~1 min)"
$PY -m pytest tests/ -v

# ── 2. Validación del dataset ─────────────────────────────────────────────
section "2/9 — Validación del dataset (5 auditorías)"
$PY -m src.validate_dataset

# ── 3. Entrenamiento ──────────────────────────────────────────────────────
section "3/9 — Entrenamiento XGBoost (CV + test)"
$PY -m src.train_model

# ── 4. Comparación de modelos ─────────────────────────────────────────────
section "4/9 — Comparación de modelos (Dummy / LogReg / RF / XGB / LGBM)"
$PY -m src.model_comparison

# ── 5. Tuning (opcional) ──────────────────────────────────────────────────
if [ "$INCLUDE_TUNING" = true ]; then
    section "5/9 — RandomizedSearchCV (~5-10 min) — pasado --tune"
    $PY -m src.tune_model
else
    section "5/9 — RandomizedSearchCV [SALTADO]  (lanza con --tune)"
fi

# ── 6. Predict standalone ─────────────────────────────────────────────────
section "6/9 — Predict standalone (smoke test)"
$PY -m src.predict

# ── 7. Simulación batch ───────────────────────────────────────────────────
section "7/9 — Simulación batch de 50 leads"
$PY -m src.simulate

section "7b/9 — Comparación de estrategias comerciales"
$PY -m src.simulate --compare

# ── 8. Histórico de experimentos ──────────────────────────────────────────
section "8/9 — Histórico de experimentos"
$PY -m src.experiment_tracker

# ── 9. Pipeline demo con LLM ──────────────────────────────────────────────
section "9/9 — Pipeline demo end-to-end (pieza estrella)"
$PY -m src.pipeline_demo --seed 7 $DEMO_LLM_FLAG

# ── Resumen ───────────────────────────────────────────────────────────────
echo
echo "═══════════════════════════════════════════════════════════════════"
echo "  ✓ run_all.sh completado"
echo "═══════════════════════════════════════════════════════════════════"
echo "  Outputs generados:"
echo "    · models/moveup_nextstep_model.joblib"
echo "    · plots/*.png"
echo "    · experiments/runs/*.json"
echo "    · experiments/model_comparison.csv"
echo "    · experiments/validation/*.json"
echo "    · experiments/simulations/*.json"
echo
