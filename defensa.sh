#!/usr/bin/env bash
# Secuencia mínima para la defensa TFM.
# Uso: bash defensa.sh

set -e  # parar si algo falla

# Cargar .env.local si existe
if [ -f .env.local ]; then
    set -a && source .env.local && set +a
fi

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
PY=.venv/bin/python

echo "═══════════════════════════════════════════════════════════════════"
echo "  PASO 1/3 — Tests (rigor del código)"
echo "═══════════════════════════════════════════════════════════════════"
$PY -m pytest tests/ -v

echo
echo "═══════════════════════════════════════════════════════════════════"
echo "  PASO 2/3 — Comparación de modelos (XGBoost vs baselines)"
echo "═══════════════════════════════════════════════════════════════════"
$PY -m src.model_comparison

echo
echo "═══════════════════════════════════════════════════════════════════"
echo "  PASO 3/3 — Pipeline demo end-to-end con LLM"
echo "═══════════════════════════════════════════════════════════════════"
$PY -m src.pipeline_demo --pause 1.5 --seed 7
