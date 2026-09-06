#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
#  defensa.sh
#  Secuencia mínima para la defensa TFM (~5 min): rigor de tests, comparación
#  de modelos frente a baselines, y demo visual end-to-end.
#
#  Uso: bash defensa.sh
# ─────────────────────────────────────────────────────────────────────────
set -e  # parar si algo falla

# Cargar .env.local si existe
if [ -f .env.local ]; then
    set -a && source .env.local && set +a
fi

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export PYTHONUTF8=1

# Localizar el intérprete del venv (activado, dentro del repo, o un nivel
# arriba; Windows usa Scripts/python.exe, Unix usa bin/python).
if [ -n "$VIRTUAL_ENV" ]; then
    PY=python
elif [ -f .venv/Scripts/python.exe ]; then
    PY=.venv/Scripts/python.exe
elif [ -f .venv/bin/python ]; then
    PY=.venv/bin/python
elif [ -f ../.venv/Scripts/python.exe ]; then
    PY=../.venv/Scripts/python.exe
elif [ -f ../.venv/bin/python ]; then
    PY=../.venv/bin/python
else
    PY=python
fi

echo "═══════════════════════════════════════════════════════════════════"
echo "  PASO 1/3 — Tests (rigor del código)"
echo "═══════════════════════════════════════════════════════════════════"
$PY -m pytest tests/ -v

echo
echo "═══════════════════════════════════════════════════════════════════"
echo "  PASO 2/3 — Comparación de modelos (6 familias vs. baseline)"
echo "═══════════════════════════════════════════════════════════════════"
$PY -m src.model_comparison

echo
echo "═══════════════════════════════════════════════════════════════════"
echo "  PASO 3/3 — Pipeline demo end-to-end con LLM"
echo "═══════════════════════════════════════════════════════════════════"
$PY -m src.pipeline_demo --profile banca --seed 47 --pause 1.5
