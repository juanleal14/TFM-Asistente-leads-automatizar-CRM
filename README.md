# MoveUp Next-Action Predictor

A machine-learning system that predicts the **next best sales action** for B2B leads in a corporate mobility CRM pipeline.

Built as a Master's thesis (TFM) project. The fictional company **MoveUp** offers corporate mobility services (Uber for Business style) across Spain and Latin America.

---

## Estado actual del proyecto — 9 mayo 2026

El proyecto ha evolucionado de un prototipo XGBoost funcional a un **pipeline ML completo y académicamente defendible**. A continuación el estado real tras la última sesión de trabajo.

### Dataset

| Metrica | Valor |
|---|---|
| Filas totales | **1.072 interacciones** (996 + 73 augmentadas) |
| Leads únicos | 528 |
| Llamadas por lead | 1-3 |
| Clases target | **6** (antes 7 — fusión de clases solapadas) |
| Clase más frecuente | "Enviar documentación" (295 casos, 28%) |
| Clase menos frecuente | "Escalar a manager del lead" (**32 casos**, 3% — antes 2 casos) |
| Desbalanceo ratio | 9.1× (antes 135.5×) — corregido vía augmentación |

### Iteraciones sobre el dataset

**Iteración 1 — Augmentación dirigida** (`src/augment_minority.py`): 30 leads adicionales con GPT-4o sesgados hacia "Escalar a manager del lead" (2 → 32 ejemplos). Eliminó warnings de StratifiedKFold con n_splits=5.

**Iteración 2 — Fusión de clases solapadas**: análisis de matriz de confusión reveló colapso de "Recontactar en X días" (recall 0%, los 13 casos se distribuyeron entre 4 clases semánticamente equivalentes). Se fusionaron "Recontactar en X días" + "Cerrar lead - nurturing" → **"Aplazar lead"** (149 casos). "Esperar confirmación cliente" se mantuvo separada (rendía bien al 78% F1).

### Resultados de comparación de modelos (test set 20%, seed=42)

Features: 849 (5 numéricas + ~76 OHE + 768 embeddings). `prev_outcome` eliminado del OHE tras detectar que sus 499 valores únicos, 498 de los cuales aparecen solo 1 vez, no aportaban señal estadística. Su semántica queda capturada por los embeddings de contexto.

| Modelo | F1-weighted | Accuracy | F1-macro | Top-3 Acc | CV ± std |
|---|---|---|---|---|---|
| Random Forest | **0.6461** | 0.661 | 0.556 | 0.930 | ±0.027 |
| **XGBoost** (referencia) | 0.6392 | 0.647 | **0.616** | **0.944** | **±0.012** |
| LightGBM | 0.6214 | 0.628 | 0.589 | 0.940 | ±0.031 |
| Logistic Regression | 0.5957 | 0.605 | 0.507 | 0.935 | ±0.027 |
| Dummy (baseline trivial) | 0.1182 | 0.274 | 0.072 | 0.437 | – |

**XGBoost se mantiene como modelo de referencia** pese a que RF gana en F1-w por 1%: XGBoost tiene mejor F1-macro (+11% relativo, importante con desbalanceo), mejor Top-3 accuracy, y CV mucho más estable (±0.012 vs ±0.027). Top-3 accuracy ≥0.93 en todos los modelos no triviales.

### Evolución de métricas a lo largo de las iteraciones (XGBoost)

| Estado | F1-w test | F1-macro | "Escalar manager" recall | "Recontactar" recall | CV ± std |
|---|---|---|---|---|---|
| Inicial (996 filas, 7 clases) | 0.6389 | 0.59 | n/a (0 en test) | 0.62 | ±0.047 |
| Tras augmentar "Escalar" | 0.5817 | 0.48 | 0.17 | **0.00** ⚠️ | ±0.041 |
| **Tras fusionar Aplazar** | **0.6392** | **0.62** | **0.50** | n/a (fusionada) | **±0.012** |

### Tuning de hiperparámetros (RandomizedSearchCV, n_iter=20)

| Modelo | Mejor CV F1-w | Mejores parámetros |
|---|---|---|
| XGBoost tuned | 0.6304 | `lr=0.15, depth=7, subsample=0.7, colsample=0.7` |
| RandomForest tuned | 0.6258 | `depth=20, max_features=None, min_samples_leaf=2` |

El tuning de XGBoost apenas mejora el modelo base (0.6589 vs 0.6619 en test), lo que sugiere que el dataset es el cuello de botella, no los hiperparámetros.

### Hallazgos de validación del dataset (validate_dataset)

- **Desbalanceo severo corregido**: "Escalar a manager del lead" pasó de 2 a 32 ejemplos vía `augment_minority.py`. El ratio de desbalanceo bajó de 135.5× a 9.1×.
- **Correlación alta** employees↔revenue (r=0.93) — artefacto esperado del proceso de generación con GPT
- **1 duplicado exacto** detectado
- **Patrones uniformes por sector**: baja varianza en la distribución de next_step por sector (entropy std=0.073) — posible limitación del prompt de generación GPT
- **`prev_outcome` eliminado del OHE**: análisis de frecuencias reveló 499 valores únicos con 498 apareciendo exactamente 1 vez — sin señal estadística posible. Eliminado de categorical_features; la semántica persiste en los embeddings de contexto.
- **Colapso de clase "Recontactar en X días" detectado vía matriz de confusión**: recall 0% — los 13 casos en test se distribuyeron entre 4 clases semánticamente equivalentes (Cerrar perdido, Enviar documentación, Agendar demo, Cerrar nurturing). Diagnóstico: solapamiento del label space, no fallo del modelo. Resuelto fusionando "Recontactar en X días" + "Cerrar lead - nurturing" → "Aplazar lead". F1-macro mejoró +30% relativo, CV stabilidad ×3.

### Tests

```
27/27 tests pasan (21 unitarios + 6 integración, 0 skips, 0 failures)
Tiempo de ejecución: ~55 segundos
```

### Simulación de trayectorias

| Estrategia | Conversión | Perdido | Nurturing | Pasos medios |
|---|---|---|---|---|
| default | 0% | 20% | 80% | 1.88 |
| aggressive | 8% | 22% | 70% | 1.68 |
| conservative | 0% | 22% | 78% | 1.86 |

La baja tasa de conversión refleja el sesgo del modelo hacia "Cerrar lead - nurturing" y "Enviar documentación" (clases mayoritarias). Documentar como limitación de la simulación basada en templates.

---

## Qué falta / trabajo futuro

- [x] ~~Augmentación dirigida de "Escalar a manager del lead"~~ — completado: 2 → 32 ejemplos
- [ ] Generar más datos para resto de clases minoritarias ("Recontactar en X días" tiene 67 ejemplos)
- [ ] Estrategia de balanceo de clases (SMOTE, class_weight)
- [ ] Calibración de probabilidades (PlattScaling o IsotonicRegression sobre LightGBM)
- [ ] Notebook de análisis exploratorio completo
- [ ] Redacción de la memoria TFM en `docs/memoria_tfm.md`

---

## Módulos del proyecto

| Script | Descripción |
|---|---|
| `src/generate_dataset.py` | Genera leads y transcripts con GPT-4o |
| `src/augment_minority.py` | **Augmentación dirigida** de clases minoritarias |
| `src/feature_engineering.py` | Embeddings + features tabulares |
| `src/train_model.py` | Entrena XGBoost, 5-fold CV, guarda artefactos |
| `src/predict.py` | Predicción standalone de siguiente acción |
| `src/summarize.py` | **Resumidor LLM** que genera `prev_outcome` en producción |
| `src/evaluate.py` | Plots + métricas completas (`evaluate_model`) |
| `src/model_comparison.py` | Compara Dummy / LogReg / RF / XGBoost / LightGBM |
| `src/tune_model.py` | RandomizedSearchCV para XGBoost y RF |
| `src/validate_dataset.py` | 5 análisis de calidad del dataset sintético |
| `src/simulate.py` | Simulación secuencial de trayectorias de leads |
| `src/pipeline_demo.py` | **Demo visual end-to-end** con resumidor LLM en el loop |
| `src/experiment_tracker.py` | Tracking de experimentos (JSON + CSV, sin MLflow) |

---

## Project structure

```
moveup-next-action-predictor/
├── config.yaml                  <- Single source of truth for all parameters
├── requirements.txt
├── pytest.ini
├── .gitignore
├── src/
│   ├── config.py                <- Loads config.yaml, exposes CONFIG dict
│   ├── utils.py                 <- JSON helpers + append_csv_row
│   ├── generate_dataset.py      <- GPT-4o dataset generator
│   ├── feature_engineering.py   <- Embeddings + tabular features
│   ├── train_model.py           <- XGBoost training pipeline
│   ├── predict.py               <- Standalone prediction function
│   ├── evaluate.py              <- Evaluation plots + evaluate_model()
│   ├── model_comparison.py      <- Multi-model benchmark
│   ├── tune_model.py            <- Hyperparameter search
│   ├── validate_dataset.py      <- Dataset quality analysis
│   ├── simulate.py              <- Sequential lead simulation
│   └── experiment_tracker.py   <- Experiment logging
├── data/
│   ├── raw/                     <- Generated CSVs (git-ignored)
│   └── processed/               <- Cached embeddings (.npz, git-ignored)
├── models/                      <- Saved .joblib artefacts (git-ignored)
├── plots/                       <- PNG evaluation plots (git-ignored)
├── experiments/                 <- Tracked runs, hyperparams, comparisons
│   ├── runs/                    <- One JSON per experiment run
│   ├── hyperparams/             <- Best params + CV results per model
│   ├── validation/              <- Dataset quality report + plots
│   ├── simulations/             <- Simulation results
│   └── summary.csv              <- Cumulative experiment index
├── tests/
│   ├── conftest.py
│   ├── fixtures/sample_data.csv <- 50-row real sample for tests
│   ├── test_feature_engineering.py
│   ├── test_predict.py
│   └── test_integration.py
├── notebooks/
└── docs/
    └── memoria_tfm.md           <- TFM memoir skeleton
```

---

## Quickstart

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set your OpenAI API key

Necesaria para `generate_dataset.py` y para el resumidor LLM en `pipeline_demo.py`. Guárdala en un fichero `.env.local` (git-ignored) y cárgala con:

```bash
set -a && source .env.local && set +a
```

O exportala directamente:

```bash
export OPENAI_API_KEY="sk-..."
```

---

## Cómo verificar que todo funciona — secuencia completa

Todos los comandos asumen entorno virtual activo y la API key cargada. El prefijo `TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1` evita un segfault en macOS al combinar XGBoost (OpenMP) y PyTorch (sentence-transformers) en el mismo proceso.

### 1️⃣ Tests unitarios — ~45 s

```bash
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 .venv/bin/python -m pytest tests/ -m "not integration" -v
```

**Verifica:** que `feature_engineering.py` y `predict.py` funcionan correctamente — schema de output, probabilidades suman 1, etiquetas válidas, encoders fiteados consistentemente. **Esperado: 21/21 passed.**

### 2️⃣ Tests de integración (incluye uso del modelo real)

```bash
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 .venv/bin/python -m pytest tests/ -v
```

**Verifica:** pipeline end-to-end con `tests/fixtures/sample_data.csv` (50 filas reales). **Esperado: 27/27 passed.**

### 3️⃣ Validación del dataset (auditoría de calidad)

```bash
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 .venv/bin/python -m src.validate_dataset
```

**Verifica:** 5 auditorías — distribución de clases, correlaciones, duplicados, coherencia temporal, patrones artificiales. Reporte JSON en `experiments/validation/`.

### 4️⃣ Entrenamiento del modelo

```bash
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 .venv/bin/python -m src.train_model
```

**Verifica:** pipeline completo de training — carga datos, embeddings, matriz X (849 features), 5-fold CV, evaluación en test, guarda artefactos en `models/moveup_nextstep_model.joblib`. **Esperado: F1-w ~0.64 en test, CV ~0.62.**

### 5️⃣ Comparación de modelos

```bash
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 .venv/bin/python -m src.model_comparison
```

**Verifica:** 5 modelos (Dummy, LogReg, RF, XGBoost, LightGBM) sobre el mismo split. **Esperado: XGBoost gana con 0.6389 F1-w; Dummy 0.1148.** Resultados en `experiments/model_comparison.csv`.

### 6️⃣ Predicción standalone (smoke test)

```bash
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 .venv/bin/python -m src.predict
```

**Verifica:** que `predict_next_step()` carga el modelo guardado y predice sobre un input de ejemplo.

### 7️⃣ Simulación batch de trayectorias

```bash
# 50 leads sintéticos en batch (templates, sin GPT)
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 .venv/bin/python -m src.simulate

# Comparación de 3 estrategias (default / aggressive / conservative)
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 .venv/bin/python -m src.simulate --compare
```

**Verifica:** simulación secuencial completa. Resultados en `experiments/simulations/`.

### 8️⃣ Pipeline demo visual con LLM (pieza estrella para la defensa)

```bash
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 .venv/bin/python -m src.pipeline_demo --seed 11
```

**Verifica:** loop completo end-to-end con resumidor LLM (`gpt-4o-mini`) cerrando el bucle de producción. En cada llamada: lead → transcript → predicción XGBoost → barras ASCII de probabilidades → acción ejecutada → resumen LLM que alimenta el `prev_outcome` de la siguiente llamada.

**Variantes útiles:**

```bash
# Sin LLM (más rápido, sin coste OpenAI)
.venv/bin/python -m src.pipeline_demo --no-llm

# Con pausa de 2s entre pasos para presentación en vivo
.venv/bin/python -m src.pipeline_demo --pause 2 --seed 11

# Lead aleatorio (no el hardcodeado)
.venv/bin/python -m src.pipeline_demo --random --seed 7
```

### 9️⃣ Tuning de hiperparámetros (opcional, ~5-10 min)

```bash
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 .venv/bin/python -m src.tune_model
```

**Verifica:** RandomizedSearchCV (n_iter=20) sobre XGBoost y RandomForest. **Esperado: mejora marginal vs baseline → confirma que el cuello de botella es el dataset, no los hiperparámetros.**

### 🔟 Histórico de experimentos

```bash
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 .venv/bin/python -m src.experiment_tracker
```

**Verifica:** lista todos los runs guardados con sus métricas, identifica el mejor.

---

## Secuencia mínima para una defensa de 10 minutos

```bash
# 1. Setup
set -a && source .env.local && set +a

# 2. Rigor de tests (1 min)
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 .venv/bin/python -m pytest tests/ -v

# 3. Comparación de modelos — XGBoost vs baselines (3 min)
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 .venv/bin/python -m src.model_comparison

# 4. Demo visual end-to-end con LLM (5 min)
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 .venv/bin/python -m src.pipeline_demo --pause 1.5 --seed 11
```

---

## Health checks rápidos antes de la defensa

```bash
# Modelo entrenado y guardado
ls -lh models/moveup_nextstep_model.joblib

# Cache de embeddings (evita 30 s de recomputación)
ls -lh data/processed/embeddings_cache.npz

# Experimentos previos registrados
head -5 experiments/summary.csv

# Plots generados
ls plots/
```

Si algo falta, regenéralo con `python -m src.train_model`.

---

## Reset si algo falla

```bash
# Borra modelo y plots, fuerza reentrenamiento limpio
rm -f models/*.joblib plots/*.png

# Reentrenar desde cero (mantiene cache de embeddings)
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 .venv/bin/python -m src.train_model
```

> **Nota:** el cache de embeddings (`.npz`) solo se invalida si cambias `embedding_model` en `config.yaml` o el texto de las columnas `current_transcript` / `prev_outcome` / `initial_interest_notes`. Cambiar `categorical_features` **no** afecta al cache, solo a la matriz X — borra solo el modelo en ese caso.

---

## Pipeline in detail

### Dataset schema

One row per call interaction. A lead with 3 calls generates 3 rows.

| Column | Type | Description |
|---|---|---|
| `interaction_id` | str | Unique ID for this call row |
| `lead_id` | str | Groups all calls for one lead |
| `contact_name` | str | Decision-maker name |
| `contact_role` | str | Job title (15 possible values) |
| `company_sector` | str | Industry (20 sectors) |
| `company_num_employees` | int | Company headcount |
| `company_annual_revenue_eur` | float | Annual revenue in EUR |
| `lead_source` | str | Origin channel (9 sources) |
| `call_number` | int | 1, 2, or 3 |
| `days_since_entry` | int | Days since lead entered CRM |
| `days_since_last_call` | int | 0 for first call |
| `prev_outcome` | str | Summary of previous call |
| `prev_next_step` | str | Action decided after previous call |
| `current_transcript` | str | Spanish dialogue (8-20 turns) |
| `current_outcome` | str | Brief summary of this call |
| `next_step` | str | **TARGET**: one of 7 categories |
| `final_status` | str | Converted / Lost / Nurturing / In Progress |

### Target variable (6 classes)

1. `Aplazar lead` (fusión: "Recontactar en X días" + "Cerrar lead - nurturing")
2. `Enviar documentación`
3. `Agendar demo/reunión con especialista`
4. `Escalar a manager del lead`
5. `Cerrar lead - no interesado`
6. `Esperar confirmación cliente`

### Feature engineering

```
Feature matrix X = [numeric (5) | one-hot categorical (variable) | embeddings (768)]
Total columns: 849
```

- **Numeric (5):** employees, revenue, call_number, days_since_entry, days_since_last_call — scaled with `StandardScaler`
- **Categorical (6):** sector, country, city, lead_source, contact_role, prev_next_step — encoded with `OneHotEncoder` (`prev_outcome` eliminado: 499 valores únicos, 498 con frecuencia 1)
- **Embeddings (768):** `paraphrase-multilingual-MiniLM-L12-v2` applied to:
  - `current_transcript` → 384 dims
  - `initial_interest_notes + prev_outcome` → 384 dims (semántica de prev_outcome preservada aquí)

### Reference model (XGBoost)

```yaml
n_estimators:     300
max_depth:        6
learning_rate:    0.1
subsample:        0.8
colsample_bytree: 0.8
min_child_weight: 3
```

Evaluation: stratified 80/20 split + 5-fold cross-validation (F1 weighted).

---

## Configuration reference (`config.yaml`)

| Key | Default | Description |
|---|---|---|
| `num_leads` | `500` | Leads to generate |
| `openai_model` | `gpt-4o` | OpenAI model for generation |
| `embedding_model` | `paraphrase-multilingual-MiniLM-L12-v2` | Sentence transformer |
| `paths.raw_data` | `data/raw/moveup_crm_dataset.csv` | Input CSV |
| `paths.model` | `models/moveup_nextstep_model.joblib` | Saved model |
| `experiments_dir` | `experiments` | Experiment tracking root |
| `comparison.*` | see file | Model comparison settings |
| `tuning.*` | see file | RandomizedSearchCV settings |
| `model_params.*` | see file | XGBoost hyperparameters |

---

## Running tests

```bash
# Unit tests only (no model required for most)
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 pytest tests/ -m "not integration" -v

# All tests including integration (requires trained model)
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 pytest tests/ -v
```

The env vars prevent a macOS segfault caused by XGBoost (OpenMP) + PyTorch (sentence-transformers) running in the same process.

---

## Tech stack

- **Python 3.12**
- **OpenAI API** — GPT-4o for synthetic transcript generation
- **sentence-transformers** — multilingual semantic embeddings
- **XGBoost / LightGBM** — gradient boosting classifiers
- **scikit-learn** — preprocessing, evaluation, RandomizedSearchCV
- **scipy** — statistical tests (chi-square, Kruskal-Wallis)
- **pandas / numpy** — data manipulation
- **matplotlib / seaborn** — visualisation
- **joblib** — model serialisation
- **PyYAML** — configuration management
