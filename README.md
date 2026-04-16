# MoveUp Next-Action Predictor

A machine-learning system that predicts the **next best sales action** for B2B leads in a corporate mobility CRM pipeline.

Built as a Master's thesis (TFM) project. The fictional company **MoveUp** offers corporate mobility services (Uber for Business style) across Spain and Latin America.

---

## Estado actual del proyecto — 16 abril 2026

El proyecto ha evolucionado de un prototipo XGBoost funcional a un **pipeline ML completo y académicamente defendible**. A continuación el estado real tras la última sesión de trabajo.

### Dataset

| Metrica | Valor |
|---|---|
| Filas totales | 996 interacciones |
| Leads únicos | 498 |
| Llamadas por lead | 1-3 |
| Clases target | 7 |
| Clase más frecuente | "Enviar documentación" (271 casos, 27%) |
| Clase menos frecuente | "Escalar a manager del lead" (**2 casos**, 0.2%) |
| Desbalanceo ratio | **135.5×** — hallazgo documentado en validate_dataset |

### Resultados de comparación de modelos (test set 20%, seed=42)

| Modelo | F1-weighted | Accuracy | F1-macro | Top-3 Acc |
|---|---|---|---|---|
| **XGBoost** (referencia) | **0.6619** | 0.675 | 0.613 | 0.915 |
| LightGBM | 0.6286 | 0.645 | 0.568 | 0.910 |
| Logistic Regression | 0.6225 | 0.630 | 0.581 | 0.920 |
| Random Forest | 0.6055 | 0.640 | 0.509 | 0.920 |
| Dummy (baseline trivial) | 0.1148 | 0.270 | 0.071 | 0.480 |

XGBoost supera a todos los baselines. El gap frente a Dummy (~0.55 en F1-weighted) justifica el enfoque de ML. Top-3 accuracy de 0.915 indica que el modelo tiene buen conocimiento del espacio de acciones.

### Tuning de hiperparámetros (RandomizedSearchCV, n_iter=20)

| Modelo | Mejor CV F1-w | Mejores parámetros |
|---|---|---|
| XGBoost tuned | 0.6304 | `lr=0.15, depth=7, subsample=0.7, colsample=0.7` |
| RandomForest tuned | 0.6258 | `depth=20, max_features=None, min_samples_leaf=2` |

El tuning de XGBoost apenas mejora el modelo base (0.6589 vs 0.6619 en test), lo que sugiere que el dataset es el cuello de botella, no los hiperparámetros.

### Hallazgos de validación del dataset (validate_dataset)

- **Desbalanceo severo**: "Escalar a manager del lead" con 2 ejemplos — el modelo no puede aprenderla
- **Correlación alta** employees↔revenue (r=0.93) — artefacto esperado del proceso de generación con GPT
- **1 duplicado exacto** detectado
- **Patrones uniformes por sector**: baja varianza en la distribución de next_step por sector (entropy std=0.073) — posible limitación del prompt de generación GPT

### Tests

```
21/21 tests pasan (0 skips, 0 failures)
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

- [ ] Generar más datos (subir de 498 a 2.000+ leads) especialmente para clases minoritarias
- [ ] Estrategia de balanceo de clases (SMOTE, class_weight, oversampling de "Escalar manager")
- [ ] Calibración de probabilidades (PlattScaling o IsotonicRegression sobre XGBoost)
- [ ] Notebook de análisis exploratorio completo
- [ ] Redacción de la memoria TFM en `docs/memoria_tfm.md`

---

## Módulos del proyecto

| Script | Descripción |
|---|---|
| `src/generate_dataset.py` | Genera leads y transcripts con GPT-4o |
| `src/feature_engineering.py` | Embeddings + features tabulares |
| `src/train_model.py` | Entrena XGBoost, 5-fold CV, guarda artefactos |
| `src/predict.py` | Predicción standalone de siguiente acción |
| `src/evaluate.py` | Plots + métricas completas (`evaluate_model`) |
| `src/model_comparison.py` | Compara Dummy / LogReg / RF / XGBoost / LightGBM |
| `src/tune_model.py` | RandomizedSearchCV para XGBoost y RF |
| `src/validate_dataset.py` | 5 análisis de calidad del dataset sintético |
| `src/simulate.py` | Simulación secuencial de trayectorias de leads |
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

### 2. Set your OpenAI API key (only needed for data generation)

```bash
export OPENAI_API_KEY="sk-..."
```

### 3. Full pipeline — recommended order

```bash
# Validate dataset quality first
python -m src.validate_dataset

# Train the reference model (XGBoost)
python -m src.train_model

# Compare all models
python -m src.model_comparison

# Hyperparameter tuning (~5-10 min)
python -m src.tune_model

# Simulate lead trajectories
python -m src.simulate                     # 50 leads (default)
python -m src.simulate --n-leads=100
python -m src.simulate --compare           # strategy comparison

# Run tests
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 pytest tests/ -m "not integration" -v

# Predict on a new interaction
python -m src.predict
```

### 4. View recorded experiments

```bash
python -m src.experiment_tracker
```

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

### Target variable (7 classes)

1. `Recontactar en X días`
2. `Enviar documentación`
3. `Agendar demo/reunión con especialista`
4. `Escalar a manager del lead`
5. `Cerrar lead - no interesado`
6. `Cerrar lead - nurturing`
7. `Esperar confirmación cliente`

### Feature engineering

```
Feature matrix X = [numeric (5) | one-hot categorical (variable) | embeddings (768)]
Total columns: ~1,348
```

- **Numeric (5):** employees, revenue, call_number, days_since_entry, days_since_last_call — scaled with `StandardScaler`
- **Categorical (7):** sector, country, city, lead_source, contact_role, prev_outcome, prev_next_step — encoded with `OneHotEncoder`
- **Embeddings (768):** `paraphrase-multilingual-MiniLM-L12-v2` applied to:
  - `current_transcript` -> 384 dims
  - `initial_interest_notes + prev_outcome` -> 384 dims

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
