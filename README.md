# MoveUp Next-Action Predictor

A machine-learning system that predicts the **next best sales action** for B2B leads in a corporate mobility CRM pipeline.

Built as a Master's thesis (TFM) project. The fictional company **MoveUp** offers corporate mobility services (Uber for Business style) across Spain and Latin America.

---

## Estado actual del proyecto — 2 septiembre 2026

El proyecto ha evolucionado de un prototipo XGBoost funcional a un **pipeline ML completo y académicamente defendible**. A continuación el estado real tras la sesión de correcciones de septiembre 2026 (ver [Correcciones de esta sesión](#correcciones-de-esta-sesión) para el detalle de bugs encontrados y arreglados).

### Dataset

| Métrica | Valor |
|---|---|
| Filas totales | **1.071 interacciones** (996 base + 75 de augmentación dirigida) |
| Leads únicos | 528 (498 base + 30 de augmentación) |
| Clases target | **6** (dataset crudo generado con 7; 2 fusionadas — ver más abajo) |
| Clase más frecuente | "Enviar documentación" (291 casos, 27%) |
| Clase menos frecuente | "Escalar a manager del lead" (**32 casos**, 3% — antes 2 casos) |
| Desbalanceo ratio | **9.09×** (antes 135.5× con solo 2 ejemplos de la clase minoritaria) |
| Duplicados exactos | 1 |

### Iteraciones sobre el dataset

**Iteración 1 — Fusión de clases solapadas.** El dataset se generó originalmente con 7 categorías de `next_step`. "Recontactar en X días" y "Cerrar lead - nurturing" resultaron semánticamente solapadas (mismo significado de negocio: "el lead necesita más tiempo, sin acción activa"), así que se fusionaron en una sola clase **"Aplazar lead"** (149 casos). Esta normalización vive centralizada en `feature_engineering.load_and_clean()` — se aplica automáticamente a cualquier script que cargue el dataset (entrenamiento, comparación, tuning, validación), no solo al que reentrena el modelo de producción.

**Iteración 2 — Augmentación dirigida** (`src/augment_minority.py`): 30 leads adicionales generados con GPT-4o, sesgados hacia "Escalar a manager del lead" (2 → 32 ejemplos, +75 filas de interacción en total ya que cada lead genera 2-3 llamadas). Bajó el ratio de desbalanceo de 135.5× a 9.09× y eliminó los warnings de `StratifiedKFold` con `n_splits=5`.

### Resultados de comparación de modelos (test set 20%, seed=42, dataset ampliado 1.071 filas)

Features: **848** (5 numéricas + ~75 OHE + 768 embeddings — una columna menos que en iteraciones previas porque `prev_next_step` ya no incluye categorías fantasma de las clases fusionadas).

Modelos con **hiperparámetros base** (`config.yaml`), sin tuning, LightGBM incluido:

| Modelo | F1-weighted | Accuracy | F1-macro | Top-3 Acc | CV F1-w ± std | Brier |
|---|---|---|---|---|---|---|
| Random Forest | **0.6478** | 0.6698 | 0.5550 | 0.930 | 0.6283 ± 0.026 | 0.0894 |
| **XGBoost** (referencia) | 0.6378 | 0.6512 | 0.5916 | **0.963** | **0.6396 ± 0.021** | 0.0860 |
| LightGBM | 0.6374 | 0.6465 | **0.6218** | 0.958 | 0.6337 ± 0.019 | 0.0954 |
| Logistic Regression | 0.6339 | 0.6419 | 0.5815 | 0.958 | 0.6027 ± 0.013 | **0.0832** |
| Dummy (baseline trivial) | 0.1146 | 0.2698 | 0.0708 | 0.433 | 0.1165 | 0.2434 |

**Hallazgo destacable: LightGBM sin tunear tiene el mejor F1-macro de los 5 modelos** (0.6218 — un +5.1% relativo sobre XGBoost, +12% sobre RF), pese a no ganar en F1-weighted. Como F1-macro pondera todas las clases por igual (no por frecuencia), esto sugiere que LightGBM reparte mejor su capacidad predictiva entre clases minoritarias como "Escalar a manager del lead", justo el tipo de robustez frente al desbalanceo que más importa en este problema. LightGBM **nunca se ha tuneado** (`tune_model.py` solo tiene grid de búsqueda para XGBoost y RandomForest en `config.yaml` — añadir uno para LightGBM es una extensión natural y barata, ver "Qué falta" más abajo).

**XGBoost se mantiene como modelo de referencia** frente a RF pese a perder por ~1.5 puntos en F1-w: XGBoost tiene mejor F1-macro (+6.6% relativo sobre RF), mejor Top-3 accuracy, y CV más estable (±0.021 vs ±0.026). Pero el cuadro completo (ver tuning más abajo) matiza bastante esta narrativa.

### Tuning de hiperparámetros (RandomizedSearchCV, n_iter=20, cv=5)

⚠️ **Tiempo real de ejecución: ~1h 51min** (XGBoost ≈ 5265 s ≈ 88 min, RandomForest ≈ 1384 s ≈ 23 min) — mucho más que la estimación previa de "5-10 min"; con `n_iter=20 × cv=5` sobre 1.071 filas × 848 features cada candidato tarda su tiempo. Ejecutar con margen, no en el sandbox de una sesión de asistente. `tune_model.py` ahora persiste el estimador reajustado (`experiments/hyperparams/{modelo}_tuned_model.joblib`) y las métricas completas (`{modelo}_tuned_test_metrics.json`), así que esto no hace falta recalcularlo a mano en el futuro.

| Modelo | Mejor CV F1-w | Test F1-w | Test Acc | Test F1-macro | Test Top-3 | Test Brier | Mejores parámetros |
|---|---|---|---|---|---|---|---|
| XGBoost tuned | 0.6389 | 0.6500 | 0.6698 | 0.5571 | 0.9535 | **0.0801** | `subsample=0.7, n_estimators=200, min_child_weight=7, max_depth=3, learning_rate=0.05, colsample_bytree=0.9` |
| **RandomForest tuned** | **0.6452** | **0.6554** | **0.6744** | **0.5627** | 0.9535 | 0.0908 | `n_estimators=200, min_samples_split=5, min_samples_leaf=4, max_features=sqrt, max_depth=15` |

**Hallazgo importante — el tuning de XGBoost mejoró F1-weighted pero empeoró F1-macro:** F1-w subió de 0.6378 (base) a 0.6500 (tuned, +1.9%), pero F1-macro **bajó** de 0.5916 a 0.5571 (**-5.8% relativo**) y Top-3 accuracy bajó de 0.963 a 0.9535. Esto es coherente con que `RandomizedSearchCV` optimiza explícitamente `scoring="f1_weighted"` (ver `config.yaml → tuning.scoring`): al buscar hiperparámetros que maximicen una métrica ponderada por frecuencia de clase, el buscador implícitamente prioriza acertar en las clases mayoritarias a costa de las minoritarias. Es un ejemplo real y citable de un problema conocido en la literatura de tuning bajo desbalanceo de clases (optimizar la métrica "equivocada" para el objetivo de negocio real). Si el objetivo del sistema es no ignorar sistemáticamente clases minoritarias como "Escalar a manager del lead", **re-tunear con `scoring="f1_macro"`** sería una mejora concreta y fácil de justificar en la memoria (cambiar una línea en `config.yaml`).

En Random Forest el efecto es el contrario y más benigno: tuning mejora F1-w (+1.2%), F1-macro (+1.4%) y Top-3 (+2.5%) a la vez, aunque empeora ligeramente el Brier score (peor calibración: 0.0894 → 0.0908).

**Ranking combinado de los 7 modelos evaluados** (base + tuneados), por criterio:

| Criterio | Mejor modelo | Valor |
|---|---|---|
| F1-weighted | RandomForest tuned | 0.6554 |
| F1-macro | **LightGBM (sin tunear)** | 0.6218 |
| Top-3 accuracy | XGBoost base | 0.9628 |
| Calibración (Brier, menor=mejor) | XGBoost tuned | 0.0801 |

**No hay un modelo que gane en todos los criterios** — es el resultado honesto a discutir en la memoria en vez de forzar una única conclusión. Recomendación según prioridad de negocio:
- Si prima **accuracy global ponderada** (la mayoría de leads se comportan "normal"): RandomForest tuned.
- Si prima **no ignorar clases minoritarias** (ej. escalado a manager, casos de alto valor): LightGBM, idealmente tuneado.
- Si prima **que el modelo no descarte nunca la acción correcta** (usarlo para sugerir top-3 opciones a un agente humano, no decidir solo): XGBoost base.
- Si prima **confianza calibrada** (usar la probabilidad reportada para decisiones automáticas, no solo el ranking): XGBoost tuned.

Este trabajo mantiene **XGBoost como modelo de producción** (`models/moveup_nextstep_model.joblib`, hiperparámetros base) por ser el que mejor equilibra las cuatro dimensiones sin ganar en ninguna de forma extrema ni perder en ninguna de forma grave — pero es una decisión de diseño explícita a defender, no la única lectura válida de los datos.

### Hallazgos de validación del dataset (`validate_dataset.py`, dataset ampliado)

- **Desbalanceo corregido**: "Escalar a manager del lead" pasó de 2 a 32 ejemplos vía `augment_minority.py`. Ratio de desbalanceo: 135.5× → **9.09×**.
- **Correlación alta** employees↔revenue (r=0.92) — artefacto esperado del proceso de generación con GPT.
- **1 duplicado exacto** detectado (mismo company+contact+call_number).
- **Patrones uniformes por sector**: baja varianza en la distribución de `next_step` por sector (entropy std=0.041) — posible limitación del prompt de generación GPT, documentado como tal.
- Ver `experiments/validation/validation_report.json` y los plots en `experiments/validation/` para el detalle completo (2/5 checks "pasan" formalmente, pero los 3 warnings son limitaciones documentadas de datos sintéticos, no errores).

### Métricas del modelo de producción actual (`retrain_model.py`, XGBoost, hiperparámetros base)

Classification report en test (215 filas, 20%):

| Clase | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Esperar confirmación cliente | 0.77 | 0.93 | 0.84 | 29 |
| Cerrar lead - no interesado | 0.72 | 0.78 | 0.75 | 40 |
| Agendar demo/reunión con especialista | 0.65 | 0.62 | 0.63 | 52 |
| Enviar documentación | 0.55 | 0.67 | 0.60 | 58 |
| Aplazar lead | 0.62 | 0.33 | 0.43 | 30 |
| Escalar a manager del lead | 1.00 | 0.17 | 0.29 | 6 |
| **accuracy** | | | **0.65** | 215 |
| **weighted avg** | 0.66 | 0.65 | 0.64 | 215 |

"Escalar a manager del lead" pasó de no tener señal alguna (2 ejemplos totales) a tener precisión perfecta pero recall bajo (1 de cada 6 casos de test detectado) — coherente con seguir siendo la clase con menos ejemplos absolutos incluso tras la augmentación. "Aplazar lead" es ahora la clase con peor recall (0.33): con más clases compitiendo por señal, el modelo prefiere clasificarla como otra cosa. Ambos son puntos de discusión legítimos para la sección de limitaciones de la memoria.

### Tests

```
27/27 tests pasan (0 skips, 0 failures)
```

### Demo end-to-end verificada (perfil "banca", seed=47)

Trayectoria de 5 llamadas con el modelo reentrenado sobre el dataset ampliado — predicción del modelo = acción ejecutada en las 5 llamadas (sin guion forzado):

| Llamada | Acción predicha | Confianza |
|---|---|---|
| 1 | Enviar documentación | 77.3% |
| 2 | Agendar demo/reunión con especialista | 63.6% |
| 3 | Esperar confirmación cliente | 46.8% |
| 4 | Aplazar lead | 69.8% |
| 5 | Agendar demo/reunión con especialista | 79.3% |
| | **✓ LEAD CONVERTIDO** (36 días) | |

```bash
python -m src.pipeline_demo --profile banca --seed 47 --no-llm --save
# añade --llm-transcripts para diálogos generados por GPT en vez de plantillas
```

---

## Correcciones de esta sesión

Al reentrenar el modelo con las 6 acciones válidas se encontraron y arreglaron los siguientes problemas, todos en el código, no solo en la ejecución puntual:

1. **Artefacto del modelo incompatible con producción** — `retrain_model.py` guardaba el `.joblib` con claves `rf_model`/`xgb_model`; `predict.py` (usado en producción y en la demo) esperaba una clave `model`. Causaba `KeyError` al predecir. Arreglado reutilizando el mismo pipeline de guardado que `train_model.py`.
2. **Normalización de clases no centralizada** — la fusión 7→6 clases solo se aplicaba en `retrain_model.py`; `train_model.py`, `model_comparison.py`, `tune_model.py` y `validate_dataset.py` cargaban el CSV crudo sin normalizar y habrían entrenado/evaluado sobre la taxonomía antigua. Centralizado en `feature_engineering.load_and_clean()`.
3. **Reproducibilidad de la demo rota** — `generate_synthetic_transcript()` elegía la plantilla de diálogo con `random` global en vez del `rng` sembrado por `--seed`; la misma semilla podía dar resultados distintos entre ejecuciones. Arreglado pasando el `rng` explícito.
4. **Leads sintéticos con sector inconsistente** — `_generate_synthetic_lead()` sorteaba dos sectores aleatorios distintos (uno para la empresa, otro para las notas iniciales), generando descripciones incoherentes.
5. **Tests con taxonomía obsoleta hardcodeada** — `VALID_NEXT_STEPS` en `test_predict.py`/`test_integration.py` no incluía "Aplazar lead"; cualquier predicción de esa clase habría hecho fallar los tests. Ahora usan `CONFIG["next_step_categories"]`.
6. **`run_all.sh` roto en Windows** — asumía `.venv/bin/python` (ruta Unix) dentro del propio repo; el venv real está en `Scripts/python.exe` y, en este entorno de desarrollo, un nivel por encima del repo. Ahora detecta el intérprete automáticamente.
7. **`UnicodeEncodeError` al redirigir la salida a archivo en Windows** — los `print()` con `→`/`█`/acentos revientan bajo `cp1252` cuando stdout no es una consola interactiva (p. ej. `run_all.sh > log.txt`). Arreglado forzando `PYTHONUTF8=1`.
8. **Import de `sentence-transformers` a nivel de módulo en `feature_engineering.py`** — cargaba PyTorch aunque hubiera caché de embeddings y nunca se fuera a usar, sin necesidad. Movido a import perezoso dentro de `generate_embeddings()` (mismo patrón que ya usaba `predict.py`).

> **Nota de entorno:** en el sandbox de esta sesión, `cross_val_score`/`RandomizedSearchCV`/incluso un `.fit()` suelto de XGBoost sobre el dataset completo llegaron a colgarse tras uso intensivo prolongado — no reproducible en una terminal normal (todo el reentreno, comparación, tuning y evaluación de esta sección se ejecutó y verificó en la terminal del usuario, no en el sandbox).

---

## Qué falta / trabajo futuro

- [x] ~~Augmentación dirigida de "Escalar a manager del lead"~~ — completado: 2 → 32 ejemplos
- [x] ~~Recalcular F1-macro/Top-3/Brier de los modelos tuneados~~ — completado, `tune_model.py` ahora persiste estimador + métricas completas
- [x] ~~Incluir LightGBM en la comparación~~ — completado: mejor F1-macro de los 5 modelos base (0.6218)
- [ ] **Añadir grid de búsqueda para LightGBM en `config.yaml → tuning.param_grids`** — es el único de los 3 candidatos serios sin tunear, y ya parte del mejor F1-macro base; podría ser el mejor modelo global tuneado.
- [ ] **Re-tunear XGBoost con `scoring="f1_macro"`** en vez de `f1_weighted` — el tuning actual empeoró su F1-macro (-5.8%) al optimizar la métrica ponderada; cambiar el scoring es una línea en `config.yaml` y un resultado citable directo para la memoria (trade-off F1-w vs F1-macro al tunear bajo desbalanceo).
- [ ] Decidir y justificar explícitamente el modelo de producción final — ver tabla de "ranking combinado por criterio" arriba; no hay un ganador único en las 4 métricas.
- [ ] Estrategia de balanceo de clases adicional (SMOTE, class_weight) — "Aplazar lead" y "Escalar a manager" siguen con recall bajo (0.33 y 0.17 respectivamente en el modelo de producción)
- [ ] Calibración de probabilidades (Platt Scaling o Isotonic Regression) — XGBoost tuned ya tiene el mejor Brier (0.0801) pero no se ha aplicado calibración explícita
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
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 python -m pytest tests/ -m "not integration" -v
```

**Verifica:** que `feature_engineering.py` y `predict.py` funcionan correctamente — schema de output, probabilidades suman 1, etiquetas válidas, encoders fiteados consistentemente. **Esperado: 21/21 passed.**

### 2️⃣ Tests de integración (incluye uso del modelo real)

```bash
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 python -m pytest tests/ -v
```

**Verifica:** pipeline end-to-end con `tests/fixtures/sample_data.csv` (50 filas reales). **Esperado: 27/27 passed.**

### 3️⃣ Validación del dataset (auditoría de calidad)

```bash
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 python -m src.validate_dataset
```

**Verifica:** 5 auditorías — distribución de clases, correlaciones, duplicados, coherencia temporal, patrones artificiales. Reporte JSON en `experiments/validation/`.

### 4️⃣ Entrenamiento del modelo

```bash
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 python -m src.train_model
```

**Verifica:** pipeline completo de training — carga datos, embeddings, matriz X (848 features), 5-fold CV, evaluación en test, guarda artefactos en `models/moveup_nextstep_model.joblib`. **Esperado: F1-w ~0.64 en test, CV ~0.64.**

### 5️⃣ Comparación de modelos

```bash
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 python -m src.model_comparison
```

**Verifica:** 5 modelos (Dummy, LogReg, RF, XGBoost, LightGBM) sobre el mismo split. **Esperado: Random Forest gana en F1-w (~0.648); XGBoost cerca (~0.638) con mejor Top-3; LightGBM con el mejor F1-macro (~0.622); Dummy ~0.115.** Resultados en `experiments/model_comparison.csv`.

### 6️⃣ Predicción standalone (smoke test)

```bash
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 python -m src.predict
```

**Verifica:** que `predict_next_step()` carga el modelo guardado y predice sobre un input de ejemplo.

### 7️⃣ Simulación batch de trayectorias

```bash
# 50 leads sintéticos en batch (templates, sin GPT)
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 python -m src.simulate

# Comparación de 3 estrategias (default / aggressive / conservative)
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 python -m src.simulate --compare
```

**Verifica:** simulación secuencial completa. Resultados en `experiments/simulations/`.

### 8️⃣ Pipeline demo visual con LLM (pieza estrella para la defensa)

```bash
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 python -m src.pipeline_demo --seed 11
```

**Verifica:** loop completo end-to-end con resumidor LLM (`gpt-4o-mini`) cerrando el bucle de producción. En cada llamada: lead → transcript → predicción XGBoost → barras ASCII de probabilidades → acción ejecutada → resumen LLM que alimenta el `prev_outcome` de la siguiente llamada.

**Variantes útiles:**

```bash
# Sin LLM (más rápido, sin coste OpenAI)
python -m src.pipeline_demo --no-llm

# Con pausa de 2s entre pasos para presentación en vivo
python -m src.pipeline_demo --pause 2 --seed 11

# Lead aleatorio (no el hardcodeado)
python -m src.pipeline_demo --random --seed 7
```

### 9️⃣ Tuning de hiperparámetros (opcional, ~2 horas — ver nota de tiempos arriba)

```bash
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 python -m src.tune_model
```

**Verifica:** RandomizedSearchCV (n_iter=20) sobre XGBoost y RandomForest. **Esperado: mejora marginal (RF tuned F1-w ~0.655, XGBoost tuned ~0.650) → confirma que el cuello de botella es el dataset, no los hiperparámetros. Con tuning, RF supera a XGBoost en F1-w.**

### 🔟 Histórico de experimentos

```bash
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 python -m src.experiment_tracker
```

**Verifica:** lista todos los runs guardados con sus métricas, identifica el mejor.

---

## Secuencia mínima para una defensa de 10 minutos

```bash
# 1. Setup
set -a && source .env.local && set +a

# 2. Rigor de tests (1 min)
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 python -m pytest tests/ -v

# 3. Comparación de modelos — XGBoost vs baselines (3 min)
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 python -m src.model_comparison

# 4. Demo visual end-to-end con LLM (5 min)
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 python -m src.pipeline_demo --pause 1.5 --seed 11
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
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 python -m src.train_model
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
| `next_step` | str | **TARGET**: raw CSV has 7 categories, normalized to 6 by `load_and_clean()` |
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
Total columns: 848  (1.071-row augmented dataset)
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

- **Python 3.13**
- **OpenAI API** — GPT-4o for synthetic transcript generation
- **sentence-transformers** — multilingual semantic embeddings
- **XGBoost / LightGBM** — gradient boosting classifiers
- **scikit-learn** — preprocessing, evaluation, RandomizedSearchCV
- **scipy** — statistical tests (chi-square, Kruskal-Wallis)
- **pandas / numpy** — data manipulation
- **matplotlib / seaborn** — visualisation
- **joblib** — model serialisation
- **PyYAML** — configuration management
