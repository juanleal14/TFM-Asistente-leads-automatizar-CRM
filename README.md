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

Modelos con **hiperparámetros base** (`config.yaml`), sin tuning — 8 modelos abarcando 6 familias distintas, no solo variantes de árboles:

| Modelo | Familia | F1-weighted | Accuracy | F1-macro | Top-3 Acc | CV F1-w ± std | Brier |
|---|---|---|---|---|---|---|---|
| Random Forest | Árboles (bagging) | **0.6478** | 0.6698 | 0.5550 | 0.930 | 0.6283 ± 0.026 | 0.0894 |
| **XGBoost** (referencia) | Árboles (boosting) | 0.6378 | 0.6512 | 0.5916 | 0.963 | **0.6396 ± 0.021** | 0.0860 |
| LightGBM | Árboles (boosting, leaf-wise) | 0.6374 | 0.6465 | **0.6218** | 0.958 | 0.6337 ± 0.019 | 0.0954 |
| Logistic Regression | Lineal | 0.6339 | 0.6419 | 0.5815 | 0.958 | 0.6027 ± 0.013 | **0.0832** |
| MLP | Neuronal (128,64) | 0.6197 | 0.6372 | 0.5273 | **0.967** | 0.6005 ± 0.012 | 0.0841 |
| k-NN (k=15) | Basado en instancias | 0.5480 | 0.5721 | 0.4676 | 0.926 | 0.5823 ± **0.004** | 0.0950 |
| Naive Bayes | Probabilístico/generativo | 0.5294 | 0.5349 | 0.4604 | 0.898 | 0.5328 ± 0.022 | 0.1521 |
| Dummy (baseline trivial) | — | 0.1146 | 0.2698 | 0.0708 | 0.433 | 0.1165 ± 0.002 | 0.2434 |

**Tres hallazgos destacables de ampliar a 6 familias de modelos:**

1. **LightGBM sin tunear tiene el mejor F1-macro de los 8 modelos** (0.6218 — +5.1% relativo sobre XGBoost, +12% sobre RF), pese a no ganar en F1-weighted. Como F1-macro pondera todas las clases por igual, LightGBM reparte mejor su capacidad predictiva hacia clases minoritarias como "Escalar a manager del lead". **Nunca se ha tuneado** (ver "Qué falta").
2. **MLP tiene el mejor Top-3 accuracy de todos los modelos evaluados, base o tuneados** (0.967) — pese a un F1-weighted mediocre (0.6197, por debajo incluso de Logistic Regression). Es decir: la red neuronal casi nunca deja la acción correcta fuera de sus 3 mejores sugerencias, aunque su predicción #1 (argmax) acierte menos que los árboles. Encaja con un uso de "sugerir 3 opciones a un agente humano" más que "decidir en automático".
3. **k-NN y Naive Bayes rinden claramente peor** que los árboles y el lineal (F1-w 0.548 y 0.529 respectivamente, F1-macro aún peor). Explicación más probable: ambos son sensibles a la escala/distribución de las features, y la matriz combina embeddings continuos (768 dims, escala razonable) con ~75 columnas one-hot binarias sin escalar — las distancias de k-NN y las probabilidades condicionales gaussianas de Naive Bayes se distorsionan con esa mezcla de escalas, mientras que los árboles son invariantes a ella. Curiosamente **k-NN es con diferencia el modelo más estable entre folds** (CV std = 0.0036, ~6× menor que XGBoost) — rinde peor pero de forma muy consistente. Es un resultado esperable y documentable, no un fallo de implementación: motiva por qué los métodos de árboles/boosting son la elección natural para esta matriz de features mixta, en vez de asumirlo sin evidencia.

**XGBoost se mantiene como modelo de referencia** frente a RF pese a perder por ~1.5 puntos en F1-w: XGBoost tiene mejor F1-macro (+6.6% relativo sobre RF), mejor Top-3 accuracy entre los árboles, y CV más estable entre los boosting/bagging (±0.021 vs ±0.026). Pero el cuadro completo (ver tuning más abajo) matiza bastante esta narrativa.

### Tuning de hiperparámetros (RandomizedSearchCV, n_iter=20, cv=5)

⚠️ **Tiempo real de ejecución: ~1h 51min** (XGBoost ≈ 5265 s ≈ 88 min, RandomForest ≈ 1384 s ≈ 23 min) — mucho más que la estimación previa de "5-10 min"; con `n_iter=20 × cv=5` sobre 1.071 filas × 848 features cada candidato tarda su tiempo. Ejecutar con margen, no en el sandbox de una sesión de asistente. `tune_model.py` ahora persiste el estimador reajustado (`experiments/hyperparams/{modelo}_tuned_model.joblib`) y las métricas completas (`{modelo}_tuned_test_metrics.json`), así que esto no hace falta recalcularlo a mano en el futuro.

| Modelo | Mejor CV F1-w | Test F1-w | Test Acc | Test F1-macro | Test Top-3 | Test Brier | Mejores parámetros |
|---|---|---|---|---|---|---|---|
| XGBoost tuned | 0.6389 | 0.6500 | 0.6698 | 0.5571 | 0.9535 | **0.0801** | `subsample=0.7, n_estimators=200, min_child_weight=7, max_depth=3, learning_rate=0.05, colsample_bytree=0.9` |
| **RandomForest tuned** | **0.6452** | **0.6554** | **0.6744** | **0.5627** | 0.9535 | 0.0908 | `n_estimators=200, min_samples_split=5, min_samples_leaf=4, max_features=sqrt, max_depth=15` |

**Hallazgo importante — el tuning de XGBoost mejoró F1-weighted pero empeoró F1-macro:** F1-w subió de 0.6378 (base) a 0.6500 (tuned, +1.9%), pero F1-macro **bajó** de 0.5916 a 0.5571 (**-5.8% relativo**) y Top-3 accuracy bajó de 0.963 a 0.9535. Esto es coherente con que `RandomizedSearchCV` optimiza explícitamente `scoring="f1_weighted"` (ver `config.yaml → tuning.scoring`): al buscar hiperparámetros que maximicen una métrica ponderada por frecuencia de clase, el buscador implícitamente prioriza acertar en las clases mayoritarias a costa de las minoritarias. Es un ejemplo real y citable de un problema conocido en la literatura de tuning bajo desbalanceo de clases (optimizar la métrica "equivocada" para el objetivo de negocio real). Si el objetivo del sistema es no ignorar sistemáticamente clases minoritarias como "Escalar a manager del lead", **re-tunear con `scoring="f1_macro"`** sería una mejora concreta y fácil de justificar en la memoria (cambiar una línea en `config.yaml`).

En Random Forest el efecto es el contrario y más benigno: tuning mejora F1-w (+1.2%), F1-macro (+1.4%) y Top-3 (+2.5%) a la vez, aunque empeora ligeramente el Brier score (peor calibración: 0.0894 → 0.0908).

**Ranking combinado de los 10 modelos evaluados** (8 base + 2 tuneados), por criterio:

| Criterio | Mejor modelo | Valor |
|---|---|---|
| F1-weighted | RandomForest tuned | 0.6554 |
| F1-macro | **LightGBM (sin tunear)** | 0.6218 |
| Top-3 accuracy | **MLP (sin tunear)** | 0.9674 |
| Calibración (Brier, menor=mejor) | XGBoost tuned | 0.0801 |

**Cuatro modelos distintos ganan en cuatro criterios distintos** — es el resultado honesto a discutir en la memoria en vez de forzar una única conclusión. Recomendación según prioridad de negocio:
- Si prima **accuracy global ponderada** (la mayoría de leads se comportan "normal"): RandomForest tuned.
- Si prima **no ignorar clases minoritarias** (ej. escalado a manager, casos de alto valor): LightGBM, idealmente tuneado.
- Si prima **que el modelo no descarte nunca la acción correcta** (usarlo para sugerir top-3 opciones a un agente humano, no decidir solo): MLP.
- Si prima **confianza calibrada** (usar la probabilidad reportada para decisiones automáticas, no solo el ranking): XGBoost tuned.

Este trabajo mantiene **XGBoost como modelo de producción** (`models/moveup_nextstep_model.joblib`, hiperparámetros base) por ser el que mejor equilibra las cuatro dimensiones sin ganar en ninguna de forma extrema ni perder en ninguna de forma grave — pero es una decisión de diseño explícita a defender, no la única lectura válida de los datos. Con seis familias de modelos evaluadas y ninguna dominando en todo, ese es en sí mismo un argumento defendible: la elección final depende de qué error importa más evitar (falso negativo en una clase minoritaria vs. mala calibración vs. descartar la opción correcta), no de una única métrica ganadora.

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

### Simulación batch (50 leads sintéticos aleatorios, `src/simulate.py`)

Con el modelo de producción actual (dataset ampliado, taxonomía de 6 clases), transcripts por plantilla:

| Estado final | % |
|---|---|
| Nurturing | 62% |
| Lost | 26% |
| Converted | 12% |

Pasos medios hasta estado terminal: 3.2 · Confianza media: 0.586 · `python -m src.simulate` para reproducir (usa plantillas, no GPT — genera embeddings más homogéneos que las transcripciones reales, ver limitación documentada en el docstring del módulo).

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
- [x] ~~Incluir LightGBM en la comparación~~ — completado: mejor F1-macro de los 8 modelos base (0.6218)
- [x] ~~Comparar modelos de familias distintas, no solo variantes de árboles~~ — completado: +Naive Bayes, k-NN, MLP (6 familias en total). MLP gana en Top-3 accuracy (0.967); k-NN/NB rinden claramente peor, probable sensibilidad a la mezcla de escalas embeddings/OHE (ver análisis arriba)
- [ ] Escalar las columnas one-hot antes de pasarlas a k-NN/MLP/Naive Bayes (o usar un pipeline de preprocesado por modelo) para descartar que su rendimiento inferior sea un artefacto de escala y no una limitación real de la familia
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
| `src/config.py` | Carga `config.yaml`, expone el dict `CONFIG` — fuente única de verdad |
| `src/utils.py` | Helpers de I/O (`save_json`, `load_json`, `append_csv_row`) |
| `src/generate_dataset.py` | Genera leads y transcripts con GPT-4o |
| `src/augment_minority.py` | **Augmentación dirigida** de clases minoritarias |
| `src/feature_engineering.py` | Carga + normalización de clases (7→6) + embeddings + features tabulares |
| `src/train_model.py` | Entrena XGBoost, 5-fold CV, guarda artefactos |
| `src/retrain_model.py` | Reentrena sobre el dataset ampliado con el mismo pipeline que `train_model.py` |
| `src/predict.py` | Predicción standalone de siguiente acción |
| `src/summarize.py` | **Resumidor LLM** que genera `prev_outcome` en producción |
| `src/evaluate.py` | Plots (confusión, importancia, calibración) + métricas completas (`evaluate_model`) |
| `src/model_comparison.py` | Compara 8 modelos de 6 familias — Dummy, LogReg, Naive Bayes, k-NN, RF, XGBoost, LightGBM, MLP |
| `src/tune_model.py` | RandomizedSearchCV para XGBoost y RF; persiste estimador + métricas completas |
| `src/validate_dataset.py` | 5 análisis de calidad del dataset sintético |
| `src/simulate.py` | Simulación secuencial de trayectorias de leads |
| `src/pipeline_demo.py` | **Demo visual end-to-end** con resumidor LLM en el loop |
| `src/experiment_tracker.py` | Tracking de experimentos (JSON + CSV, sin MLflow) |

---

## Project structure

```
TFM-Asistente-leads-automatizar-CRM/
├── config.yaml                  <- Single source of truth for all parameters
├── requirements.txt
├── pytest.ini
├── .gitignore
├── run_all.sh                   <- Full pipeline, 9 steps (tests → demo)
├── defensa.sh                   <- Minimal ~5 min defense sequence
├── src/
│   ├── config.py                <- Loads config.yaml, exposes CONFIG dict
│   ├── utils.py                 <- save_json / load_json / append_csv_row
│   ├── generate_dataset.py      <- GPT-4o dataset generator
│   ├── augment_minority.py      <- Targeted augmentation for minority classes
│   ├── feature_engineering.py   <- load_and_clean (incl. 7→6 class normalization) + embeddings + feature matrix
│   ├── train_model.py           <- XGBoost training pipeline (canonical train() + save_model())
│   ├── retrain_model.py         <- Retrain on the augmented dataset, same pipeline as train_model.py
│   ├── predict.py               <- Standalone prediction function
│   ├── summarize.py             <- LLM call summarizer (prev_outcome for production)
│   ├── evaluate.py              <- Evaluation plots + evaluate_model()
│   ├── model_comparison.py      <- 8-model, 6-family benchmark
│   ├── tune_model.py            <- Hyperparameter search (persists tuned estimator + metrics)
│   ├── validate_dataset.py      <- Dataset quality analysis (5 checks)
│   ├── simulate.py              <- Sequential lead simulation
│   ├── pipeline_demo.py         <- End-to-end visual demo
│   └── experiment_tracker.py    <- Experiment logging
├── data/
│   ├── raw/                     <- Generated CSVs
│   └── processed/               <- Cached embeddings (.npz)
├── models/                      <- Saved .joblib artefacts (git-ignored, regenerable)
├── plots/                       <- PNG evaluation plots
├── experiments/                 <- Tracked runs, hyperparams, comparisons
│   ├── runs/                    <- One JSON per experiment run
│   ├── hyperparams/             <- Best params + CV results + tuned test metrics (tuned models themselves git-ignored)
│   ├── validation/              <- Dataset quality report + plots
│   ├── simulations/             <- Batch simulation results
│   └── summary.csv              <- Cumulative experiment index
├── tests/
│   ├── conftest.py
│   ├── fixtures/sample_data.csv <- 50-row real sample for tests
│   ├── test_feature_engineering.py
│   ├── test_predict.py
│   └── test_integration.py
├── notebooks/                   <- Empty — exploratory notebook is pending future work
└── docs/
    ├── architecture.md          <- Pipeline diagrams (mermaid)
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

### 3. Primer arranque

`models/`, `data/processed/embeddings_cache.npz` y `plots/` están vacíos tras clonar (son artefactos regenerables, git-ignored). El dataset (`data/raw/moveup_crm_dataset.csv`) y los resultados de experimentos ligeros (CSVs, JSONs, PNGs de validación) sí vienen en el repo como evidencia. Para generar el modelo desde cero:

```bash
python -m src.train_model    # embeddings (cacheados) + entrena + guarda models/*.joblib + plots/*.png
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

**Verifica:** 8 modelos de 6 familias (Dummy, LogReg, Naive Bayes, k-NN, RF, XGBoost, LightGBM, MLP) sobre el mismo split — la lista la controla `config.yaml → comparison.models`. **Esperado: Random Forest gana en F1-w (~0.648); LightGBM con el mejor F1-macro (~0.622); MLP con el mejor Top-3 (~0.967); k-NN/Naive Bayes claramente peor que árboles/lineal (features mixtas sin escalar); Dummy ~0.115.** Resultados en `experiments/model_comparison.csv`.

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

**Verifica:** simulación secuencial completa. **Esperado (batch): ~12% converted, ~26% lost, ~62% nurturing, 3.2 pasos medios** (ver "Simulación batch" arriba). Resultados en `experiments/simulations/`.

### 8️⃣ Pipeline demo visual con LLM (pieza estrella para la defensa)

```bash
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 python -m src.pipeline_demo --profile banca --seed 47
```

**Verifica:** loop completo end-to-end con resumidor LLM (`gpt-4o-mini`) cerrando el bucle de producción. En cada llamada: lead → transcript → predicción XGBoost → barras ASCII de probabilidades → acción ejecutada → resumen LLM que alimenta el `prev_outcome` de la siguiente llamada. `--profile banca --seed 47` es la combinación validada (ver "Demo end-to-end verificada" arriba): 5 llamadas, la predicción del modelo es siempre la acción ejecutada (sin guion), termina en conversión. El perfil `logistica` (por defecto) queda atrapado en un ciclo de solo 3 de las 6 acciones — es una limitación real del modelo documentada, no un bug.

**Variantes útiles:**

```bash
# Sin LLM (más rápido, sin coste OpenAI, determinista)
python -m src.pipeline_demo --profile banca --seed 47 --no-llm

# Con pausa de 1.5s entre pasos para presentación en vivo
python -m src.pipeline_demo --profile banca --seed 47 --pause 1.5

# Lead aleatorio (no uno de los perfiles fijos)
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
bash defensa.sh
```

Ejecuta, en orden: tests (rigor del código), comparación de modelos (6 familias vs. baseline), y la demo visual end-to-end con LLM y la semilla validada (`--profile banca --seed 47`). Carga `.env.local` automáticamente si existe.

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
| `num_leads` | `500` | Leads to generate on a fresh run of `generate_dataset.py` (current committed dataset has 528: 498 from the original generation + 30 from `augment_minority.py`) |
| `openai_model` | `gpt-4o` | OpenAI model for generation |
| `embedding_model` | `paraphrase-multilingual-MiniLM-L12-v2` | Sentence transformer |
| `paths.raw_data` | `data/raw/moveup_crm_dataset.csv` | Input CSV |
| `paths.model` | `models/moveup_nextstep_model.joblib` | Saved model |
| `experiments_dir` | `experiments` | Experiment tracking root |
| `comparison.*` | see file | Model comparison settings |
| `tuning.*` | see file | RandomizedSearchCV settings |
| `model_params.*` | see file | XGBoost hyperparameters |

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
