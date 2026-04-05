# MoveUp Next-Action Predictor

A machine-learning system that predicts the **next best sales action** for B2B leads in a corporate mobility CRM pipeline.

Built as a Master's thesis (TFM) project. The fictional company **MoveUp** offers corporate mobility services (Uber for Business style) across Spain and Latin America.

---

## What it does

| Step | Script | Description |
|---|---|---|
| 1 | `src/generate_dataset.py` | Calls GPT-4o to generate realistic Spanish call transcripts for synthetic CRM leads |
| 2 | `src/feature_engineering.py` | Embeds transcripts, one-hot encodes categoricals, scales numerics |
| 3 | `src/train_model.py` | Trains XGBoost, runs 5-fold CV, saves all artefacts |
| 4 | `src/predict.py` | Loads saved model and predicts next action for a new interaction |
| 5 | `src/evaluate.py` | Generates confusion matrix, feature importance, and distribution plots |

---

## Project structure

```
moveup-next-action-predictor/
├── config.yaml                  ← Single source of truth for all parameters
├── requirements.txt
├── .gitignore
├── src/
│   ├── config.py                ← Loads config.yaml, exposes CONFIG dict
│   ├── generate_dataset.py      ← GPT-4o dataset generator
│   ├── feature_engineering.py   ← Embeddings + tabular features
│   ├── train_model.py           ← XGBoost training pipeline
│   ├── predict.py               ← Standalone prediction function
│   ├── evaluate.py              ← Evaluation plots
│   └── utils.py                 ← JSON helpers
├── data/
│   ├── raw/                     ← Generated CSVs (git-ignored)
│   └── processed/               ← Cached embeddings (.npz, git-ignored)
├── models/                      ← Saved .joblib artefacts (git-ignored)
├── plots/                       ← PNG evaluation plots (git-ignored)
├── notebooks/                   ← Jupyter notebooks
├── tests/                       ← Pytest test stubs
└── docs/
    └── memoria_tfm.md           ← TFM memoir skeleton
```

---

## Quickstart

### 1. Install dependencies

```bash
cd moveup-next-action-predictor
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set your OpenAI API key

```bash
export OPENAI_API_KEY="sk-..."    # Windows: set OPENAI_API_KEY=sk-...
```

### 3. Generate the dataset

```bash
# Default: 2 leads (set num_leads: 500 in config.yaml for the full dataset)
python -m src.generate_dataset
```

Output: `data/raw/moveup_crm_dataset.csv`

### 4. Train the model

```bash
python -m src.train_model
```

Output:
- `models/moveup_nextstep_model.joblib`
- `plots/confusion_matrix.png`
- `plots/feature_importance.png`
- `plots/distribution_comparison.png`

### 5. Predict

```bash
python -m src.predict
```

Output (example):

```json
{
  "predicted_next_step": "Agendar demo/reunión con especialista",
  "confidence": 0.7812,
  "probabilities": {
    "Agendar demo/reunión con especialista": 0.7812,
    "Enviar documentación": 0.0934,
    "Recontactar en X días": 0.0541,
    ...
  }
}
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
```

- **Numeric (5):** employees, revenue, call_number, days_since_entry, days_since_last_call — scaled with `StandardScaler`
- **Categorical (7):** sector, country, city, lead_source, contact_role, prev_outcome, prev_next_step — encoded with `OneHotEncoder`
- **Embeddings (768):** `paraphrase-multilingual-MiniLM-L12-v2` applied to:
  - `current_transcript` → 384 dims
  - `initial_interest_notes + prev_outcome` → 384 dims

### Model

XGBoost multi-class classifier (`multi:softprob`):

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
| `num_leads` | `2` | Leads to generate (set to 500 for full dataset) |
| `openai_model` | `gpt-4o` | OpenAI model for generation |
| `embedding_model` | `paraphrase-multilingual-MiniLM-L12-v2` | Sentence transformer |
| `paths.raw_data` | `data/raw/moveup_crm_dataset.csv` | Input CSV |
| `paths.model` | `models/moveup_nextstep_model.joblib` | Saved model |
| `model_params.*` | see file | XGBoost hyperparameters |

---

## Running tests

```bash
pytest tests/ -v
```

Most tests are stubs (`pytest.skip`) pending dataset and model availability.

---

## Tech stack

- **Python 3.11+**
- **OpenAI API** — GPT-4o for transcript generation
- **sentence-transformers** — multilingual semantic embeddings
- **XGBoost** — gradient boosting classifier
- **scikit-learn** — preprocessing and evaluation
- **pandas / numpy** — data manipulation
- **matplotlib / seaborn** — visualisation
- **joblib** — model serialisation
- **PyYAML** — configuration management
