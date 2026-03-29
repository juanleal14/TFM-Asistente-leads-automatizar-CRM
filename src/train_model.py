"""
MoveUp CRM - Paso 2: Feature Engineering + Entrenamiento del Modelo
====================================================================
Lee el CSV crudo generado en el Paso 1, transforma los datos en features
numéricos (embeddings + encoding) y entrena un modelo de clasificación
para predecir el next_step.

Requisitos:
    pip install pandas scikit-learn xgboost sentence-transformers matplotlib seaborn

Uso:
    python train_model.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
import os
from datetime import datetime

from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay, accuracy_score, f1_score
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────
INPUT_FILE = "moveup_crm_dataset.csv"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # multilingüe, bueno para español
EMBEDDING_CACHE_FILE = "embeddings_cache.npz"
OUTPUT_MODEL_FILE = "moveup_nextstep_model.joblib"
OUTPUT_ENCODERS_FILE = "moveup_encoders.joblib"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Features categóricos y numéricos
CAT_FEATURES = [
    "company_sector", "company_country", "company_city",
    "lead_source", "contact_role", "prev_outcome", "prev_next_step"
]
NUM_FEATURES = [
    "company_num_employees", "company_annual_revenue_eur",
    "call_number", "days_since_entry", "days_since_last_call"
]
TARGET = "next_step"


# ─────────────────────────────────────────────
# PASO 2.1: CARGA Y LIMPIEZA
# ─────────────────────────────────────────────
def load_and_clean(filepath: str) -> pd.DataFrame:
    """Carga el CSV crudo y hace limpieza básica."""
    print("📂 Cargando dataset...")
    df = pd.read_csv(filepath, encoding="utf-8-sig")

    print(f"   Filas totales: {len(df)}")
    print(f"   Leads únicos: {df['lead_id'].nunique()}")
    print(f"   Columnas: {list(df.columns)}")

    # Rellenar vacíos en prev_outcome y prev_next_step (primera llamada)
    df["prev_outcome"] = df["prev_outcome"].fillna("PRIMERA_LLAMADA").replace("", "PRIMERA_LLAMADA")
    df["prev_next_step"] = df["prev_next_step"].fillna("PRIMERA_LLAMADA").replace("", "PRIMERA_LLAMADA")

    # Verificar que el target no tiene nulos
    assert df[TARGET].notna().all(), "⚠️ Hay valores nulos en la columna target (next_step)"

    # Stats del target
    print(f"\n📊 Distribución del target ({TARGET}):")
    for step, count in df[TARGET].value_counts().items():
        pct = count / len(df) * 100
        print(f"   {step}: {count} ({pct:.1f}%)")

    return df


# ─────────────────────────────────────────────
# PASO 2.2: EMBEDDINGS DE TRANSCRIPTS
# ─────────────────────────────────────────────
def generate_embeddings(df: pd.DataFrame) -> np.ndarray:
    """Genera embeddings de los transcripts con sentence-transformers."""

    # Usar caché si existe (los embeddings tardan)
    if os.path.exists(EMBEDDING_CACHE_FILE):
        print(f"📦 Cargando embeddings desde caché ({EMBEDDING_CACHE_FILE})...")
        data = np.load(EMBEDDING_CACHE_FILE)
        cached = data["embeddings"]
        if len(cached) == len(df):
            print(f"   Forma: {cached.shape}")
            return cached
        else:
            print("   ⚠️ Caché no coincide en tamaño, regenerando...")

    print(f"🧠 Generando embeddings con {EMBEDDING_MODEL}...")
    print("   (Esto puede tardar unos minutos la primera vez)")

    model = SentenceTransformer(EMBEDDING_MODEL)

    # Embedding del transcript actual
    transcripts = df["current_transcript"].tolist()
    embeddings_current = model.encode(
        transcripts,
        show_progress_bar=True,
        batch_size=32,
        normalize_embeddings=True
    )

    # Embedding del prev context (initial_interest_notes + prev_outcome combinados)
    # Esto le da al modelo contexto de "qué pasó antes" más allá del campo categórico
    prev_context = (
        df["initial_interest_notes"].fillna("") + " | " +
        df["prev_outcome"].fillna("")
    ).tolist()
    embeddings_prev = model.encode(
        prev_context,
        show_progress_bar=True,
        batch_size=32,
        normalize_embeddings=True
    )

    # Concatenar ambos embeddings
    embeddings = np.hstack([embeddings_current, embeddings_prev])

    print(f"   Embedding transcript actual: {embeddings_current.shape}")
    print(f"   Embedding contexto previo: {embeddings_prev.shape}")
    print(f"   Embedding combinado: {embeddings.shape}")

    # Guardar caché
    np.savez_compressed(EMBEDDING_CACHE_FILE, embeddings=embeddings)
    print(f"   💾 Caché guardada: {EMBEDDING_CACHE_FILE}")

    return embeddings


# ─────────────────────────────────────────────
# PASO 2.3: CONSTRUIR FEATURE MATRIX
# ─────────────────────────────────────────────
def build_feature_matrix(df: pd.DataFrame, embeddings: np.ndarray):
    """Combina features tabulares + embeddings en una sola matriz."""

    print("\n🔧 Construyendo feature matrix...")

    # --- Features numéricos ---
    X_num = df[NUM_FEATURES].values.astype(float)
    print(f"   Features numéricos: {X_num.shape[1]} columnas")

    # --- Features categóricos (one-hot) ---
    cat_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_cat = cat_encoder.fit_transform(df[CAT_FEATURES].astype(str))
    cat_feature_names = cat_encoder.get_feature_names_out(CAT_FEATURES)
    print(f"   Features categóricos (one-hot): {X_cat.shape[1]} columnas")

    # --- Embeddings ---
    print(f"   Embeddings: {embeddings.shape[1]} columnas")

    # --- Combinar todo ---
    X = np.hstack([X_num, X_cat, embeddings])
    print(f"   Feature matrix final: {X.shape}")

    # --- Target ---
    label_enc = LabelEncoder()
    y = label_enc.fit_transform(df[TARGET])
    print(f"   Clases del target: {list(label_enc.classes_)}")

    return X, y, label_enc, cat_encoder, cat_feature_names


# ─────────────────────────────────────────────
# PASO 2.4: ENTRENAMIENTO
# ─────────────────────────────────────────────
def train_model(X: np.ndarray, y: np.ndarray, label_enc: LabelEncoder):
    """Entrena XGBoost con validación cruzada y evalúa en test set."""

    print("\n🏋️ Entrenando modelo XGBoost...")

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"   Train: {X_train.shape[0]} filas | Test: {X_test.shape[0]} filas")

    # Escalar features numéricos (los primeros NUM_FEATURES columnas)
    scaler = StandardScaler()
    n_num = len(NUM_FEATURES)
    X_train[:, :n_num] = scaler.fit_transform(X_train[:, :n_num])
    X_test[:, :n_num] = scaler.transform(X_test[:, :n_num])

    # Modelo XGBoost
    n_classes = len(label_enc.classes_)
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        use_label_encoder=False,
        verbosity=0
    )

    # Validación cruzada en train
    print("\n📐 Validación cruzada (5-fold) en train set...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_weighted")
    print(f"   F1 (weighted) por fold: {[f'{s:.3f}' for s in cv_scores]}")
    print(f"   F1 medio: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Entrenar modelo final en todo el train set
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    # Evaluar en test
    y_pred = model.predict(X_test)

    print(f"\n📊 Resultados en TEST set:")
    print(f"   Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"   F1 (weighted): {f1_score(y_test, y_pred, average='weighted'):.3f}")

    print(f"\n📋 Classification Report:")
    target_names = list(label_enc.classes_)
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

    return model, scaler, X_test, y_test, y_pred


# ─────────────────────────────────────────────
# PASO 2.5: VISUALIZACIONES
# ─────────────────────────────────────────────
def plot_results(y_test, y_pred, label_enc, model, cat_feature_names):
    """Genera gráficos de evaluación del modelo."""

    os.makedirs("plots", exist_ok=True)
    target_names = list(label_enc.classes_)

    # 1. Confusion Matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
    ax.set_title("Confusion Matrix - Next Step Prediction", fontsize=14)
    plt.tight_layout()
    plt.savefig("plots/confusion_matrix.png", dpi=150)
    plt.close()
    print("   📈 Guardado: plots/confusion_matrix.png")

    # 2. Feature Importance (top 30)
    fig, ax = plt.subplots(figsize=(10, 8))

    # Construir nombres de features
    num_names = NUM_FEATURES
    cat_names = list(cat_feature_names)
    emb_current_names = [f"emb_transcript_{i}" for i in range(384)]
    emb_prev_names = [f"emb_prev_context_{i}" for i in range(384)]
    all_names = num_names + cat_names + emb_current_names + emb_prev_names

    importances = model.feature_importances_
    # Tomar top 30
    top_idx = np.argsort(importances)[-30:]
    top_names = [all_names[i] if i < len(all_names) else f"feat_{i}" for i in top_idx]
    top_values = importances[top_idx]

    ax.barh(range(len(top_names)), top_values, color="steelblue")
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names, fontsize=8)
    ax.set_xlabel("Importance")
    ax.set_title("Top 30 Feature Importances", fontsize=14)
    plt.tight_layout()
    plt.savefig("plots/feature_importance.png", dpi=150)
    plt.close()
    print("   📈 Guardado: plots/feature_importance.png")

    # 3. Distribución de predicciones vs real
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    real_counts = pd.Series(y_test).map(dict(enumerate(target_names))).value_counts()
    pred_counts = pd.Series(y_pred).map(dict(enumerate(target_names))).value_counts()

    real_counts.plot(kind="barh", ax=axes[0], color="steelblue")
    axes[0].set_title("Distribución Real (Test)")

    pred_counts.plot(kind="barh", ax=axes[1], color="coral")
    axes[1].set_title("Distribución Predicha (Test)")

    plt.tight_layout()
    plt.savefig("plots/distribution_comparison.png", dpi=150)
    plt.close()
    print("   📈 Guardado: plots/distribution_comparison.png")


# ─────────────────────────────────────────────
# PASO 2.6: GUARDAR MODELO Y ARTEFACTOS
# ─────────────────────────────────────────────
def save_artifacts(model, scaler, label_enc, cat_encoder):
    """Guarda el modelo entrenado y los encoders para producción."""

    artifacts = {
        "model": model,
        "scaler": scaler,
        "label_encoder": label_enc,
        "cat_encoder": cat_encoder,
        "num_features": NUM_FEATURES,
        "cat_features": CAT_FEATURES,
        "embedding_model_name": EMBEDDING_MODEL,
        "trained_at": datetime.now().isoformat()
    }

    joblib.dump(artifacts, OUTPUT_MODEL_FILE)
    print(f"\n💾 Modelo guardado: {OUTPUT_MODEL_FILE}")
    print(f"   Para cargar: artifacts = joblib.load('{OUTPUT_MODEL_FILE}')")


# ─────────────────────────────────────────────
# PASO 2.7: FUNCIÓN DE PREDICCIÓN (PRODUCCIÓN)
# ─────────────────────────────────────────────
def predict_next_step(
    artifacts_path: str,
    company_sector: str,
    company_country: str,
    company_city: str,
    company_num_employees: int,
    company_annual_revenue_eur: int,
    lead_source: str,
    contact_role: str,
    call_number: int,
    days_since_entry: int,
    days_since_last_call: int,
    prev_outcome: str,
    prev_next_step: str,
    current_transcript: str,
    initial_interest_notes: str = ""
) -> dict:
    """
    Predice el next_step para una nueva interacción.
    Esto es lo que se llamaría en producción cuando entra un nuevo transcript.

    Returns:
        dict con "predicted_next_step" y "probabilities" por categoría.
    """
    # Cargar artefactos
    arts = joblib.load(artifacts_path)
    model = arts["model"]
    scaler = arts["scaler"]
    label_enc = arts["label_encoder"]
    cat_enc = arts["cat_encoder"]

    # Generar embeddings
    emb_model = SentenceTransformer(arts["embedding_model_name"])
    emb_current = emb_model.encode([current_transcript], normalize_embeddings=True)
    prev_context = f"{initial_interest_notes} | {prev_outcome}"
    emb_prev = emb_model.encode([prev_context], normalize_embeddings=True)
    embeddings = np.hstack([emb_current, emb_prev])

    # Features numéricos
    X_num = np.array([[
        company_num_employees, company_annual_revenue_eur,
        call_number, days_since_entry, days_since_last_call
    ]], dtype=float)
    X_num = scaler.transform(X_num)

    # Features categóricos
    cat_df = pd.DataFrame([{
        "company_sector": company_sector,
        "company_country": company_country,
        "company_city": company_city,
        "lead_source": lead_source,
        "contact_role": contact_role,
        "prev_outcome": prev_outcome if prev_outcome else "PRIMERA_LLAMADA",
        "prev_next_step": prev_next_step if prev_next_step else "PRIMERA_LLAMADA"
    }])
    X_cat = cat_enc.transform(cat_df.astype(str))

    # Combinar
    X = np.hstack([X_num, X_cat, embeddings])

    # Predecir
    pred = model.predict(X)[0]
    probs = model.predict_proba(X)[0]

    result = {
        "predicted_next_step": label_enc.inverse_transform([pred])[0],
        "confidence": float(probs[pred]),
        "probabilities": {
            label_enc.inverse_transform([i])[0]: float(p)
            for i, p in enumerate(probs)
        }
    }

    return result


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("MoveUp CRM - Feature Engineering + Entrenamiento")
    print("=" * 60)

    # 2.1 Cargar datos
    df = load_and_clean(INPUT_FILE)

    # 2.2 Generar embeddings
    embeddings = generate_embeddings(df)

    # 2.3 Construir features
    X, y, label_enc, cat_encoder, cat_feature_names = build_feature_matrix(df, embeddings)

    # 2.4 Entrenar modelo
    model, scaler, X_test, y_test, y_pred = train_model(X, y, label_enc)

    # 2.5 Visualizaciones
    print("\n📊 Generando visualizaciones...")
    plot_results(y_test, y_pred, label_enc, model, cat_feature_names)

    # 2.6 Guardar
    save_artifacts(model, scaler, label_enc, cat_encoder)

    # 2.7 Ejemplo de predicción
    print("\n" + "=" * 60)
    print("🔮 EJEMPLO DE PREDICCIÓN CON NUEVO LEAD:")
    print("=" * 60)

    example_result = predict_next_step(
        artifacts_path=OUTPUT_MODEL_FILE,
        company_sector="Consultoría IT",
        company_country="España",
        company_city="Madrid",
        company_num_employees=200,
        company_annual_revenue_eur=15000000,
        lead_source="Formulario web",
        contact_role="Head of People",
        call_number=1,
        days_since_entry=1,
        days_since_last_call=0,
        prev_outcome="",
        prev_next_step="",
        current_transcript="""Agente: Buenos días, soy Carlos de MoveUp. Veo que habéis pedido información.
Contacto: Sí, tenemos 150 consultores desplazándose cada semana y el control de gastos es un caos.
Agente: Justo para eso estamos. ¿Usáis alguna solución ahora?
Contacto: No, cada uno va por su cuenta con taxis y luego pasa la factura. Es inmanejable.
Agente: Con MoveUp centralizáis todo. ¿Os interesaría una demo?
Contacto: Sí, pero estamos en plena reestructuración. ¿Podéis llamarme en 3 semanas?""",
        initial_interest_notes="Consultora IT con 150 consultores, busca centralizar movilidad."
    )

    print(f"\n   ✅ Predicción: {example_result['predicted_next_step']}")
    print(f"   📊 Confianza: {example_result['confidence']:.1%}")
    print(f"\n   Probabilidades por categoría:")
    for step, prob in sorted(example_result["probabilities"].items(), key=lambda x: -x[1]):
        bar = "█" * int(prob * 30)
        print(f"      {step:45s} {prob:.1%} {bar}")


if __name__ == "__main__":
    main()
