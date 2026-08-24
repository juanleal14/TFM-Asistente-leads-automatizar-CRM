"""
retrain_model.py
────────────────
Reentrenamiento del modelo con las 6 acciones válidas actuales.

Este script:
1. Mapea clases antiguas a las 6 acciones válidas
2. Genera embeddings
3. Entrena XGBoost + Random Forest
4. Guarda el nuevo modelo

Uso:
    python -m src.retrain_model [--output-dir models/]
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

from src.config import CONFIG, resolve_path
from src.feature_engineering import load_and_clean, generate_embeddings, build_feature_matrix
from src.train_model import train, save_model
from src.utils import save_json

# ── Mapeo de clases antiguas a clases válidas ─────────────────────────────────
CLASS_MAPPING = {
    # Clases antiguas → Clases válidas (6 acciones)
    "Cerrar lead - nurturing": "Aplazar lead",
    "Recontactar en X días": "Aplazar lead",
    "Enviar documentación": "Enviar documentación",
    "Agendar demo/reunión con especialista": "Agendar demo/reunión con especialista",
    "Escalar a manager del lead": "Escalar a manager del lead",
    "Esperar confirmación cliente": "Esperar confirmación cliente",
    "Cerrar lead - no interesado": "Cerrar lead - no interesado",
    "Aplazar lead": "Aplazar lead",  # Ya válida
}

VALID_ACTIONS = [
    "Aplazar lead",
    "Enviar documentación",
    "Agendar demo/reunión con especialista",
    "Escalar a manager del lead",
    "Esperar confirmación cliente",
    "Cerrar lead - no interesado",
]


def normalize_classes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza las clases en el dataset a las 6 acciones válidas.
    """
    df = df.copy()
    df["next_step"] = df["next_step"].map(CLASS_MAPPING)
    
    # Filtrar filas con clases no mapeadas
    before = len(df)
    df = df[df["next_step"].isin(VALID_ACTIONS)]
    after = len(df)
    
    print(f"✓ Dataset normalizado: {before} → {after} filas")
    print(f"\n  Distribución de clases:")
    print(df["next_step"].value_counts())
    
    return df


def retrain_model(data_path: str = None, output_dir: str = None):
    """
    Entrena un nuevo modelo con las 6 acciones válidas.
    """
    if output_dir is None:
        output_dir = "models"
    
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(exist_ok=True)
    
    print("════════════════════════════════════════════════════════════════")
    print("  REENTRENAMIENTO CON 6 ACCIONES VÁLIDAS")
    print("════════════════════════════════════════════════════════════════\n")
    
    # Cargar dataset
    print("1. Cargando dataset...")
    df = load_and_clean(data_path)
    print(f"   Filas originales: {len(df)}\n")
    
    # Normalizar clases
    print("2. Normalizando clases...")
    df = normalize_classes(df)
    print()
    
    # Generar embeddings
    print("3. Generando embeddings...")
    embeddings = generate_embeddings(df)
    print(f"   Embeddings: {embeddings.shape}\n")
    
    # Construir feature matrix
    print("4. Construyendo feature matrix...")
    X, y, scaler, cat_encoder, label_encoder, feature_names = build_feature_matrix(
        df, embeddings, fit=True
    )
    print(f"   Features: {X.shape}")
    print(f"   Target: {y.shape}\n")
    
    # Entrenar (mismo pipeline que train_model.py: XGBoost + split train/test + CV)
    print("5. Entrenando XGBoost (con split train/test)...")
    model, X_test, y_test = train(X, y, num_classes=len(label_encoder.classes_))

    y_pred = np.argmax(model.predict_proba(X_test), axis=1)
    present = np.union1d(y_test, y_pred)
    report = classification_report(
        y_test, y_pred,
        labels=present,
        target_names=label_encoder.inverse_transform(present),
        zero_division=0,
    )
    print(report)

    # Guardar modelo + artefacts en el MISMO formato que espera predict.py
    print("6. Guardando artefacts...")
    model_path = output_dir_path / "moveup_nextstep_model.joblib"
    save_model(model, scaler, cat_encoder, label_encoder, feature_names, model_path=model_path)

    # Estadísticas
    stats = {
        "training_samples": len(df),
        "valid_actions": VALID_ACTIONS,
        "class_distribution": df["next_step"].value_counts().to_dict(),
        "feature_count": X.shape[1],
        "classification_report": classification_report(
            y_test, y_pred,
            labels=present,
            target_names=label_encoder.inverse_transform(present),
            zero_division=0,
            output_dict=True,
        ),
        "timestamp": pd.Timestamp.now().isoformat(),
    }
    stats_path = output_dir_path / "retrain_stats.json"
    save_json(stats, stats_path)
    print(f"   ✓ Guardado: {stats_path}\n")

    print("════════════════════════════════════════════════════════════════")
    print("  ✓ REENTRENAMIENTO COMPLETADO")
    print("════════════════════════════════════════════════════════════════\n")
    print(f"Modelo guardado en: {model_path}")
    print(f"Acciones válidas: {len(VALID_ACTIONS)}\n")


if __name__ == "__main__":
    data_path = None
    output_dir = "models"
    
    if "--data-path" in sys.argv:
        idx = sys.argv.index("--data-path")
        data_path = sys.argv[idx + 1]
    
    if "--output-dir" in sys.argv:
        idx = sys.argv.index("--output-dir")
        output_dir = sys.argv[idx + 1]
    
    retrain_model(data_path=data_path, output_dir=output_dir)
