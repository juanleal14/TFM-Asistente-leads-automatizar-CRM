"""
tests/conftest.py
-----------------
Shared pytest fixtures for the TFM test suite.

Fixture hierarchy:
  minimal_df      - 10-row DataFrame built inline (no file I/O)
  sample_csv_path - path to tests/fixtures/sample_data.csv (50 real rows)
  model_path      - path to models/moveup_nextstep_model.joblib
  model_available - bool: whether the trained model exists

NOTE: TOKENIZERS_PARALLELISM and OMP_NUM_THREADS are set at import time to
prevent a macOS segfault caused by the XGBoost (OpenMP) + sentence-transformers
(PyTorch) combination in the same process.  Does not affect prediction quality.
"""
from __future__ import annotations

import io
import os
import pathlib

import numpy as np
import pandas as pd
import pytest

# Must be set BEFORE any torch / xgboost import
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

PROJECT_ROOT = pathlib.Path(__file__).parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "moveup_nextstep_model.joblib"
SAMPLE_CSV = PROJECT_ROOT / "tests" / "fixtures" / "sample_data.csv"

# ── Path fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def model_path() -> pathlib.Path:
    return MODEL_PATH


@pytest.fixture(scope="session")
def model_available() -> bool:
    return MODEL_PATH.exists()


@pytest.fixture(scope="session")
def sample_csv_path() -> pathlib.Path:
    return SAMPLE_CSV


# ── Minimal inline DataFrame (no file I/O, no model required) -----------------

_MINIMAL_CSV = """interaction_id,lead_id,contact_name,contact_role,company_name,company_sector,company_country,company_city,company_num_employees,company_annual_revenue_eur,lead_source,lead_entry_date,initial_interest_notes,call_number,call_timestamp,days_since_entry,days_since_last_call,prev_outcome,prev_next_step,current_transcript,current_outcome,next_step,final_status
row-01,lead-01,Ana Garcia,CEO,Tech A,Tecnologia,Espana,Madrid,100,10000000.0,LinkedIn,2025-01-01,Interes en movilidad.,1,2025-01-06,5,0,,,Agente: Hola. Contacto: Buenos dias.,Positiva,Enviar documentacion,Converted
row-02,lead-01,Ana Garcia,CEO,Tech A,Tecnologia,Espana,Madrid,100,10000000.0,LinkedIn,2025-01-01,Interes en movilidad.,2,2025-01-13,12,7,Interes confirmado.,Enviar documentacion,Agente: Seguimiento. Contacto: Revise el material.,Interes mantenido,Agendar demo/reunion con especialista,Converted
row-03,lead-02,Carlos Lopez,CFO,Corp B,Logistica,Espana,Barcelona,250,30000000.0,Formulario web,2025-01-05,Gestion de flota.,1,2025-01-10,5,0,,,Agente: Hola Corp B. Contacto: Si cuenteme.,Primera toma de contacto,Recontactar en X dias,Nurturing
row-04,lead-03,Maria Ruiz,Director General,Empresa C,Banca,Espana,Valencia,50,5000000.0,Llamada en frio,2025-01-08,Sin notas.,1,2025-01-09,1,0,,,Agente: Hola Empresa C. Contacto: Estamos ocupados.,Sin interes inmediato,Cerrar lead - no interesado,Lost
row-05,lead-04,Javier Torres,COO,Global D,Retail,Espana,Sevilla,500,80000000.0,Feria o evento,2025-01-10,Vio la plataforma en feria.,1,2025-01-15,5,0,,,Agente: Seguimiento feria. Contacto: Si recordamos.,Interes post-feria,Enviar documentacion,In Progress
row-06,lead-05,Sofia Moreno,Director de RRHH,Services E,Consultoria IT,Espana,Bilbao,75,8000000.0,Email marketing,2025-01-12,Email campaign.,1,2025-01-17,5,0,,,Agente: Email follow-up. Contacto: Si lo vi.,Contacto abierto,Esperar confirmacion cliente,Nurturing
row-07,lead-06,Pablo Fernandez,Director Comercial,Industria F,Manufactura,Espana,Zaragoza,200,25000000.0,Google Ads,2025-01-14,Google lead.,1,2025-01-19,5,0,,,Agente: Google lead. Contacto: Me aparecieron en busqueda.,Primera llamada,Recontactar en X dias,In Progress
row-08,lead-07,Laura Diaz,Director de Operaciones,Pharma G,Farmaceutica,Espana,Madrid,400,60000000.0,Partner comercial,2025-01-15,Partner referral.,1,2025-01-20,5,0,,,Agente: Referencia partner. Contacto: Si nos lo recomendaron.,Buena predisposicion,Agendar demo/reunion con especialista,Converted
row-09,lead-08,Diego Sanchez,Director de Logistica,Energy H,Energia,Espana,Barcelona,1000,200000000.0,Referencia de cliente,2025-01-16,Cliente actual referencia.,1,2025-01-21,5,0,,,Agente: Referencia cliente. Contacto: Si somos del mismo grupo.,Alta intencion,Escalar a manager del lead,In Progress
row-10,lead-09,Marta Iglesias,Director Financiero,Auto I,Automocion,Espana,Madrid,800,150000000.0,Inbound / Blog,2025-01-18,Blog article.,1,2025-01-23,5,0,,,Agente: Blog reader. Contacto: Si lei el articulo.,Interes tecnico,Cerrar lead - nurturing,Nurturing
"""


@pytest.fixture
def minimal_df() -> pd.DataFrame:
    """10-row DataFrame built inline -- no file I/O, no model required."""
    return pd.read_csv(io.StringIO(_MINIMAL_CSV))


@pytest.fixture
def dummy_embeddings(minimal_df: pd.DataFrame) -> np.ndarray:
    """Zero-valued embeddings matching the shape expected by build_feature_matrix."""
    return np.zeros((len(minimal_df), 768))
