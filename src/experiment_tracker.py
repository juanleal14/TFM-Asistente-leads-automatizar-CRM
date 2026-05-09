"""
experiment_tracker.py
─────────────────────
Simple file-based experiment tracking for TFM.
No MLflow required — each run is a JSON file; a summary CSV is maintained.

Directory layout (under experiments/):
  runs/{run_id}.json          — full run record
  hyperparams/{model}_best_params.json
  hyperparams/{model}_cv_results.csv
  summary.csv                 — one row per run, append-only

Usage:
    from src.experiment_tracker import ExperimentTracker, list_experiments, get_best_run

    tracker = ExperimentTracker()
    run_id  = tracker.start_run("xgboost_baseline", config={"random_state": 42})
    tracker.log_model_info("XGBoost", {"n_estimators": 300})
    tracker.log_metrics({"accuracy": 0.85, "f1_weighted": 0.83})
    tracker.end_run()
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import CONFIG


# ── JSON encoder that handles numpy scalars ───────────────────────────────────

class _NumpyEncoder(json.JSONEncoder):
    """Convert numpy scalar types to native Python for JSON serialisation."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ── Helper: resolve experiments directory ─────────────────────────────────────

def _resolve_experiments_dir(experiments_dir: Path | None) -> Path:
    if experiments_dir is not None:
        return Path(experiments_dir)
    experiments_key = CONFIG.get("experiments_dir", "experiments")
    from src.config import _PROJECT_ROOT  # type: ignore[attr-defined]
    return _PROJECT_ROOT / experiments_key


# ── Summary CSV columns (flat schema) ─────────────────────────────────────────

_SUMMARY_COLS = [
    "run_id", "experiment_name", "timestamp", "model_name",
    "accuracy", "f1_macro", "f1_weighted",
    "precision_macro", "precision_weighted",
    "recall_macro", "recall_weighted",
    "top3_accuracy", "brier_score_avg",
    "cv_f1_weighted_mean", "cv_f1_weighted_std",
    "duration_seconds",
]


def _extract_standard_metric(metrics: dict, standard_name: str) -> float | None:
    """Localiza el valor correspondiente a una métrica estándar dentro de un dict
    que puede usar nombres prefijados.

    Reglas de búsqueda (en orden):
    1. Coincidencia exacta:       metrics[standard_name]
    2. Sufijo estándar:           cualquier key que termine en "_{standard_name}"
                                  (ej: "best_f1_weighted", "xgboost_f1_weighted")
                                  → devuelve el MÁXIMO de los valores numéricos.
    3. None si no hay coincidencia.

    Esto permite que el summary CSV se rellene correctamente cuando
    model_comparison.py y tune_model.py loggean métricas con prefijo de modelo.
    """
    if standard_name in metrics:
        v = metrics[standard_name]
        return float(v) if isinstance(v, (int, float)) else None

    suffix = f"_{standard_name}"
    candidates = [
        v for k, v in metrics.items()
        if k.endswith(suffix) and isinstance(v, (int, float))
    ]
    return float(max(candidates)) if candidates else None


# ── ExperimentTracker class ───────────────────────────────────────────────────

class ExperimentTracker:
    """Lightweight experiment tracker backed by JSON files and a CSV index."""

    def __init__(self, experiments_dir: Path | None = None) -> None:
        self._base = _resolve_experiments_dir(experiments_dir)
        self._runs_dir = self._base / "runs"
        self._hp_dir = self._base / "hyperparams"
        self._summary_path = self._base / "summary.csv"

        # Create directory structure
        for d in (self._runs_dir, self._hp_dir):
            d.mkdir(parents=True, exist_ok=True)

        # Internal state of the active run
        self._current_run: dict | None = None
        self._start_time: float | None = None

    # ── Public API ────────────────────────────────────────────────────────────

    def start_run(
        self,
        experiment_name: str,
        config: dict | None = None,
    ) -> str:
        """Start a new run.  Returns the generated run_id."""
        timestamp = datetime.now()
        run_id = f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{experiment_name}"

        self._current_run = {
            "run_id": run_id,
            "experiment_name": experiment_name,
            "timestamp": timestamp.isoformat(),
            "config_snapshot": config or {},
            "model_info": {},
            "metrics": {},
        }
        self._start_time = time.perf_counter()
        return run_id

    def log_metrics(self, metrics_dict: dict) -> None:
        """Merge *metrics_dict* into the active run's metrics."""
        self._assert_active()
        self._current_run["metrics"].update(metrics_dict)  # type: ignore[index]

    def log_model_info(self, model_name: str, params: dict) -> None:
        """Record the model name and its hyperparameters."""
        self._assert_active()
        self._current_run["model_info"] = {"model_name": model_name, "params": params}  # type: ignore[index]

    def end_run(self) -> Path:
        """Close the active run, write JSON and update summary.csv."""
        self._assert_active()
        assert self._start_time is not None

        duration = time.perf_counter() - self._start_time
        self._current_run["duration_seconds"] = round(duration, 3)  # type: ignore[index]

        run = self._current_run
        run_id = run["run_id"]

        # Write full JSON record
        json_path = self._runs_dir / f"{run_id}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(run, f, ensure_ascii=False, indent=2, cls=_NumpyEncoder)

        # Append flat row to summary CSV
        metrics = run.get("metrics", {})
        model_info = run.get("model_info", {})
        summary_row = {col: None for col in _SUMMARY_COLS}
        summary_row.update({
            "run_id": run_id,
            "experiment_name": run["experiment_name"],
            "timestamp": run["timestamp"],
            "model_name": model_info.get("model_name", ""),
            "duration_seconds": run["duration_seconds"],
        })
        # Skip non-metric columns when scanning
        _meta_cols = {"run_id", "experiment_name", "timestamp",
                      "model_name", "duration_seconds"}
        for col in _SUMMARY_COLS:
            if col in _meta_cols:
                continue
            value = _extract_standard_metric(metrics, col)
            if value is not None:
                summary_row[col] = value

        exists = self._summary_path.exists()
        pd.DataFrame([summary_row]).to_csv(
            self._summary_path, mode="a", header=not exists, index=False
        )

        # Reset state
        self._current_run = None
        self._start_time = None

        print(f"  [tracker] Run saved → {json_path}")
        return json_path

    def save_hyperparams(self, model_name: str, best_params: dict) -> Path:
        """Persist best hyperparams for *model_name* (no active run required)."""
        path = self._hp_dir / f"{model_name}_best_params.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(best_params, f, ensure_ascii=False, indent=2, cls=_NumpyEncoder)
        print(f"  [tracker] Best params saved → {path}")
        return path

    # ── Private helpers ───────────────────────────────────────────────────────

    def _assert_active(self) -> None:
        if self._current_run is None:
            raise RuntimeError(
                "No active run. Call start_run() before log_metrics() / end_run()."
            )


# ── Standalone query functions ────────────────────────────────────────────────

def list_experiments(experiments_dir: Path | None = None) -> pd.DataFrame:
    """Return summary.csv as a DataFrame sorted by timestamp DESC.

    Returns an empty DataFrame with correct columns if no runs exist yet.
    """
    summary_path = _resolve_experiments_dir(experiments_dir) / "summary.csv"
    if not summary_path.exists():
        return pd.DataFrame(columns=_SUMMARY_COLS)
    df = pd.read_csv(summary_path)
    return df.sort_values("timestamp", ascending=False).reset_index(drop=True)


def rebuild_summary(experiments_dir: Path | None = None) -> pd.DataFrame:
    """Reconstruye summary.csv leyendo todos los run JSONs en runs/.

    Útil cuando se cambia la lógica de extracción de métricas y los runs
    antiguos quedan con valores incorrectos o NaN.
    """
    base = _resolve_experiments_dir(experiments_dir)
    runs_dir = base / "runs"
    summary_path = base / "summary.csv"

    if not runs_dir.exists():
        return pd.DataFrame(columns=_SUMMARY_COLS)

    rows: list[dict] = []
    for json_path in sorted(runs_dir.glob("*.json")):
        with open(json_path, "r", encoding="utf-8") as f:
            run = json.load(f)
        metrics = run.get("metrics", {})
        model_info = run.get("model_info", {})

        row = {col: None for col in _SUMMARY_COLS}
        row.update({
            "run_id": run.get("run_id", json_path.stem),
            "experiment_name": run.get("experiment_name", ""),
            "timestamp": run.get("timestamp", ""),
            "model_name": model_info.get("model_name", ""),
            "duration_seconds": run.get("duration_seconds"),
        })

        _meta_cols = {"run_id", "experiment_name", "timestamp",
                      "model_name", "duration_seconds"}
        for col in _SUMMARY_COLS:
            if col in _meta_cols:
                continue
            value = _extract_standard_metric(metrics, col)
            if value is not None:
                row[col] = value

        rows.append(row)

    df = pd.DataFrame(rows, columns=_SUMMARY_COLS)
    df.to_csv(summary_path, index=False)
    print(f"  [tracker] Summary reconstruido ({len(df)} runs) → {summary_path}")
    return df


def get_best_run(
    metric: str = "f1_weighted",
    experiments_dir: Path | None = None,
) -> dict:
    """Return the run with the highest value for *metric*.

    Raises
    ------
    FileNotFoundError  if summary.csv does not exist.
    ValueError         if *metric* is not a column in summary.csv.
    """
    summary_path = _resolve_experiments_dir(experiments_dir) / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"No experiments found at {summary_path}. Run an experiment first."
        )
    df = pd.read_csv(summary_path)
    if metric not in df.columns:
        raise ValueError(
            f"Metric '{metric}' not found in summary. Available: {list(df.columns)}"
        )
    best_row = df.loc[df[metric].idxmax()]
    return best_row.to_dict()


# ── Standalone entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if "--rebuild" in sys.argv:
        rebuild_summary()

    df = list_experiments()
    if df.empty:
        print("No experiments recorded yet.")
    else:
        print(f"Registered experiments ({len(df)} runs):")
        print(df[["run_id", "model_name", "f1_weighted", "timestamp"]].to_string(index=False))
        try:
            best = get_best_run()
            print(f"\nBest run by f1_weighted: {best['run_id']}  ({best['f1_weighted']:.4f})")
        except Exception as e:
            print(f"Could not compute best run: {e}")
