"""Utility helpers shared across modules."""
import json
import pathlib
from typing import Any


def save_json(data: Any, path: str | pathlib.Path) -> None:
    """Serialise *data* to JSON at *path* (creates parent dirs if needed)."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: str | pathlib.Path) -> Any:
    """Load and return JSON from *path*."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def append_csv_row(row: dict, path: str | pathlib.Path) -> None:
    """Append a single row (dict) to a CSV file.

    Creates the file with a header if it does not yet exist.
    Thread-unsafe by design — intended for sequential TFM experiment tracking.
    """
    import pandas as pd

    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    pd.DataFrame([row]).to_csv(path, mode="a", header=not exists, index=False)
