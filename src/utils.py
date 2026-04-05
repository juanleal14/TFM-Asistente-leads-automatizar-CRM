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
