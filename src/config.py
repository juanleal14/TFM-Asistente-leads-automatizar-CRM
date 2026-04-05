"""Loads config.yaml from the project root and exposes a CONFIG dict."""
import pathlib
import yaml

_PROJECT_ROOT = pathlib.Path(__file__).parents[1]
_CONFIG_PATH = _PROJECT_ROOT / "config.yaml"

with open(_CONFIG_PATH, "r", encoding="utf-8") as _f:
    CONFIG: dict = yaml.safe_load(_f)

# Convenience: resolve relative paths to absolute, anchored at project root
def resolve_path(key: str) -> pathlib.Path:
    """Return an absolute Path for a key inside CONFIG['paths']."""
    return _PROJECT_ROOT / CONFIG["paths"][key]
