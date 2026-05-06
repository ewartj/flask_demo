"""
Central config loader.

Reads config.yaml from the project root.  Environment variables always take
precedence over file values — use them for secrets and per-environment overrides.
"""
from __future__ import annotations
import os
from pathlib import Path
import yaml

_ROOT = Path(__file__).parent.parent
_CONFIG_PATH = _ROOT / "config.yaml"

_cfg: dict | None = None


def _load() -> dict:
    global _cfg
    if _cfg is None:
        with open(_CONFIG_PATH) as f:
            _cfg = yaml.safe_load(f) or {}
    return _cfg


def get(section: str, key: str, default=None):
    return _load().get(section, {}).get(key, default)


# ── Typed accessors ───────────────────────────────────────────────────────────

def mlflow_config() -> dict:
    cfg = _load().get("mlflow", {})
    return {
        "tracking_uri":         os.environ.get("MLFLOW_TRACKING_URI")           or cfg.get("tracking_uri", ""),
        "sentiment_model_uri":  os.environ.get("MLFLOW_SENTIMENT_MODEL_URI")    or cfg.get("sentiment_model_uri", "models:/Sentiment/Production"),
        "theme_model_uri":      os.environ.get("MLFLOW_THEME_MODEL_URI")        or cfg.get("theme_model_uri", "models:/Theme/Production"),
        "tracking_token":       os.environ.get("MLFLOW_TRACKING_TOKEN", ""),
    }


def sentiment_labels() -> dict[int, str]:
    raw = _load().get("labels", {}).get("sentiment", {})
    return {int(k): v for k, v in raw.items()}


def theme_labels() -> dict[int, str]:
    raw = _load().get("labels", {}).get("theme", {})
    return {int(k): v for k, v in raw.items()}


def theme_candidates() -> list[str]:
    return _load().get("theme_labels", [])


def server_config() -> dict:
    cfg = _load().get("server", {})
    return {
        "port":    int(os.environ.get("APP_PORT", cfg.get("port", 5005))),
        "workers": int(cfg.get("workers", 1)),
        "threads": int(cfg.get("threads", 4)),
        "timeout": int(cfg.get("timeout", 120)),
    }
