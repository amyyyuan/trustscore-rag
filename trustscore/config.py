"""Configuration loading. One YAML file drives every script."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml


def _deep_update(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | os.PathLike, overrides: list[str] | None = None) -> dict[str, Any]:
    """Load a YAML config. A config may name a `base` file to inherit from.

    `overrides` are strings of the form `a.b.c=value` (YAML-parsed values).
    """
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("base", None)
    if base:
        base_cfg = load_config(path.parent / base)
        cfg = _deep_update(base_cfg, cfg)
    for item in overrides or []:
        key, _, raw = item.partition("=")
        value = yaml.safe_load(raw)
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    cfg["_config_path"] = str(path)
    return cfg


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve(cfg: dict, key: str) -> Path:
    """Resolve a path from cfg['paths'] relative to the repo root."""
    return (repo_root() / cfg["paths"][key]).resolve()
