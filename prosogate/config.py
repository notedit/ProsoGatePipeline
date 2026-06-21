"""Config loading. Plain YAML + dict-style access; no pydantic dependency."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    """Dict with attribute-style access for nested keys."""

    def __getattr__(self, name: str) -> Any:
        if name in self:
            v = self[name]
            return Config(v) if isinstance(v, dict) else v
        raise AttributeError(name)

    def get_path(self, dotted: str, default: Any = None) -> Any:
        cur: Any = self
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur


def load_config(path: str | Path) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return Config(data)


def config_hash(cfg: Any) -> str:
    """Stable hash of a config sub-tree for stage-level idempotency."""
    blob = json.dumps(cfg, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
