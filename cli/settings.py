"""Outer configuration adapter: environment and arguments into typed settings."""

from __future__ import annotations

import os
from argparse import Namespace
from decimal import Decimal
from pathlib import Path
from typing import Dict

from model import EnsembleSettings, GemmaSettings, Settings

TEAM_NAME = "Astrea"
CONTACT_EMAIL = "pashpichug@mail.ru"


def _read_env_file(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def load_settings(args: Namespace) -> Settings:
    file_values = _read_env_file(args.env_file)

    def value(key: str, default: str) -> str:
        return os.environ.get(key, file_values.get(key, default))

    model_id = args.gemma_model or value("GEMMA_MODEL_ID", "gemma4:e2b")
    endpoint = args.gemma_endpoint or value("GEMMA_ENDPOINT", "http://localhost:11434")
    llm_mode = args.llm_mode or value("LLM_MODE", "gaps-only")
    if llm_mode not in {"gaps-only", "always"}:
        raise ValueError("LLM_MODE must be gaps-only or always")
    workers = args.workers or int(value("MAX_WORKERS", "1"))
    fx = Decimal(args.fx_eur_usd or value("FX_EUR_USD", "1.00"))
    team = args.team or TEAM_NAME
    email = args.contact_email or CONTACT_EMAIL
    enabled = not args.offline
    gemma = GemmaSettings(model_id=model_id, endpoint=endpoint, enabled=enabled)
    ensemble = EnsembleSettings(gemma=gemma, fx_eur_usd=fx, llm_mode=llm_mode)
    return Settings(
        ensemble=ensemble,
        max_workers=workers,
        cache_dir=None if args.no_cache else Path(value("LLM_CACHE_DIR", ".llm_cache")),
        team=team,
        contact_email=email,
        model_name="{} + numeric".format(model_id) if enabled else "offline numeric",
    )
