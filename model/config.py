"""Typed configuration supplied by an outer adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Literal, Optional


@dataclass(frozen=True)
class GemmaSettings:
    model_id: str = "gemma-3-27b-it"
    endpoint: str = "http://localhost:11434"
    temperature: float = 0.0
    max_output_tokens: int = 512
    request_timeout: int = 180
    keep_alive: str = "30m"
    think: bool = False
    enabled: bool = True
    num_ctx: int = 32768


@dataclass(frozen=True)
class EnsembleSettings:
    gemma: GemmaSettings = field(default_factory=GemmaSettings)
    numeric_enabled: bool = True
    fx_eur_usd: Decimal = Decimal("1.00")
    related_party_threshold: Decimal = Decimal("0.20")
    llm_mode: Literal["gaps-only", "always"] = "gaps-only"


@dataclass(frozen=True)
class Settings:
    ensemble: EnsembleSettings = field(default_factory=EnsembleSettings)
    max_workers: int = 2
    cache_dir: Optional[Path] = Path(".llm_cache")
    max_context_chars: int = 300_000
    ocr_fallback_enabled: bool = True
    ocr_language: str = "rus+eng"
    team: str = "covenant-agent"
    contact_email: str = "team@example.com"
    model_name: str = "gemma-3-27b-it + numeric"
