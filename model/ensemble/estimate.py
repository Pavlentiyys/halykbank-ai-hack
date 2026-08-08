"""A field-complete estimate emitted by one ensemble participant."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

from model.domain.types import Status


@dataclass(frozen=True)
class CovenantEstimate:
    source: str
    status: Optional[Status] = None
    actual: Optional[Decimal] = None
    threshold: Optional[Decimal] = None
    evidence_txn_id: Optional[str] = None
    confidence: float = 0.0
    quote: Optional[str] = None
    used_document: Optional[str] = None
    notes: str = ""

