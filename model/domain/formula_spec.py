"""Declarative description of a covenant calculation."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Mapping, Optional, Tuple

from model.domain.rules import Direction

Scope = str

SCOPES = ("period", "quarter_max", "quarter_last")


@dataclass(frozen=True)
class FormulaSpec:
    """What to compute, expressed only in category and metric names."""

    numerator: Tuple[str, ...] = ()
    numerator_less: Tuple[str, ...] = ()
    denominator: Tuple[str, ...] = ()
    denominator_less: Tuple[str, ...] = ()
    threshold: Optional[Decimal] = None
    direction: Optional[Direction] = None
    scope: Scope = "period"
    related_only: bool = False
    note: str = ""

    def is_usable(self) -> bool:
        return bool(self.numerator) and self.threshold is not None and self.direction is not None


def covenant_key(text: str) -> str:
    """Stable identity of a covenant wording, insensitive to formatting and amounts."""
    compact = " ".join(text.split()).casefold()
    skeleton = re.sub(r"[0-9][0-9,.]*", "#", compact)
    return hashlib.sha256(skeleton.encode("utf-8")).hexdigest()[:16]


def _decimal(value: object) -> Optional[Decimal]:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def _names(value: object) -> Tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item) for item in value if str(item).strip())


def spec_from_payload(payload: Mapping[str, Any]) -> FormulaSpec:
    scope = str(payload.get("scope") or "period")
    direction = payload.get("direction")
    return FormulaSpec(
        numerator=_names(payload.get("numerator")),
        numerator_less=_names(payload.get("numerator_less")),
        denominator=_names(payload.get("denominator")),
        denominator_less=_names(payload.get("denominator_less")),
        threshold=_decimal(payload.get("threshold")),
        direction=direction if direction in ("at_least", "at_most") else None,
        scope=scope if scope in SCOPES else "period",
        related_only=bool(payload.get("related_only")),
        note=str(payload.get("note") or ""),
    )


def spec_to_payload(spec: FormulaSpec) -> Dict[str, Any]:
    return {
        "numerator": list(spec.numerator),
        "numerator_less": list(spec.numerator_less),
        "denominator": list(spec.denominator),
        "denominator_less": list(spec.denominator_less),
        "threshold": str(spec.threshold) if spec.threshold is not None else None,
        "direction": spec.direction,
        "scope": spec.scope,
        "related_only": spec.related_only,
        "note": spec.note,
    }


SPEC_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "numerator": {"type": "array", "items": {"type": "string"}},
        "numerator_less": {"type": "array", "items": {"type": "string"}},
        "denominator": {"type": "array", "items": {"type": "string"}},
        "denominator_less": {"type": "array", "items": {"type": "string"}},
        "threshold": {"type": "number"},
        "direction": {"type": "string", "enum": ["at_least", "at_most"]},
        "scope": {"type": "string", "enum": list(SCOPES)},
        "related_only": {"type": "boolean"},
        "note": {"type": "string"},
    },
    "required": ["numerator", "threshold", "direction"],
}
