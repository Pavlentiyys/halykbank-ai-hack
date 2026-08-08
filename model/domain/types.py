"""Public domain types shared by all invocation adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Dict, Literal, Mapping, Optional, Tuple

from .ledger import Txn

Status = Literal["COMPLIANT", "BREACH"]


@dataclass(frozen=True)
class DatasetRef:
    root: Path

    def __init__(self, root: object = Path(".")) -> None:
        object.__setattr__(self, "root", Path(root))

    def resolve(self, filename: str) -> Path:
        direct = self.root / filename
        if direct.exists():
            return direct
        parent = self.root.parent / filename
        if parent.exists():
            return parent
        return direct


@dataclass(frozen=True)
class DocumentRef:
    path: Path


@dataclass(frozen=True)
class PageChunk:
    doc_name: str
    page: int
    text: str


@dataclass(frozen=True)
class CovenantTask:
    scenario_id: str
    covenant_id: str
    text: str = ""
    period_from: Optional[str] = None
    period_to: Optional[str] = None


@dataclass(frozen=True)
class BorrowerContext:
    scenario_id: str
    account_id: str
    pages: Tuple[PageChunk, ...] = ()
    metrics: Mapping[str, Decimal] = field(default_factory=dict)
    transactions: Tuple[Txn, ...] = ()
    related_parties: Tuple[str, ...] = ()
    covenant_texts: Mapping[str, str] = field(default_factory=dict)
    document_kinds: Mapping[str, str] = field(default_factory=dict)
    excluded_txn_ids: Tuple[str, ...] = ()
    category_overrides: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class CovenantAnswer:
    scenario_id: str
    covenant_id: str
    status: Status
    actual: float
    evidence_txn_id: Optional[str] = None
    threshold: Optional[float] = None
    quote: Optional[str] = None
    used_document: Optional[str] = None
    reasoning: Optional[str] = None
    is_fallback: bool = False
    has_disagreement: bool = False


@dataclass(frozen=True)
class Submission:
    team: str
    contact_email: str
    model: str
    answers: Tuple[CovenantAnswer, ...]

    def to_submission_dict(self) -> Dict[str, object]:
        nested: Dict[str, Dict[str, Dict[str, object]]] = {}
        for answer in self.answers:
            nested.setdefault(answer.scenario_id, {})[answer.covenant_id] = {
                "status": answer.status,
                "actual": round(abs(float(answer.actual)), 2),
                "evidence_txn_id": answer.evidence_txn_id,
            }
        return {
            "team": self.team,
            "contact_email": self.contact_email,
            "model": self.model,
            "answers": nested,
        }

