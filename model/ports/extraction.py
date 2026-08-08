"""Document and ledger extraction ports."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence

from model.domain.ledger import Ledger
from model.domain.types import DocumentRef, PageChunk


class TextExtractor(Protocol):
    def extract(self, source: DocumentRef) -> Sequence[PageChunk]: ...


class LedgerRepository(Protocol):
    def load(self, path: Path) -> Ledger: ...

