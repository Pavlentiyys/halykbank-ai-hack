"""Ledger value objects represented with exact decimals."""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal
from typing import Iterable, Optional, Tuple


@dataclass(frozen=True)
class Txn:
    txn_id: str
    date: date
    account_id: str
    counterparty: str
    description: str
    amount: Optional[Decimal]
    currency: str

    def with_amount(self, amount: Decimal) -> "Txn":
        return replace(self, amount=amount)


@dataclass(frozen=True)
class Ledger:
    transactions: Tuple[Txn, ...]

    @classmethod
    def from_iterable(cls, transactions: Iterable[Txn]) -> "Ledger":
        return cls(tuple(transactions))

    def for_account(self, account_id: str) -> Tuple[Txn, ...]:
        return tuple(txn for txn in self.transactions if txn.account_id == account_id)

