"""CSV ledger adapter with exact decimal parsing and explicit FX conversion."""

from __future__ import annotations

import csv
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Optional

from model.domain.ledger import Ledger, Txn


def to_decimal(value: str) -> Optional[Decimal]:
    cleaned = (value or "").strip().replace(",", "")
    if not cleaned:
        return None
    try:
        return Decimal(cleaned)
    except InvalidOperation as exc:
        raise ValueError("invalid ledger amount: {!r}".format(value)) from exc


class CsvLedgerRepository:
    def __init__(self, fx_eur_usd: Decimal) -> None:
        self._fx_eur_usd = fx_eur_usd

    def load(self, path: Path) -> Ledger:
        transactions = []
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {
                "txn_id",
                "date",
                "account_id",
                "counterparty",
                "description",
                "amount",
                "currency",
            }
            if set(reader.fieldnames or ()) != required:
                raise ValueError("unexpected ledger columns: {}".format(reader.fieldnames))
            for row in reader:
                amount = to_decimal(row["amount"])
                currency = row["currency"].strip().upper()
                if amount is not None and currency == "EUR":
                    amount *= self._fx_eur_usd
                elif currency not in {"USD", "EUR"}:
                    raise ValueError("unsupported currency {}".format(currency))
                transactions.append(
                    Txn(
                        txn_id=row["txn_id"].strip(),
                        date=date.fromisoformat(row["date"].strip()),
                        account_id=row["account_id"].strip(),
                        counterparty=row["counterparty"].strip(),
                        description=row["description"].strip(),
                        amount=amount,
                        currency=currency,
                    )
                )
        return Ledger.from_iterable(transactions)

