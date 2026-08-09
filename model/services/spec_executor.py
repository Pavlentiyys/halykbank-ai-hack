"""Deterministic execution of a declarative covenant specification."""

from __future__ import annotations

from decimal import Decimal
from typing import Dict, Mapping, Sequence, Tuple

from model.domain.formula_spec import FormulaSpec
from model.domain.ledger import Txn
from model.services.matching import counterparties_match
from model.services.metrics import expenses, quarter, sum_abs

INCOME_CATEGORIES = frozenset({"revenue", "financing"})


def vocabulary(grouped: Mapping[str, Sequence[Txn]], metrics: Mapping[str, Decimal]) -> Tuple[str, ...]:
    """Names a specification may refer to, for the current borrower."""
    return tuple(sorted(set(grouped) | set(metrics)))


def _term_value(
    name: str,
    grouped: Mapping[str, Sequence[Txn]],
    metrics: Mapping[str, Decimal],
    scope: str,
) -> Tuple[Decimal, Tuple[Txn, ...]]:
    rows = tuple(grouped.get(name, ()))
    if not rows:
        return metrics.get(name, Decimal("0")), ()

    if scope == "quarter_max":
        best_value, best_rows = Decimal("0"), ()
        for number in (1, 2, 3, 4):
            selected = quarter(rows, number)
            value = _aggregate(name, selected)
            if value > best_value:
                best_value, best_rows = value, tuple(selected)
        return best_value, best_rows
    if scope == "quarter_last":
        selected = quarter(rows, 4)
        return _aggregate(name, selected), tuple(selected)
    return _aggregate(name, rows), rows


def _aggregate(name: str, rows: Sequence[Txn]) -> Decimal:
    if name in INCOME_CATEGORIES:
        return sum(
            (txn.amount or Decimal("0") for txn in rows if (txn.amount or 0) > 0),
            Decimal("0"),
        )
    return sum_abs(expenses(rows))


def _restrict_to_related(
    grouped: Mapping[str, Sequence[Txn]],
    related_parties: Sequence[str],
) -> Dict[str, Tuple[Txn, ...]]:
    return {
        name: tuple(
            txn
            for txn in rows
            if any(counterparties_match(txn.counterparty, party) for party in related_parties)
        )
        for name, rows in grouped.items()
    }


def execute_spec(
    spec: FormulaSpec,
    grouped: Mapping[str, Sequence[Txn]],
    metrics: Mapping[str, Decimal],
    related_parties: Sequence[str] = (),
) -> Tuple[object, Tuple[Txn, ...]]:
    """Return (value, evidence rows). Value is None when the spec cannot be applied."""
    source = _restrict_to_related(grouped, related_parties) if spec.related_only else grouped

    numerator = Decimal("0")
    rows: Tuple[Txn, ...] = ()
    for name in spec.numerator:
        value, term_rows = _term_value(name, source, metrics, spec.scope)
        numerator += value
        rows += term_rows
    for name in spec.numerator_less:
        value, term_rows = _term_value(name, source, metrics, "period")
        numerator -= value
        rows += term_rows

    if not spec.denominator:
        return numerator, rows

    denominator = Decimal("0")
    for name in spec.denominator:
        value, term_rows = _term_value(name, source, metrics, "period")
        denominator += value
        rows += term_rows
    for name in spec.denominator_less:
        value, term_rows = _term_value(name, source, metrics, "period")
        denominator -= value
        rows += term_rows

    if not denominator:
        return None, rows
    return numerator / denominator, rows
