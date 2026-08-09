"""Pure deterministic aggregates over ledger transactions."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from model.domain.ledger import Txn
from model.services.matching import counterparties_match


def txns_for(
    transactions: Iterable[Txn],
    scenario_id: str,
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
) -> Tuple[Txn, ...]:
    prefix = "TXN-{}-".format(scenario_id)
    return tuple(
        txn
        for txn in transactions
        if txn.txn_id.startswith(prefix)
        and (date_from is None or txn.date >= date_from)
        and (date_to is None or txn.date <= date_to)
    )


def sum_signed(transactions: Iterable[Txn]) -> Decimal:
    return sum((txn.amount or Decimal("0") for txn in transactions), Decimal("0"))


def sum_abs(transactions: Iterable[Txn]) -> Decimal:
    return sum((abs(txn.amount) for txn in transactions if txn.amount is not None), Decimal("0"))


def expenses(transactions: Iterable[Txn]) -> Tuple[Txn, ...]:
    return tuple(txn for txn in transactions if txn.amount is not None and txn.amount < 0)


def sum_to_counterparties(transactions: Iterable[Txn], names: Iterable[str]) -> Decimal:
    parties = tuple(names)
    selected = (
        txn
        for txn in transactions
        if txn.amount is not None
        and txn.amount < 0
        and any(counterparties_match(txn.counterparty, name) for name in parties)
    )
    return sum_abs(selected)


def largest_line(transactions: Iterable[Txn]) -> Optional[Txn]:
    values = [txn for txn in transactions if txn.amount is not None]
    return max(values, key=lambda txn: abs(txn.amount), default=None)


def quarter(transactions: Iterable[Txn], number: int) -> Tuple[Txn, ...]:
    if number not in {1, 2, 3, 4}:
        raise ValueError("quarter must be in 1..4")
    start_month = (number - 1) * 3 + 1
    return tuple(txn for txn in transactions if start_month <= txn.date.month <= start_month + 2)


CAPTION_CATEGORIES = (
    ("маркетинговые расходы", "marketing"),
    ("капитальные затраты", "capex"),
    ("операционные расходы", "operating"),
    ("расходы на оплату труда", "personnel"),
    ("коммунальные услуги", "utilities"),
    ("страховые премии", "insurance"),
    ("процентные расходы", "interest"),
    ("арендные платежи", "rent"),
    ("консультационные услуги", "consulting"),
    ("выручка", "revenue"),
    ("налоги", "taxes"),
)


def category_from_caption(caption: str) -> str:
    """Map a covenant/audit line caption onto a ledger category."""
    text = caption.casefold().strip(" «»\"")
    for marker, category in CAPTION_CATEGORIES:
        if marker in text:
            return category
    return "other"


def category_for(description: str) -> str:
    text = description.casefold()
    if any(
        token in text
        for token in ("sales settlement", "sales —", "sales -", "subscription revenue", "interconnect revenue")
    ):
        return "revenue"
    if "principal repayment" in text:
        return "debt_principal"
    if any(
        token in text
        for token in (
            "facility drawdown",
            "loan proceeds",
            "loan drawdown",
            "term loan drawdown",
            "promissory note proceeds",
            "note proceeds",
        )
    ) or ("drawdown" in text and any(token in text for token in ("syndicated", "tranche", "working-capital", "working capital"))):
        return "financing"
    if "transfer" in text and any(
        token in text for token in ("subsidiar", "group entity", "intra-group", "transfer of")
    ):
        return "subsidiary_transfer"
    if (
        "purchase of " in text
        and any(token in text for token in ("equipment", "rig", "conveyor", "generator", "plant"))
    ) or (
        any(token in text for token in ("equipment", "machinery", "network installation", "backbone"))
        and not any(token in text for token in ("lease", "rental", " rent", "rent ", "interest"))
    ):
        return "capex"
    if "insurance" in text or "premium" in text:
        return "insurance"
    if "telecom" in text:
        return "other"
    if any(token in text for token in ("payroll", "staff", "personnel", "crew")):
        return "personnel"
    if any(
        token in text
        for token in (
            "electricity",
            "water supply",
            "water utilit",
            "waste water",
            "natural gas utility",
            "heating utility",
            "utility charge",
            "utilities supply",
            "compressed air utility",
        )
    ):
        return "utilities"
    if any(token in text for token in ("tax", "duty", "levy")):
        return "taxes"
    if any(token in text for token in ("interest", "coupon")):
        return "interest"
    if any(token in text for token in (" rent", "rent ", "lease", "rental")):
        return "rent"
    if "operating" in text or "servicing" in text or "maintenance expenses" in text:
        return "operating"
    if any(
        token in text
        for token in (
            "management advisory",
            "advisory engagement",
            "advisory retainer",
            "advisory settlement",
            "strategy consulting",
            "consulting engagement",
            "consulting fees",
            "consulting retainer",
            "management retainer",
        )
    ):
        return "consulting"
    if any(
        token in text
        for token in (
            "marketing",
            "ad campaign",
            "advertising",
            "media buy",
            "billboard",
            "outdoor media",
            "exhibition stand",
            "sponsorship",
        )
    ):
        return "marketing"
    return "other"


def by_category(
    transactions: Iterable[Txn],
    overrides: Optional[Mapping[str, str]] = None,
) -> Dict[str, Tuple[Txn, ...]]:
    grouped: Dict[str, list] = {}
    overrides = overrides or {}
    for txn in transactions:
        category = overrides.get(txn.txn_id) or category_for(txn.description)
        grouped.setdefault(category, []).append(txn)
    return {name: tuple(rows) for name, rows in grouped.items()}


def aggregate_metrics(
    transactions: Sequence[Txn],
    related_parties: Sequence[str],
    overrides: Optional[Mapping[str, str]] = None,
    excluded_txn_ids: Sequence[str] = (),
) -> Dict[str, Decimal]:
    excluded = set(excluded_txn_ids)
    included = tuple(txn for txn in transactions if txn.txn_id not in excluded)
    grouped = by_category(included, overrides)
    result: Dict[str, Decimal] = {}
    for category, rows in grouped.items():
        result[category] = sum_abs(expenses(rows))
    result["revenue"] = sum(
        (txn.amount or Decimal("0") for txn in grouped.get("revenue", ()) if (txn.amount or 0) > 0),
        Decimal("0"),
    )
    result["financing"] = sum(
        (txn.amount or Decimal("0") for txn in grouped.get("financing", ()) if (txn.amount or 0) > 0),
        Decimal("0"),
    )
    result["related_party_payments"] = sum_to_counterparties(included, related_parties)
    result["missing_amount_count"] = Decimal(
        sum(1 for txn in included if txn.amount is None)
    )
    return result


def swing_analysis(
    transactions: Sequence[Txn],
    calculate: Callable[[Sequence[Txn]], Decimal],
    verdict: Callable[[Decimal], str],
) -> Optional[Txn]:
    original = verdict(calculate(transactions))
    swings = []
    for index, txn in enumerate(transactions):
        without = transactions[:index] + transactions[index + 1 :]
        if verdict(calculate(without)) != original:
            swings.append(txn)
    return swings[0] if len(swings) == 1 else None
