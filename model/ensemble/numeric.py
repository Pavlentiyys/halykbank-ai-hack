"""Deterministic formula evaluation over semantic ledger categories."""

from __future__ import annotations

import re
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional, Sequence, Tuple

from model.config import EnsembleSettings
from model.domain.ledger import Txn
from model.domain.rules import Direction, compare
from model.domain.types import BorrowerContext, CovenantTask, Status
from model.ensemble.estimate import CovenantEstimate
from model.services.matching import counterparties_match
from model.services.metrics import by_category, expenses, quarter, sum_abs, sum_to_counterparties


def _money(value: str) -> Decimal:
    return Decimal(value.replace(",", "").replace(" ", ""))


def extract_threshold(text: str) -> Tuple[Optional[Decimal], Optional[Direction]]:
    compact = " ".join(text.split())
    patterns = (
        (r"не (?:должн\w+ )?превыш\w*(?:\s+величин[уы])?\s*\$?([0-9][0-9, ]*(?:\.[0-9]+)?)(?:x)?", "at_most"),
        (r"не более\s*\$?([0-9][0-9, ]*(?:\.[0-9]+)?)(?:x)?", "at_most"),
        (r"ниже (?:величины\s*)?\$?([0-9][0-9, ]*(?:\.[0-9]+)?)(?:x)?", "at_least"),
        (r"не менее\s*\$?([0-9][0-9, ]*(?:\.[0-9]+)?)(?:x)?", "at_least"),
        (r"не ниже\s*\$?([0-9][0-9, ]*(?:\.[0-9]+)?)(?:x)?", "at_least"),
        (r"составлял[аио]?\s+не менее\s*\$?([0-9][0-9, ]*(?:\.[0-9]+)?)(?:x)?", "at_least"),
    )
    for pattern, direction in patterns:
        match = re.search(pattern, compact, flags=re.IGNORECASE)
        if match:
            return _money(match.group(1)), direction  # type: ignore[return-value]
    ratio = re.search(r"([0-9]+(?:\.[0-9]+)?)x", compact, flags=re.IGNORECASE)
    if ratio:
        direction: Direction = (
            "at_least"
            if any(token in compact.casefold() for token in ("минимальн", "не менее", "не ниже", "покрыт"))
            else "at_most"
        )
        return Decimal(ratio.group(1)), direction
    monetary = re.search(r"\$([0-9][0-9,]*(?:\.[0-9]+)?)", compact)
    if monetary:
        direction = (
            "at_least"
            if any(token in compact.casefold() for token in ("минимальн", "не менее", "не ниже"))
            else "at_most"
        )
        return _money(monetary.group(1)), direction
    return None, None


def _amount(grouped: dict, category: str) -> Decimal:
    return sum_abs(expenses(grouped.get(category, ())))


def _safe_ratio(numerator: Decimal, denominator: Decimal) -> Optional[Decimal]:
    return numerator / denominator if denominator else None


def _reported_value(value: Decimal) -> Decimal:
    """Return the two-decimal value that is actually submitted and assessed."""
    return abs(value).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _unique_rows(rows: Sequence[Txn]) -> Tuple[Txn, ...]:
    return tuple({txn.txn_id: txn for txn in rows}.values())


def _related_rows(transactions: Sequence[Txn], parties: Sequence[str]) -> Tuple[Txn, ...]:
    return tuple(
        txn
        for txn in transactions
        if txn.amount is not None
        and txn.amount < 0
        and any(counterparties_match(txn.counterparty, party) for party in parties)
    )


def _status_for_reported_precision(
    actual: Decimal,
    threshold: Decimal,
    direction: Direction,
    selected: Sequence[Txn],
    ctx: BorrowerContext,
) -> Status:
    raw_status = compare(abs(actual), threshold, direction)
    reported_status = compare(_reported_value(actual), threshold, direction)
    if raw_status == reported_status:
        return raw_status

    # A document-backed transaction can determine a breach hidden by the
    # mandatory two-decimal representation (for example 0.043 -> 0.04).
    mentioned_ids = {
        txn.txn_id
        for txn in selected
        if any(txn.txn_id in page.text for page in ctx.pages)
    }
    return raw_status if mentioned_ids else reported_status


def _evaluate(
    task: CovenantTask,
    ctx: BorrowerContext,
    transactions: Sequence[Txn],
) -> Tuple[Optional[Decimal], Tuple[Txn, ...], str]:
    excluded = set(ctx.excluded_txn_ids)
    included = tuple(txn for txn in transactions if txn.txn_id not in excluded)
    grouped = by_category(included, ctx.category_overrides)
    text = task.text.casefold()
    revenue_rows = tuple(grouped.get("revenue", ()))
    revenue = sum(
        (txn.amount or Decimal("0") for txn in revenue_rows if (txn.amount or 0) > 0),
        Decimal("0"),
    )
    financing_rows = tuple(grouped.get("financing", ()))
    financing = sum(
        (txn.amount or Decimal("0") for txn in financing_rows if (txn.amount or 0) > 0),
        Decimal("0"),
    )
    capex = _amount(grouped, "capex")
    personnel = _amount(grouped, "personnel")
    utilities = _amount(grouped, "utilities")
    taxes = _amount(grouped, "taxes")
    insurance = _amount(grouped, "insurance")
    interest = _amount(grouped, "interest")
    rent = _amount(grouped, "rent")
    operating = _amount(grouped, "operating")
    related = sum_to_counterparties(included, ctx.related_parties)
    ebitda = revenue - operating

    if "четвёрт" in text and "выруч" in text:
        rows = quarter(revenue_rows, 4)
        return sum((txn.amount or Decimal("0") for txn in rows), Decimal("0")), rows, "q4 revenue"
    if "выручка за вычетом наибольш" in text:
        return revenue - max(personnel, taxes), revenue_rows + grouped.get("personnel", ()) + grouped.get("taxes", ()), "revenue less largest overhead"
    if "individual overhead line ceiling" in text:
        category = "personnel" if personnel >= utilities else "utilities"
        return max(personnel, utilities), tuple(grouped.get(category, ())), "largest overhead line"
    if "коэффициент покрытия процентов" in text:
        return _safe_ratio(ebitda, interest), tuple(revenue_rows + grouped.get("operating", ()) + grouped.get("interest", ())), "EBITDA / interest"
    if "cover of applications by sources" in text:
        value = _safe_ratio(revenue + financing, operating + capex)
        rows = revenue_rows + financing_rows + grouped.get("operating", ()) + grouped.get("capex", ())
        return value, tuple(rows), "(revenue + financing) / (operating + capex)"
    if "springing drawdown leverage" in text:
        rows = financing_rows + revenue_rows + grouped.get("operating", ())
        return _safe_ratio(financing, ebitda), tuple(rows), "financing / EBITDA"
    if "капиталоёмк" in text or "capital intensity" in text:
        return _safe_ratio(capex, operating + rent), tuple(grouped.get("capex", ()) + grouped.get("operating", ()) + grouped.get("rent", ())), "capex / (operating + rent)"
    if "скорректированная рентабельность" in text:
        adjusted = ebitda + ctx.metrics.get("one_time_addback", Decimal("0"))
        return _safe_ratio(adjusted, revenue), tuple(revenue_rows + grouped.get("operating", ())), "adjusted EBITDA / revenue"
    if "капитальных затрат группы" in text:
        group_capex = ctx.metrics.get("group_capex")
        return (_safe_ratio(group_capex, ebitda) if group_capex is not None else None), tuple(revenue_rows + grouped.get("operating", ())), "group capex / borrower EBITDA"
    if "покрытие расходов на персонал" in text:
        rows = revenue_rows + grouped.get("personnel", ()) + grouped.get("utilities", ())
        return _safe_ratio(revenue, personnel + utilities), tuple(rows), "revenue / (personnel + utilities)"
    if "налоговой и коммунальной нагрузки" in text:
        rows = grouped.get("taxes", ()) + grouped.get("utilities", ()) + revenue_rows + grouped.get("operating", ())
        return _safe_ratio(taxes + utilities, ebitda), tuple(rows), "(taxes + utilities) / EBITDA"
    if "обязательства по персоналу" in text:
        obligation = ctx.metrics.get("severance_obligation", Decimal("0"))
        return personnel + obligation, tuple(grouped.get("personnel", ())), "personnel costs + disclosed obligation"
    if "неограниченн" in text and "дочерн" in text:
        transfer_rows = tuple(grouped.get("subsidiary_transfer", ()))
        unrestricted_rows = tuple(
            txn for txn in transfer_rows if "holding" in txn.counterparty.casefold()
        )
        unrestricted = sum_abs(expenses(unrestricted_rows))
        total_capex = capex + sum_abs(expenses(transfer_rows))
        return _safe_ratio(unrestricted, total_capex), tuple(transfer_rows + grouped.get("capex", ())), "unrestricted subsidiary assets / total capex"
    if "страховое покрытие расходов" in text:
        rows = grouped.get("insurance", ()) + grouped.get("rent", ()) + grouped.get("utilities", ())
        return _safe_ratio(insurance, rent + utilities), tuple(rows), "insurance / (rent + utilities)"
    if "proportion of revenue" in text:
        related_rows = _related_rows(included, tuple(ctx.related_parties))
        return _safe_ratio(related, revenue), _unique_rows(revenue_rows + related_rows), "related payments / revenue"
    if "доля платежей связанным" in text and "операцион" in text:
        related_rows = _related_rows(included, tuple(ctx.related_parties))
        rows = related_rows + grouped.get("operating", ())
        return _safe_ratio(related, operating), _unique_rows(rows), "related payments / operating costs"
    if "платеж" in text and ("связан" in text or "аффилирован" in text):
        selected = tuple(
            txn
            for txn in included
            if txn.amount is not None
            and txn.amount < 0
            and sum_to_counterparties((txn,), ctx.related_parties) > 0
        )
        return related, selected, "related-party payments"
    if "капитальн" in text and "затрат" in text and ("максим" in text or "не превыш" in text):
        return capex, tuple(grouped.get("capex", ())), "capital expenditure"
    if "минимальн" in text and "выруч" in text and "покрыт" not in text:
        return revenue, revenue_rows, "audited revenue"
    return None, (), "unsupported formula"


class NumericEstimator:
    name = "numeric"

    def __init__(self, settings: EnsembleSettings) -> None:
        self._settings = settings

    def estimate(self, task: CovenantTask, ctx: BorrowerContext) -> CovenantEstimate:
        threshold, direction = extract_threshold(task.text)
        actual, selected, notes = _evaluate(task, ctx, ctx.transactions)
        if actual is None or threshold is None or direction is None:
            return CovenantEstimate(source=self.name, confidence=0.0, notes=notes)

        status = _status_for_reported_precision(
            actual,
            threshold,
            direction,
            selected,
            ctx,
        )
        trigger = re.search(
            r"только при условии,? что .*?превышают\s*\$([0-9][0-9,]*(?:\.[0-9]+)?)",
            " ".join(task.text.split()),
            flags=re.IGNORECASE,
        )
        if trigger and ctx.metrics.get("financing", Decimal("0")) <= _money(trigger.group(1)):
            status = "COMPLIANT"

        swing = None
        if status == "BREACH":
            candidates = []
            for txn in selected:
                reduced = tuple(row for row in ctx.transactions if row.txn_id != txn.txn_id)
                reduced_actual, _, _ = _evaluate(task, ctx, reduced)
                if (
                    reduced_actual is not None
                    and compare(_reported_value(reduced_actual), threshold, direction) != status
                ):
                    candidates.append(txn)

            # An accepted audit reclassification is evidence even when removing a
            # larger ordinary line would also move the aggregate across the limit.
            reclassified = [
                txn for txn in candidates if txn.txn_id in ctx.category_overrides
            ]
            related_candidates = [
                txn
                for txn in candidates
                if any(
                    counterparties_match(txn.counterparty, party)
                    for party in ctx.related_parties
                )
            ]
            if len(reclassified) == 1:
                swing = reclassified[0]
            elif (
                ("связан" in task.text.casefold() or "related-party" in task.text.casefold())
                and len(related_candidates) == 1
            ):
                swing = related_candidates[0]
            elif len(candidates) == 1:
                swing = candidates[0]
        confidence = 0.95 if selected or notes.endswith("obligation") else 0.82
        return CovenantEstimate(
            source=self.name,
            status=status,
            actual=abs(actual),
            threshold=threshold,
            evidence_txn_id=swing.txn_id if swing else None,
            confidence=confidence,
            notes=notes,
        )
