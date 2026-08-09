"""Deterministic formula evaluation over semantic ledger categories."""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional, Sequence, Tuple

from model.config import EnsembleSettings
from model.domain.ledger import Txn
from model.domain.rules import Direction, compare
from model.domain.types import BorrowerContext, CovenantTask, Status
from model.ensemble.estimate import CovenantEstimate
from model.services.matching import counterparties_match
from model.services.metrics import (
    by_category,
    category_from_caption,
    expenses,
    quarter,
    sum_abs,
    sum_to_counterparties,
)


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


@dataclass(frozen=True)
class FormulaResult:
    """A formula outcome, optionally carrying its own limit and verdict."""

    value: Optional[Decimal]
    rows: Tuple[Txn, ...] = ()
    note: str = ""
    threshold: Optional[Decimal] = None
    direction: Optional[Direction] = None
    forced: Optional[Status] = None


def _amount(grouped: dict, category: str) -> Decimal:
    return sum_abs(expenses(grouped.get(category, ())))


def _income(grouped: dict, category: str) -> Decimal:
    return sum(
        (txn.amount or Decimal("0") for txn in grouped.get(category, ()) if (txn.amount or 0) > 0),
        Decimal("0"),
    )


def _limits(text: str) -> Tuple[Decimal, ...]:
    """Every monetary or ratio limit mentioned in the clause, in order."""
    compact = " ".join(text.split())
    found = []
    for match in re.finditer(r"\$([0-9][0-9,]*(?:\.[0-9]+)?)|([0-9]+\.[0-9]+)x", compact):
        raw = match.group(1) or match.group(2)
        found.append(_money(raw))
    return tuple(found)


def _percent(text: str) -> Optional[Decimal]:
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*%", " ".join(text.split()))
    return Decimal(match.group(1)) / Decimal("100") if match else None


def _caption_category(text: str) -> str:
    match = re.search(r"«([^»]{3,60})»", text)
    return category_from_caption(match.group(1)) if match else "other"


def _quarterly_max(rows: Sequence[Txn]) -> Tuple[Decimal, Tuple[Txn, ...]]:
    best_value = Decimal("0")
    best_rows: Tuple[Txn, ...] = ()
    for number in (1, 2, 3, 4):
        selected = quarter(rows, number)
        value = sum_abs(expenses(selected)) or _income({"q": selected}, "q")
        if value > best_value:
            best_value, best_rows = value, tuple(selected)
    return best_value, best_rows


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


def _leverage(
    grouped: dict,
    ctx: BorrowerContext,
    *,
    addback_cap: Optional[Decimal] = None,
    plus_debt: Decimal = Decimal("0"),
    less_debt: Decimal = Decimal("0"),
) -> Tuple[Optional[Decimal], Tuple[Txn, ...]]:
    revenue = _income(grouped, "revenue")
    operating = _amount(grouped, "operating")
    addback = ctx.metrics.get("one_time_addback", Decimal("0"))
    if addback_cap is not None:
        addback = min(addback, addback_cap)
    ebitda = revenue - operating + addback
    debt = _income(grouped, "financing") + plus_debt - less_debt
    rows = (
        tuple(grouped.get("financing", ()))
        + tuple(grouped.get("revenue", ()))
        + tuple(grouped.get("operating", ()))
    )
    return _safe_ratio(debt, ebitda), rows


def _evaluate_gated(
    task: CovenantTask,
    ctx: BorrowerContext,
    grouped: dict,
) -> Optional[FormulaResult]:
    """Springing and dual-condition covenants: the gate and the reported metric differ."""
    text = task.text.casefold()
    limits = _limits(task.text)
    revenue = _income(grouped, "revenue")
    operating = _amount(grouped, "operating")
    ebitda = revenue - operating
    capex = _amount(grouped, "capex")
    rent = _amount(grouped, "rent")
    insurance = _amount(grouped, "insurance")
    interest = _amount(grouped, "interest")
    personnel = _amount(grouped, "personnel")
    principal = _amount(grouped, "debt_principal")

    if "двойное условие дефолта" in text:
        leverage, _ = _leverage(grouped, ctx)
        gate_ratio = next((value for value in limits if value < 100), Decimal("3.50"))
        cap = next((value for value in limits if value >= 100), Decimal("2000000"))
        breached = leverage is not None and leverage > gate_ratio and capex > cap
        return FormulaResult(
            capex,
            tuple(grouped.get("capex", ())),
            "dual default test: reported capex",
            threshold=cap,
            direction="at_most",
            forced="BREACH" if breached else "COMPLIANT",
        )

    if "двойной поддерживающий тест" in text:
        leverage, rows = _leverage(grouped, ctx)
        cover = _safe_ratio(ebitda, interest)
        gate_ratio = limits[0] if limits else Decimal("3.25")
        cover_floor = limits[1] if len(limits) > 1 else Decimal("2.00")
        breached = (
            leverage is not None
            and cover is not None
            and leverage > gate_ratio
            and cover < cover_floor
        )
        return FormulaResult(
            leverage,
            rows,
            "dual maintenance test: reported leverage",
            threshold=gate_ratio,
            direction="at_most",
            forced="BREACH" if breached else "COMPLIANT",
        )

    if "условие досрочного погашения" in text:
        leverage, _ = _leverage(grouped, ctx)
        dscr = _safe_ratio(ebitda, interest + principal)
        gate_ratio = limits[0] if limits else Decimal("3.00")
        dscr_floor = limits[1] if len(limits) > 1 else Decimal("1.30")
        triggered = (leverage is not None and leverage > gate_ratio) or (
            dscr is not None and dscr < dscr_floor
        )
        rows = (
            tuple(grouped.get("interest", ()))
            + tuple(grouped.get("debt_principal", ()))
            + tuple(grouped.get("revenue", ()))
        )
        return FormulaResult(
            dscr,
            rows,
            "cash sweep condition: reported DSCR",
            threshold=dscr_floor,
            direction="at_least",
            forced="BREACH" if triggered else "COMPLIANT",
        )

    if "transfers to unrestricted subsidiaries" in text:
        transfers = tuple(grouped.get("subsidiary_transfer", ()))
        value = sum_abs(expenses(transfers))
        cap = next((limit for limit in limits if limit >= 100), Decimal("500000"))
        return FormulaResult(
            value,
            transfers,
            "springing transfer limit",
            threshold=cap,
            direction="at_most",
        )

    if "insurance cover linked to capital expenditure" in text:
        gate = next((limit for limit in limits if limit >= 1000000), Decimal("1500000"))
        floor = next((limit for limit in limits if limit < 1000000), Decimal("250000"))
        rows = tuple(grouped.get("insurance", ())) + tuple(grouped.get("capex", ()))
        return FormulaResult(
            insurance,
            rows,
            "insurance floor gated on capex",
            threshold=floor,
            direction="at_least",
            forced=None if capex > gate else "COMPLIANT",
        )

    if "property rental cap with insurance proviso" in text:
        cap = next((limit for limit in limits if limit >= 1000000), Decimal("1000000"))
        proviso = next((limit for limit in limits if limit < 1000000), Decimal("200000"))
        rows = tuple(grouped.get("rent", ())) + tuple(grouped.get("insurance", ()))
        return FormulaResult(
            rent,
            rows,
            "rental cap with insurance proviso",
            threshold=cap,
            direction="at_most",
            forced="COMPLIANT" if insurance >= proviso else None,
        )

    if "springing property rental cap" in text:
        share = _percent(task.text) or Decimal("0.30")
        cap = next((limit for limit in limits if limit >= 1000), Decimal("900000"))
        gate_on = revenue > 0 and personnel > share * revenue
        rows = tuple(grouped.get("rent", ())) + tuple(grouped.get("personnel", ()))
        return FormulaResult(
            rent,
            rows,
            "springing rental cap",
            threshold=cap,
            direction="at_most",
            forced=None if gate_on else "COMPLIANT",
        )

    if "с ограничением корректировок ebitda" in text:
        share = _percent(task.text) or Decimal("0.05")
        value, rows = _leverage(grouped, ctx, addback_cap=share * revenue)
        ratio = next((limit for limit in limits if limit < 100), Decimal("3.00"))
        return FormulaResult(value, rows, "leverage with capped addbacks", threshold=ratio, direction="at_most")

    if "с учётом поручительств" in text:
        value, rows = _leverage(
            grouped, ctx, plus_debt=ctx.metrics.get("guarantee_obligations", Decimal("0"))
        )
        return FormulaResult(value, rows, "leverage including guarantees")

    if "maximum net debt leverage" in text:
        value, rows = _leverage(grouped, ctx, less_debt=principal)
        return FormulaResult(value, rows + tuple(grouped.get("debt_principal", ())), "net debt leverage")

    return None


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
    marketing = _amount(grouped, "marketing")
    consulting = _amount(grouped, "consulting")
    principal = _amount(grouped, "debt_principal")

    if "коэффициент покрытия обслуживания долга" in text:
        rows = revenue_rows + grouped.get("operating", ()) + grouped.get("interest", ()) + grouped.get("debt_principal", ())
        return _safe_ratio(ebitda, interest + principal), tuple(rows), "EBITDA / (interest + principal)"
    if "коэффициент покрытия постоянных платежей" in text:
        rows = revenue_rows + grouped.get("operating", ()) + grouped.get("interest", ()) + grouped.get("rent", ())
        return _safe_ratio(ebitda + rent, interest + rent), tuple(rows), "fixed charge cover"
    if "minimum fixed overhead cover" in text:
        rows = revenue_rows + grouped.get("rent", ()) + grouped.get("utilities", ()) + grouped.get("insurance", ())
        return _safe_ratio(ebitda, rent + utilities + insurance), tuple(rows), "EBITDA / fixed overhead"
    if "post-personnel operating margin" in text:
        rows = revenue_rows + grouped.get("personnel", ()) + grouped.get("taxes", ())
        return _safe_ratio(revenue - personnel - taxes, revenue), tuple(rows), "post-personnel margin"
    if "fiscal burden ratio" in text:
        rows = grouped.get("taxes", ()) + grouped.get("interest", ()) + revenue_rows
        return _safe_ratio(taxes + interest, revenue), tuple(rows), "(taxes + interest) / revenue"
    if "quarterly revenue concentration" in text:
        peak, peak_rows = _quarterly_max(revenue_rows)
        return _safe_ratio(peak, revenue), peak_rows, "peak quarter revenue share"
    if "консультационных услуг к ebitda" in text:
        rows = grouped.get("consulting", ()) + revenue_rows + grouped.get("operating", ())
        return _safe_ratio(consulting, ebitda), tuple(rows), "consulting / EBITDA"
    if "minimum retained financing proceeds" in text:
        rows = financing_rows + grouped.get("interest", ()) + grouped.get("taxes", ())
        return financing - interest - taxes, tuple(rows), "retained financing proceeds"
    if "вклад в ликвидность" in text or "minimum liquidity contribution" in text:
        rows = revenue_rows + grouped.get("operating", ()) + financing_rows
        return ebitda + financing, tuple(rows), "EBITDA + financing"
    if "запас покрытия постоянных расходов" in text:
        rows = revenue_rows + grouped.get("operating", ()) + grouped.get("personnel", ()) + grouped.get("rent", ())
        return revenue - (operating + personnel + rent), tuple(rows), "fixed cost buffer"
    if "совокупные расходы на содержание помещений" in text:
        rows = grouped.get("rent", ()) + grouped.get("utilities", ()) + grouped.get("insurance", ())
        return rent + utilities + insurance, tuple(rows), "occupancy costs"
    if "разрешённой долговой корзины" in text:
        undisclosed = ctx.metrics.get("undisclosed_debt", Decimal("0"))
        return financing + undisclosed, financing_rows, "permitted debt basket"
    if "капитальных затрат на уровне группы" in text:
        group_capex = ctx.metrics.get("group_capex")
        return group_capex, tuple(grouped.get("capex", ())), "group capital expenditure"
    if "в любом отдельном финансовом квартале" in text and "расход" in text:
        category = _caption_category(task.text) or "marketing"
        peak, peak_rows = _quarterly_max(grouped.get(category, ()))
        return peak, peak_rows, "largest quarterly {} spend".format(category)
    if "расходы по категории" in text:
        category = _caption_category(task.text)
        if category != "other":
            rows = tuple(grouped.get(category, ()))
            return sum_abs(expenses(rows)), rows, "{} spend".format(category)
    if "коэффициент долговой нагрузки" in text or "предельная долговая нагрузка" in text:
        value, rows = _leverage(grouped, ctx)
        return value, rows, "financing / EBITDA"

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


def evaluate(
    task: CovenantTask,
    ctx: BorrowerContext,
    transactions: Sequence[Txn],
) -> FormulaResult:
    """Single entry point: gated covenants first, then the plain formula bank."""
    excluded = set(ctx.excluded_txn_ids)
    included = tuple(txn for txn in transactions if txn.txn_id not in excluded)
    gated = _evaluate_gated(task, ctx, by_category(included, ctx.category_overrides))
    if gated is not None:
        return gated
    value, rows, note = _evaluate(task, ctx, transactions)
    return FormulaResult(value=value, rows=rows, note=note)


class NumericEstimator:
    name = "numeric"

    def __init__(self, settings: EnsembleSettings) -> None:
        self._settings = settings

    def estimate(self, task: CovenantTask, ctx: BorrowerContext) -> CovenantEstimate:
        result = evaluate(task, ctx, ctx.transactions)
        actual, selected, notes = result.value, result.rows, result.note
        extracted_threshold, extracted_direction = extract_threshold(task.text)
        threshold = result.threshold if result.threshold is not None else extracted_threshold
        direction = result.direction if result.direction is not None else extracted_direction
        if actual is None or threshold is None or direction is None:
            return CovenantEstimate(source=self.name, confidence=0.0, notes=notes)

        status = result.forced or _status_for_reported_precision(
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
        if (
            result.forced is None
            and trigger
            and ctx.metrics.get("financing", Decimal("0")) <= _money(trigger.group(1))
        ):
            status = "COMPLIANT"

        swing = None
        if status == "BREACH":
            candidates = []
            for txn in selected:
                reduced = tuple(row for row in ctx.transactions if row.txn_id != txn.txn_id)
                reduced_actual = evaluate(task, ctx, reduced).value
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
