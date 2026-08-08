from datetime import date
from decimal import Decimal

from model.adapters.ledger_csv import to_decimal
from model.domain.ledger import Txn
from model.services.matching import normalize_counterparty
from model.services.metrics import (
    category_for,
    quarter,
    sum_abs,
    sum_to_counterparties,
    swing_analysis,
)


def _txn(identifier: str, amount: str, counterparty: str = "Vendor LLP") -> Txn:
    return Txn(
        txn_id=identifier,
        date=date(2025, 10, 1),
        account_id="ACC-7801",
        counterparty=counterparty,
        description="Management advisory retainer",
        amount=Decimal(amount),
        currency="USD",
    )


def test_empty_amount_is_tolerated() -> None:
    assert to_decimal("") is None
    assert to_decimal(" 1,234.50 ") == Decimal("1234.50")


def test_counterparty_normalization_and_sum() -> None:
    assert normalize_counterparty("Aktau Holdings L.L.P. (Almaty office)") == "aktau holdings"
    rows = (
        _txn("one", "-10", "Aktau Holdings L.L.P. (Almaty office)"),
        _txn("two", "-99", "Aktau Terminal Properties LLP"),
    )
    assert sum_to_counterparties(rows, ("Aktau Holdings LLP",)) == Decimal("10")


def test_categories_do_not_treat_every_positive_as_revenue() -> None:
    assert category_for("Port handling sales settlement 2025") == "revenue"
    assert category_for("VAT refund received") != "revenue"
    assert category_for("Telecom leased line") != "rent"
    assert category_for("Travel insurance for staff") == "insurance"


def test_quarter_and_swing_analysis() -> None:
    rows = (_txn("one", "-60"), _txn("two", "-50"))
    assert quarter(rows, 4) == rows
    assert sum_abs(rows) == Decimal("110")
    swing = swing_analysis(
        rows,
        lambda values: sum_abs(values),
        lambda value: "BREACH" if value > Decimal("100") else "COMPLIANT",
    )
    assert swing is None
    single = swing_analysis(
        (_txn("one", "-110"), _txn("two", "-1")),
        lambda values: sum_abs(values),
        lambda value: "BREACH" if value > Decimal("100") else "COMPLIANT",
    )
    assert single is not None and single.txn_id == "one"

