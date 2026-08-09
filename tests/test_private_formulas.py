"""Formula families added for the private dataset."""

from datetime import date
from decimal import Decimal

from model.domain.ledger import Txn
from model.domain.types import BorrowerContext, CovenantTask
from model.ensemble.numeric import evaluate
from model.services.metrics import category_for, category_from_caption


def _txn(txn_id: str, amount: str, description: str, counterparty: str = "Acme LLP") -> Txn:
    return Txn(
        txn_id=txn_id,
        date=date(2025, 3, 1),
        account_id="ACC-1",
        counterparty=counterparty,
        description=description,
        amount=Decimal(amount),
        currency="USD",
    )


def _context(transactions, **kwargs) -> BorrowerContext:
    return BorrowerContext(
        scenario_id="T1",
        account_id="ACC-1",
        transactions=tuple(transactions),
        **kwargs,
    )


def _run(text: str, transactions, **kwargs):
    ctx = _context(transactions, **kwargs)
    task = CovenantTask(scenario_id="T1", covenant_id="6.1", text=text)
    return evaluate(task, ctx, ctx.transactions)


LEDGER = [
    _txn("TXN-T1-0001", "10000.00", "Widget sales settlement 2025"),
    _txn("TXN-T1-0002", "-4000.00", "Plant operating costs 2025"),
    _txn("TXN-T1-0003", "9000.00", "Syndicated term loan drawdown tranche A"),
    _txn("TXN-T1-0004", "-1000.00", "Interest on senior facility"),
    _txn("TXN-T1-0005", "-500.00", "Term loan principal repayment 2025"),
    _txn("TXN-T1-0006", "-300.00", "Outdoor media buy campaign"),
]


def test_new_categories_are_recognised() -> None:
    assert category_for("Syndicated term loan drawdown tranche A") == "financing"
    assert category_for("Term loan principal repayment 2025") == "debt_principal"
    assert category_for("Outdoor media buy campaign") == "marketing"
    assert category_for("Strategy consulting engagement 2025") == "consulting"
    assert category_for("Imported clinker grinding equipment") == "capex"
    assert category_for("Wholesale interconnect revenue") == "revenue"


def test_consulting_wins_over_marketing() -> None:
    assert category_for("Brand advisory engagement") == "consulting"


def test_lease_lines_stay_rent_not_capex() -> None:
    assert category_for("Interest on finance sublease") == "interest"
    assert category_for("Warehouse land lease payments 2025") == "rent"


def test_caption_mapping() -> None:
    assert category_from_caption("«Маркетинговые расходы»") == "marketing"
    assert category_from_caption("«Капитальные затраты»") == "capex"


def test_base_leverage_uses_financing_over_ebitda() -> None:
    result = _run("Пункт 6.1 Максимальный коэффициент долговой нагрузки … 3.00x", LEDGER)
    assert result.value == Decimal("9000.00") / Decimal("6000.00")


def test_net_debt_leverage_subtracts_principal_repayments() -> None:
    result = _run("Section 5.2 Maximum Net Debt Leverage … 3.00x", LEDGER)
    assert result.value == Decimal("8500.00") / Decimal("6000.00")


def test_leverage_with_guarantees_adds_disclosed_obligation() -> None:
    result = _run(
        "Пункт 6.1 Максимальная скорректированная долговая нагрузка с учётом поручительств … 3.00x",
        LEDGER,
        metrics={"guarantee_obligations": Decimal("3000.00")},
    )
    assert result.value == Decimal("12000.00") / Decimal("6000.00")


def test_addback_cap_limits_the_ebitda_uplift() -> None:
    result = _run(
        "Пункт 6.1 Долговая нагрузка с ограничением корректировок EBITDA, не превышающих 5% of Revenue … 3.00x",
        LEDGER,
        metrics={"one_time_addback": Decimal("4000.00")},
    )
    assert result.value == Decimal("9000.00") / Decimal("6500.00")
    assert result.threshold == Decimal("3.00")


def test_debt_service_cover_includes_principal() -> None:
    result = _run("Пункт 6.1 Минимальный коэффициент покрытия обслуживания долга … 1.25x", LEDGER)
    assert result.value == Decimal("6000.00") / Decimal("1500.00")


def test_category_cap_reads_the_caption() -> None:
    result = _run(
        "Пункт 6.1 Максимальные расходы по категории. Расходы по статье «Маркетинговые расходы» "
        "не должны превышать $400,000.00",
        LEDGER,
    )
    assert result.value == Decimal("300.00")


def test_springing_cap_stays_compliant_when_the_gate_is_off() -> None:
    ledger = LEDGER + [_txn("TXN-T1-0007", "-8000.00", "Warehouse rent 2025")]
    result = _run(
        "Пункт 6.2 Springing Property Rental Cap. Ограничение применяется, если расходы на персонал "
        "превышают 30.0% выручки; предел составляет $900,000.00",
        ledger,
    )
    assert result.value == Decimal("8000.00")
    assert result.forced == "COMPLIANT"


def test_dual_default_test_reports_capex_not_leverage() -> None:
    ledger = LEDGER + [_txn("TXN-T1-0008", "-1500.00", "Purchase of conveyor equipment")]
    result = _run(
        "Пункт 6.1 Двойное условие дефолта (долговая нагрузка и капзатраты). Нарушение наступает, "
        "если долговая нагрузка превышает 3.50x и капитальные затраты превышают $2,000,000.00",
        ledger,
    )
    assert result.value == Decimal("1500.00")
    assert result.forced == "COMPLIANT"
