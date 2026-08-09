"""Declarative covenant specifications: execution, safety guards and the library."""

import json
from datetime import date
from decimal import Decimal

from model.adapters.spec_library import SpecLibrary
from model.domain.formula_spec import FormulaSpec, covenant_key, spec_from_payload
from model.domain.ledger import Txn
from model.domain.types import BorrowerContext, CovenantTask
from model.ensemble.synthesized import SynthesizedFormulaEstimator
from model.ports.inference import LLMResponse
from model.services.metrics import by_category
from model.services.spec_executor import execute_spec

CONFIG = __import__("model.config", fromlist=["EnsembleSettings"]).EnsembleSettings()


def _txn(txn_id: str, amount: str, description: str, counterparty: str = "Acme LLP") -> Txn:
    return Txn(
        txn_id=txn_id,
        date=date(2025, 2, 1),
        account_id="ACC-1",
        counterparty=counterparty,
        description=description,
        amount=Decimal(amount),
        currency="USD",
    )


LEDGER = (
    _txn("TXN-T1-0001", "10000.00", "Widget sales settlement 2025"),
    _txn("TXN-T1-0002", "-4000.00", "Plant operating costs 2025"),
    _txn("TXN-T1-0003", "-1000.00", "Outdoor media buy campaign"),
    _txn("TXN-T1-0004", "-500.00", "Management advisory retainer", "Related Holdings LLP"),
)
GROUPED = by_category(LEDGER)


class StubModel:
    def __init__(self, payload: object) -> None:
        self._payload = payload
        self.calls = 0

    def complete(self, request: object) -> LLMResponse:
        self.calls += 1
        return LLMResponse(text=json.dumps(self._payload))


def _estimate(payload: object, text: str, library=None, ctx: BorrowerContext = None):
    model = StubModel(payload)
    estimator = SynthesizedFormulaEstimator(model, CONFIG, library)
    context = ctx or BorrowerContext(scenario_id="T1", account_id="ACC-1", transactions=LEDGER)
    task = CovenantTask(scenario_id="T1", covenant_id="6.1", text=text)
    return estimator.estimate(task, context), model


def test_absolute_limit_sums_the_named_category() -> None:
    spec = FormulaSpec(
        numerator=("marketing",), threshold=Decimal("400000"), direction="at_most"
    )
    value, rows = execute_spec(spec, GROUPED, {})
    assert value == Decimal("1000.00")
    assert [txn.txn_id for txn in rows] == ["TXN-T1-0003"]


def test_ratio_uses_income_for_revenue_and_expense_for_costs() -> None:
    spec = FormulaSpec(
        numerator=("revenue",),
        numerator_less=("operating",),
        denominator=("revenue",),
        threshold=Decimal("0.5"),
        direction="at_least",
    )
    value, _ = execute_spec(spec, GROUPED, {})
    assert value == Decimal("6000.00") / Decimal("10000.00")


def test_metric_names_fall_back_to_context_metrics() -> None:
    spec = FormulaSpec(
        numerator=("guarantee_obligations",), threshold=Decimal("1"), direction="at_most"
    )
    value, _ = execute_spec(spec, GROUPED, {"guarantee_obligations": Decimal("777.00")})
    assert value == Decimal("777.00")


def test_related_only_restricts_to_related_counterparties() -> None:
    spec = FormulaSpec(
        numerator=("consulting",),
        threshold=Decimal("1"),
        direction="at_most",
        related_only=True,
    )
    value, _ = execute_spec(spec, GROUPED, {}, related_parties=("Related Holdings LLP",))
    assert value == Decimal("500.00")
    empty, _ = execute_spec(spec, GROUPED, {}, related_parties=("Someone Else LLP",))
    assert empty == Decimal("0")


def test_zero_denominator_yields_no_value() -> None:
    spec = FormulaSpec(
        numerator=("marketing",),
        denominator=("subsidiary_transfer",),
        threshold=Decimal("1"),
        direction="at_most",
    )
    value, _ = execute_spec(spec, GROUPED, {})
    assert value is None


def test_specification_drives_a_full_estimate() -> None:
    estimate, _ = _estimate(
        {"numerator": ["marketing"], "threshold": 400, "direction": "at_most"},
        "Максимальные расходы по статье «Маркетинговые расходы» — не более $400.00",
    )
    assert estimate is not None
    assert estimate.actual == Decimal("1000.00")
    assert estimate.status == "BREACH"


def test_unknown_names_are_rejected_rather_than_guessed() -> None:
    estimate, _ = _estimate(
        {"numerator": ["EBITDA"], "threshold": 1, "direction": "at_most"},
        "Лимит расходов $1.00",
    )
    assert estimate is None


def test_ratio_covenant_without_denominator_is_rejected() -> None:
    estimate, _ = _estimate(
        {"numerator": ["marketing"], "threshold": 1, "direction": "at_most"},
        "Максимальное отношение маркетинговых расходов к выручке — 1.00x",
    )
    assert estimate is None


def test_group_level_covenants_are_left_to_the_default() -> None:
    estimate, model = _estimate(
        {"numerator": ["capex"], "threshold": 20000000, "direction": "at_most"},
        "Лимит совокупных капитальных затрат на уровне Группы — не более $20,000,000.00",
    )
    assert estimate is None
    assert model.calls == 0


def test_percentage_limit_against_an_absolute_total_is_rejected() -> None:
    estimate, _ = _estimate(
        {"numerator": ["interest", "rent"], "threshold": 0.05, "direction": "at_most"},
        "Совокупные процентные и арендные платежи не превышают $5,000.00",
    )
    assert estimate is None


def test_library_replays_a_learned_specification_without_calling_the_model() -> None:
    library = SpecLibrary()
    text = "Максимальные расходы по статье «Маркетинговые расходы» — не более $400.00"
    library.remember(
        covenant_key(text),
        FormulaSpec(numerator=("marketing",), threshold=Decimal("400"), direction="at_most"),
    )
    estimate, model = _estimate({"numerator": ["nonsense"]}, text, library=library)
    assert estimate is not None
    assert model.calls == 0


def test_covenant_key_ignores_amounts_and_spacing() -> None:
    assert covenant_key("Лимит $400,000.00") == covenant_key("Лимит   $999,999.00")
    assert covenant_key("Лимит выручки") != covenant_key("Лимит расходов")


def test_payload_parsing_drops_invalid_fields() -> None:
    spec = spec_from_payload(
        {"numerator": ["revenue"], "threshold": "не число", "direction": "sideways"}
    )
    assert spec.threshold is None
    assert spec.direction is None
    assert not spec.is_usable()
