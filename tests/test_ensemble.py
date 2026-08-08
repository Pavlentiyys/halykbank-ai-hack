"""Ensemble contract: participant isolation and field-level resolution."""

from decimal import Decimal

import pytest

from model.domain.types import BorrowerContext, CovenantTask
from model.ensemble.ensemble import CovenantEnsemble, make_default_answer
from model.ensemble.estimate import CovenantEstimate
from model.ensemble.policy import NumericAuthoritativePolicy

TASK = CovenantTask(scenario_id="P1", covenant_id="6.3")
CONTEXT = BorrowerContext(scenario_id="P1", account_id="ACC-7801")


class StubEstimator:
    def __init__(self, name: str, estimate: object) -> None:
        self.name = name
        self._estimate = estimate

    def estimate(self, task: CovenantTask, ctx: BorrowerContext) -> object:
        if isinstance(self._estimate, Exception):
            raise self._estimate
        return self._estimate


def _numeric(**overrides: object) -> CovenantEstimate:
    values = {
        "source": "numeric",
        "status": "BREACH",
        "actual": Decimal("-283664.18"),
        "confidence": 0.9,
    }
    values.update(overrides)
    return CovenantEstimate(**values)  # type: ignore[arg-type]


def _gemma(**overrides: object) -> CovenantEstimate:
    values = {
        "source": "gemma",
        "status": "COMPLIANT",
        "actual": Decimal("1.00"),
        "threshold": Decimal("450000"),
        "quote": "Пункт 6.3",
        "used_document": "8d878af064f2.pdf",
        "confidence": 0.5,
    }
    values.update(overrides)
    return CovenantEstimate(**values)  # type: ignore[arg-type]


def _ensemble(*estimators: StubEstimator) -> CovenantEnsemble:
    return CovenantEnsemble(
        estimators=list(estimators),
        policy=NumericAuthoritativePolicy(),
        fallback=make_default_answer,
    )


# --------------------------------------------------------------- isolation


def test_failing_participant_does_not_sink_the_cell() -> None:
    ensemble = _ensemble(
        StubEstimator("gemma", RuntimeError("model is down")),
        StubEstimator("numeric", _numeric()),
    )

    answer = ensemble.analyze(TASK, CONTEXT)

    assert answer.is_fallback is False
    assert answer.status == "BREACH"


def test_all_participants_failing_yields_a_valid_default() -> None:
    ensemble = _ensemble(
        StubEstimator("gemma", RuntimeError("boom")),
        StubEstimator("numeric", ValueError("boom")),
    )

    answer = ensemble.analyze(TASK, CONTEXT)

    assert answer.is_fallback is True
    assert answer.status in ("COMPLIANT", "BREACH")
    assert answer.actual > 0
    assert (answer.scenario_id, answer.covenant_id) == ("P1", "6.3")


def test_non_estimate_return_value_is_ignored() -> None:
    ensemble = _ensemble(
        StubEstimator("gemma", {"status": "COMPLIANT"}),
        StubEstimator("numeric", _numeric()),
    )

    assert ensemble.analyze(TASK, CONTEXT).status == "BREACH"


def test_resolution_failure_falls_back_instead_of_raising() -> None:
    ensemble = _ensemble(StubEstimator("gemma", _gemma(status=None, actual=None)))

    assert ensemble.analyze(TASK, CONTEXT).is_fallback is True


def test_empty_ensemble_still_answers() -> None:
    assert _ensemble().analyze(TASK, CONTEXT).is_fallback is True


# ------------------------------------------------------------------ policy


def test_numeric_wins_the_numbers_and_gemma_the_semantics() -> None:
    answer = NumericAuthoritativePolicy().resolve(TASK, [_gemma(), _numeric()])

    assert answer.status == "BREACH"                    # от numeric
    assert answer.actual == 283664.18                   # от numeric, по модулю
    assert answer.quote == "Пункт 6.3"                  # от gemma
    assert answer.used_document == "8d878af064f2.pdf"   # от gemma


def test_numeric_wins_even_with_lower_confidence() -> None:
    answer = NumericAuthoritativePolicy().resolve(
        TASK,
        [_gemma(confidence=0.99), _numeric(confidence=0.1)],
    )

    assert answer.status == "BREACH"


def test_gemma_fills_gaps_the_numeric_estimator_left() -> None:
    answer = NumericAuthoritativePolicy().resolve(
        TASK,
        [_gemma(evidence_txn_id="TXN-P1-0031"), _numeric(evidence_txn_id=None)],
    )

    assert answer.evidence_txn_id == "TXN-P1-0031"


def test_actual_is_positive_and_rounded_to_two_places() -> None:
    answer = NumericAuthoritativePolicy().resolve(
        TASK,
        [_numeric(actual=Decimal("-1234.5678"))],
    )

    assert answer.actual == 1234.57


def test_disagreement_is_recorded_for_the_report() -> None:
    agreed = NumericAuthoritativePolicy().resolve(
        TASK,
        [_gemma(status="BREACH"), _numeric(status="BREACH")],
    )
    conflicted = NumericAuthoritativePolicy().resolve(TASK, [_gemma(), _numeric()])

    assert agreed.has_disagreement is False
    assert conflicted.has_disagreement is True


def test_an_answerless_estimate_set_is_rejected() -> None:
    with pytest.raises(ValueError):
        NumericAuthoritativePolicy().resolve(TASK, [_gemma(status=None, actual=None)])
