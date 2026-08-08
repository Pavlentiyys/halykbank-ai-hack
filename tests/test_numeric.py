from datetime import date
from decimal import Decimal

from model.config import EnsembleSettings
from model.domain.ledger import Txn
from model.domain.types import BorrowerContext, CovenantTask, PageChunk
from model.ensemble.numeric import NumericEstimator


def _txn(identifier: str, amount: str, counterparty: str, description: str) -> Txn:
    return Txn(
        txn_id=identifier,
        date=date(2025, 6, 1),
        account_id="ACC-TEST",
        counterparty=counterparty,
        description=description,
        amount=Decimal(amount),
        currency="USD",
    )


def test_ratio_status_uses_submitted_two_decimal_value() -> None:
    context = BorrowerContext(
        scenario_id="T1",
        account_id="ACC-TEST",
        related_parties=("Related Holdings LLP",),
        transactions=(
            _txn("revenue", "1000", "Customer", "Service sales settlement 2025"),
            _txn("related", "-41", "Related Holdings L.L.P.", "Management advisory retainer"),
        ),
    )
    task = CovenantTask(
        "T1",
        "6.3",
        "Maximum Related-Party Payments as a Proportion of Revenue must not exceed 0.04x",
    )

    estimate = NumericEstimator(EnsembleSettings()).estimate(task, context)

    assert estimate.actual is not None
    assert estimate.status == "COMPLIANT"


def test_related_party_swing_uses_normalized_counterparty_name() -> None:
    context = BorrowerContext(
        scenario_id="T1",
        account_id="ACC-TEST",
        related_parties=("Related Holdings LLP",),
        transactions=(
            _txn("revenue", "1000", "Customer", "Service sales settlement 2025"),
            _txn("related", "-46", "Related Holdings L.L.P.", "Management advisory retainer"),
        ),
    )
    task = CovenantTask(
        "T1",
        "6.3",
        "Maximum Related-Party Payments as a Proportion of Revenue must not exceed 0.04x",
    )

    estimate = NumericEstimator(EnsembleSettings()).estimate(task, context)

    assert estimate.status == "BREACH"
    assert estimate.evidence_txn_id == "related"


def test_document_backed_transaction_preserves_raw_ratio_breach() -> None:
    context = BorrowerContext(
        scenario_id="T1",
        account_id="ACC-TEST",
        pages=(PageChunk("workpaper.pdf", 2, "Operation related was reviewed."),),
        related_parties=("Related Holdings LLP",),
        transactions=(
            _txn("revenue", "1000", "Customer", "Service sales settlement 2025"),
            _txn("related", "-43", "Related Holdings L.L.P.", "Management advisory retainer"),
        ),
    )
    task = CovenantTask(
        "T1",
        "6.3",
        "Maximum Related-Party Payments as a Proportion of Revenue must not exceed 0.04x",
    )

    estimate = NumericEstimator(EnsembleSettings()).estimate(task, context)

    assert estimate.status == "BREACH"
    assert estimate.evidence_txn_id == "related"
