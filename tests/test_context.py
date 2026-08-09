from decimal import Decimal
from datetime import date

from model.domain.ledger import Ledger, Txn
from model.domain.types import PageChunk
from model.services.context_builder import (
    classify_document,
    extract_covenants,
    extract_related_parties,
    scenario_accounts,
)


def test_document_status_and_covenant_extraction() -> None:
    inactive = (PageChunk("old.pdf", 1, "НЕДЕЙСТВУЮЩАЯ РЕДАКЦИЯ НЕ ПРИМЕНЯЕТСЯ"),)
    assert classify_document(inactive) == "inactive_contract"
    text = (
        "Статья 6 — Финансовые ковенанты Пункт 6.1 Первый текст. "
        "Пункт 6.2 Второй текст. Пункт 6.3 Третий текст. Статья 7 — Ограничения"
    )
    clauses = extract_covenants(text)
    assert set(clauses) == {"6.1", "6.2", "6.3"}
    assert "Второй" in clauses["6.2"]


def test_covenant_extraction_uses_template_ids_and_supports_english() -> None:
    text = (
        "Financial Covenants Section 5.1 First covenant. "
        "Section 5.2 Second covenant. Section 5.3 Third covenant. "
        "Section 15.1 Governing law."
    )

    clauses = extract_covenants(text, ("5.1", "5.2", "5.3"))

    assert set(clauses) == {"5.1", "5.2", "5.3"}
    assert "Second covenant" in clauses["5.2"]
    assert "Governing law" not in clauses["5.3"]


def test_scenario_accounts_are_derived_from_the_ledger() -> None:
    ledger = Ledger.from_iterable(
        (
            Txn("TXN-X1-0001", date(2025, 1, 1), "ACC-1", "A", "x", Decimal("1"), "USD"),
            Txn("TXN-KC-0001", date(2025, 1, 1), "TELE-9", "B", "y", Decimal("1"), "USD"),
        )
    )

    assert scenario_accounts(ledger, ("X1", "KC")) == {
        "X1": "ACC-1",
        "KC": "TELE-9",
    }


def test_english_document_statuses_are_recognised() -> None:
    active = (PageChunk("current.pdf", 1, "CREDIT AGREEMENT LOAN REFERENCE ACC-1"),)
    inactive = (
        PageChunk("old.pdf", 1, "SUPERSEDED PRIOR-YEAR CREDIT AGREEMENT. NOT OPERATIVE."),
    )
    final = (PageChunk("audit.pdf", 1, "A U D I T O R ' S F I L E C O P Y"),)

    assert classify_document(active) == "active_contract"
    assert classify_document(inactive) == "inactive_contract"
    assert classify_document(final) == "final_audit"


def test_kyc_threshold_comes_from_document() -> None:
    text = (
        "Доля голосующих прав Alpha Holding LLP 34.5% Beta Works LLP 18.7% "
        "Организации, в которых Группа владеет 30.0% и более голосующих прав, "
        "признаются связанными сторонами"
    )
    assert extract_related_parties(text, Decimal("0.20")) == ("Alpha Holding LLP",)
