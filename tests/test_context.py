from decimal import Decimal

from model.domain.types import PageChunk
from model.services.context_builder import (
    classify_document,
    extract_covenants,
    extract_related_parties,
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


def test_kyc_threshold_comes_from_document() -> None:
    text = (
        "Доля голосующих прав Alpha Holding LLP 34.5% Beta Works LLP 18.7% "
        "Организации, в которых Группа владеет 30.0% и более голосующих прав, "
        "признаются связанными сторонами"
    )
    assert extract_related_parties(text, Decimal("0.20")) == ("Alpha Holding LLP",)

