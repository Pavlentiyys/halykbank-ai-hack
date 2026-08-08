"""Gemma participant: document status, covenant semantics and classifications."""

from __future__ import annotations

import json
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Optional

from model.config import EnsembleSettings
from model.domain.types import BorrowerContext, CovenantTask
from model.ensemble.estimate import CovenantEstimate
from model.ports.inference import LanguageModel, LLMRequest

COVENANT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["COMPLIANT", "BREACH"]},
        "actual": {"type": "number"},
        "threshold": {"type": ["number", "null"]},
        "evidence_txn_id": {"type": ["string", "null"]},
        "quote": {"type": "string"},
        "used_document": {"type": "string"},
        "related_parties_used": {"type": "array", "items": {"type": "string"}},
        "reasoning": {"type": "string"},
    },
    "required": ["status", "actual", "reasoning"],
}

SYSTEM_PROMPT = """Ты — аудитор финансовых ковенантов. Верни только JSON по схеме.
Обязательные правила:
1. Сначала проверь статус документа. Недействующую редакцию 2024 года не используй. Промежуточную ведомость не применяй.
2. Корректировки окончательного аудиторского дела применяй обязательно.
3. Связанная сторона определяется только таблицей KYC и указанным там порогом доли. Похожие названия не подходят.
4. Категорию операции определяй по description. Возвраты, ребейты, кредит-ноты, возврат депозита, drawdown и процентный доход не являются выручкой.
5. Пустую сумму операции восстанавливай из документов.
6. quote должен быть дословной цитатой из действующего договора.
7. Используй переданные рассчитанные показатели; не выполняй неточную арифметику.
8. actual всегда положителен и содержит фактический показатель даже для springing-теста.
9. evidence_txn_id — только единственная операция, удаление которой меняет вердикт; иначе null.
10. Если данных недостаточно, дай лучшую оценку, не оставляй обязательные поля пустыми.
"""


def _decimal(value: object) -> Optional[Decimal]:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


class SemanticEstimator:
    name = "gemma"

    def __init__(
        self,
        llm: LanguageModel,
        settings: EnsembleSettings,
        max_context_chars: int = 300_000,
    ) -> None:
        self._llm = llm
        self._settings = settings
        self._max_context_chars = max_context_chars

    def estimate(self, task: CovenantTask, ctx: BorrowerContext) -> CovenantEstimate:
        response = self._llm.complete(self._build_request(task, ctx))
        try:
            payload = json.loads(response.text)
        except json.JSONDecodeError as exc:
            raise ValueError("semantic model returned invalid JSON") from exc
        status = payload.get("status")
        if status not in {"COMPLIANT", "BREACH"}:
            status = None
        actual = _decimal(payload.get("actual"))
        confidence = 0.78 if status and actual is not None else 0.0
        return CovenantEstimate(
            source=self.name,
            status=status,
            actual=abs(actual) if actual is not None else None,
            threshold=_decimal(payload.get("threshold")),
            evidence_txn_id=payload.get("evidence_txn_id") or None,
            confidence=confidence,
            quote=str(payload.get("quote") or ""),
            used_document=str(payload.get("used_document") or ""),
            notes=str(payload.get("reasoning") or ""),
        )

    def _build_request(self, task: CovenantTask, ctx: BorrowerContext) -> LLMRequest:
        metrics = "\n".join("{}={}".format(key, value) for key, value in sorted(ctx.metrics.items()))
        ledger = "\n".join(
            "{} | {} | {} | {} | {} {}".format(
                txn.txn_id,
                txn.date.isoformat(),
                txn.counterparty,
                txn.description,
                "" if txn.amount is None else txn.amount,
                txn.currency,
            )
            for txn in ctx.transactions
        )
        documents = "\n\n".join(
            "[DOC={} KIND={} PAGE={}]\n{}".format(
                page.doc_name,
                ctx.document_kinds.get(page.doc_name, "other"),
                page.page,
                page.text,
            )
            for page in ctx.pages
        )
        prompt = """СЦЕНАРИЙ: {scenario}
КОВЕНАНТ: {covenant}
ТЕКСТ ДЕЙСТВУЮЩЕГО ПУНКТА:
{text}

СВЯЗАННЫЕ СТОРОНЫ ИЗ KYC:
{related}

РАССЧИТАННЫЕ ПОКАЗАТЕЛИ (USD, Decimal):
{metrics}

ЛЕДЖЕР:
{ledger}

ДОКУМЕНТЫ:
{documents}
""".format(
            scenario=task.scenario_id,
            covenant=task.covenant_id,
            text=task.text,
            related=", ".join(ctx.related_parties),
            metrics=metrics,
            ledger=ledger,
            documents=documents,
        )
        return LLMRequest(
            system=SYSTEM_PROMPT,
            prompt=prompt[: self._max_context_chars],
            schema=COVENANT_SCHEMA,
        )

