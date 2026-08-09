"""Gemma participant: document status, covenant semantics and classifications."""

from __future__ import annotations

import json
import re
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from model.config import EnsembleSettings
from model.domain.rules import compare
from model.domain.types import BorrowerContext, CovenantTask, PageChunk
from model.ensemble.estimate import CovenantEstimate
from model.ensemble.numeric import extract_threshold
from model.ports.inference import LanguageModel, LLMRequest

COVENANT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["COMPLIANT", "BREACH"]},
        "actual": {"type": "number", "exclusiveMinimum": 0},
        "threshold": {"type": ["number", "null"]},
        "evidence_txn_id": {"type": ["string", "null"]},
        "quote": {"type": "string", "minLength": 1},
        "used_document": {"type": "string", "minLength": 1},
        "related_parties_used": {"type": "array", "items": {"type": "string"}},
        "reasoning": {"type": "string", "minLength": 1},
    },
    "required": [
        "status",
        "actual",
        "threshold",
        "evidence_txn_id",
        "quote",
        "used_document",
        "related_parties_used",
        "reasoning",
    ],
    "additionalProperties": False,
}

FULL_DOCUMENT_KINDS = frozenset({"final_audit", "kyc", "treasury_memo"})
ALLOWED_SOURCE_KINDS = FULL_DOCUMENT_KINDS | {"active_contract"}
RESPONSE_FIELDS = frozenset(COVENANT_SCHEMA["properties"])

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


def _section(text: str, start_marker: str, end_marker: str) -> str:
    start = text.find(start_marker)
    if start < 0:
        return ""
    end = text.find(end_marker, start + len(start_marker))
    return text[start : end if end >= 0 else len(text)]


def _group_pages(pages: Iterable[PageChunk]) -> Dict[str, Sequence[PageChunk]]:
    grouped: Dict[str, list] = {}
    for page in pages:
        grouped.setdefault(page.doc_name, []).append(page)
    return grouped


def compact_document_context(ctx: BorrowerContext) -> str:
    """Keep relevant evidence and definitions, not the complete loan contract."""
    chunks = []
    for doc_name, pages in _group_pages(ctx.pages).items():
        kind = ctx.document_kinds.get(doc_name, "other")
        full_text = "\n".join(page.text for page in pages)
        if kind in FULL_DOCUMENT_KINDS:
            excerpt = full_text
        elif kind == "active_contract":
            header = pages[0].text[:1200] if pages else ""
            definitions = _section(full_text, "Статья 1", "Статья 2")
            excerpt = "{}\n{}".format(header, definitions or full_text[:5000])
        else:
            continue
        chunks.append("[DOC={} KIND={}]\n{}".format(doc_name, kind, excerpt.strip()))
    return "\n\n".join(chunks)


def _decimal(value: object) -> Optional[Decimal]:
    if value is None:
        return None
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return parsed if parsed.is_finite() else None


def _normalise_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _required_text(payload: Mapping[str, object], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("semantic response field {!r} must be a non-empty string".format(field))
    return value.strip()


def _validate_task_inputs(task: CovenantTask, ctx: BorrowerContext) -> None:
    """Reject questions whose mandatory source value is absent before calling the LLM."""
    if not task.text.strip():
        raise ValueError("active covenant text is missing")

    text = _normalise_text(task.text)
    if "капитальных затрат группы" in text:
        group_capex = ctx.metrics.get("group_capex")
        if group_capex is None or not group_capex.is_finite() or group_capex <= 0:
            raise ValueError(
                "group_capex is not disclosed in the borrower documents; "
                "a semantic actual would be unsupported"
            )


def _canonical_document(value: str, ctx: BorrowerContext) -> str:
    matches = [name for name in ctx.document_kinds if name == value or name in value]
    if len(matches) != 1:
        raise ValueError("used_document does not identify exactly one borrower document")
    document = matches[0]
    if ctx.document_kinds[document] not in ALLOWED_SOURCE_KINDS:
        raise ValueError("used_document points to an ineligible document")
    return document


def _validate_quote(quote: str, task: CovenantTask, ctx: BorrowerContext) -> None:
    source_text = "\n".join(
        page.text
        for page in ctx.pages
        if ctx.document_kinds.get(page.doc_name) == "active_contract"
    )
    haystack = _normalise_text(source_text or task.text)
    if _normalise_text(quote) not in haystack:
        raise ValueError("quote is not present in the active contract")


def _validate_evidence(value: object, ctx: BorrowerContext) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError("evidence_txn_id must be a transaction id or null")
    if value not in {txn.txn_id for txn in ctx.transactions}:
        raise ValueError("evidence_txn_id is not present in the borrower ledger")
    return value


def _validate_related_parties(value: object, ctx: BorrowerContext) -> None:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError("related_parties_used must be an array of strings")
    allowed = {_normalise_text(name) for name in ctx.related_parties}
    if any(_normalise_text(item) not in allowed for item in value):
        raise ValueError("related_parties_used contains a party absent from KYC")


def _triggered_status(
    task: CovenantTask,
    ctx: BorrowerContext,
    actual: Decimal,
    threshold: Decimal,
    direction: str,
) -> str:
    trigger = re.search(
        r"только при условии,? что .*?превышают\s*\$([0-9][0-9,]*(?:\.[0-9]+)?)",
        " ".join(task.text.split()),
        flags=re.IGNORECASE,
    )
    if trigger:
        trigger_value = Decimal(trigger.group(1).replace(",", ""))
        if ctx.metrics.get("financing", Decimal("0")) <= trigger_value:
            return "COMPLIANT"
    return compare(actual, threshold, direction)  # type: ignore[arg-type]


def _validate_ratio_reasoning(actual: Decimal, reasoning: str, task: CovenantTask) -> None:
    if not re.search(r"[0-9]+(?:[.,][0-9]+)?\s*[xх]\b", task.text, flags=re.IGNORECASE):
        return
    mentioned = [
        Decimal(value.replace(",", "."))
        for value in re.findall(
            r"(?<![\w])([0-9]+(?:[.,][0-9]+)?)\s*[xх]\b",
            reasoning,
            flags=re.IGNORECASE,
        )
    ]
    tolerance = max(Decimal("0.01"), abs(actual) * Decimal("0.01"))
    if not mentioned or not any(abs(value - actual) <= tolerance for value in mentioned):
        raise ValueError("actual is inconsistent with the ratio stated in reasoning")


def validate_semantic_payload(
    payload: object,
    task: CovenantTask,
    ctx: BorrowerContext,
) -> Dict[str, object]:
    """Validate provenance and deterministic invariants of a Gemma answer."""
    if not isinstance(payload, dict):
        raise ValueError("semantic response must be a JSON object")
    missing = RESPONSE_FIELDS - set(payload)
    if missing:
        raise ValueError("semantic response is missing fields: {}".format(", ".join(sorted(missing))))

    status = payload.get("status")
    if status not in {"COMPLIANT", "BREACH"}:
        raise ValueError("semantic response has an invalid status")
    actual = _decimal(payload.get("actual"))
    if actual is None or actual <= 0:
        raise ValueError("semantic response actual must be a positive finite number")

    reasoning = _required_text(payload, "reasoning")
    quote = _required_text(payload, "quote")
    used_document = _canonical_document(_required_text(payload, "used_document"), ctx)
    _validate_quote(quote, task, ctx)
    evidence = _validate_evidence(payload.get("evidence_txn_id"), ctx)
    _validate_related_parties(payload.get("related_parties_used"), ctx)

    expected_threshold, direction = extract_threshold(task.text)
    reported_threshold = _decimal(payload.get("threshold"))
    if expected_threshold is not None and direction is not None:
        if reported_threshold is None:
            raise ValueError("semantic response omitted the contract threshold")
        tolerance = max(Decimal("0.005"), abs(expected_threshold) * Decimal("0.000001"))
        if abs(reported_threshold - expected_threshold) > tolerance:
            raise ValueError("semantic threshold differs from the active contract")
        expected_status = _triggered_status(task, ctx, actual, expected_threshold, direction)
        if status != expected_status:
            raise ValueError("semantic status is inconsistent with actual and threshold")

    _validate_ratio_reasoning(actual, reasoning, task)
    validated = dict(payload)
    validated["actual"] = actual
    validated["threshold"] = reported_threshold
    validated["evidence_txn_id"] = evidence
    validated["used_document"] = used_document
    return validated


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
        _validate_task_inputs(task, ctx)
        response = self._llm.complete(self._build_request(task, ctx))
        try:
            payload = json.loads(response.text)
        except json.JSONDecodeError as exc:
            raise ValueError("semantic model returned invalid JSON") from exc
        payload = validate_semantic_payload(payload, task, ctx)
        status = payload["status"]
        actual = payload["actual"]
        return CovenantEstimate(
            source=self.name,
            status=status,  # type: ignore[arg-type]
            actual=abs(actual),  # type: ignore[arg-type]
            threshold=payload["threshold"],  # type: ignore[arg-type]
            evidence_txn_id=payload["evidence_txn_id"],  # type: ignore[arg-type]
            confidence=0.78,
            quote=str(payload["quote"]),
            used_document=str(payload["used_document"]),
            notes=str(payload["reasoning"]),
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
        documents = compact_document_context(ctx)
        # The prefix is identical for all three cells of one borrower, allowing
        # Ollama to reuse the KV cache when --llm-mode=always and workers=1.
        prompt = """КОНТЕКСТ ЗАЁМЩИКА {scenario}

СВЯЗАННЫЕ СТОРОНЫ ИЗ KYC:
{related}

РАССЧИТАННЫЕ ПОКАЗАТЕЛИ (USD, Decimal):
{metrics}

ЛЕДЖЕР:
{ledger}

ДОКУМЕНТЫ:
{documents}

TASK — ответь только по этой ячейке:
КОВЕНАНТ: {covenant}
ТЕКСТ ДЕЙСТВУЮЩЕГО ПУНКТА:
{text}
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
