"""Third participant: the model describes the calculation, Python performs it."""

from __future__ import annotations

import json
import logging
from decimal import Decimal
from typing import Optional

from model.config import EnsembleSettings
from model.domain.formula_spec import (
    SPEC_SCHEMA,
    FormulaSpec,
    covenant_key,
    spec_from_payload,
)
from model.domain.rules import compare
from model.domain.types import BorrowerContext, CovenantTask
from model.ensemble.estimate import CovenantEstimate
from model.ports.inference import LanguageModel, LLMRequest
from model.services.metrics import by_category
from model.services.spec_executor import execute_spec, vocabulary

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """Ты переводишь текст финансового ковенанта в спецификацию расчёта.
Верни только JSON по схеме. Сам ничего не вычисляй — числа посчитает программа.

Правила:
1. numerator и denominator — списки имён из предоставленного словаря, ничего кроме них.
2. Для абсолютного лимита denominator оставь пустым; для коэффициента заполни оба списка.
3. numerator_less и denominator_less — то, что вычитается: например EBITDA это
   numerator=["revenue"], numerator_less=["operating"].
4. threshold — числовой лимит из текста пункта. Для процентов дай долю: 30% это 0.30.
5. direction: at_most, если показатель не должен превышать лимит; at_least, если не должен быть ниже.
6. scope: quarter_max, если ограничение применяется к любому отдельному кварталу;
   quarter_last — если только к четвёртому; иначе period.
7. related_only=true, только если ограничение касается платежей связанным сторонам.
"""


RATIO_MARKERS = ("отношение", "коэффициент", "доля", "ratio", "margin", "cover", "leverage", "%")

# Показатели уровня Группы считаются по консолидированной отчётности материнской
# компании. Собственный леджер заёмщика её не заменяет, и подстановка своей суммы
# даёт уверенно неверный ответ вместо честного отказа.
BEYOND_LEDGER_MARKERS = (
    "групп",
    "консолидирован",
    "материнск",
    "group level",
    "consolidated",
    "ultimate parent",
)


def _wants_ratio(text: str) -> bool:
    lowered = text.casefold()
    return any(marker in lowered for marker in RATIO_MARKERS)


def _beyond_ledger(text: str) -> bool:
    lowered = text.casefold()
    return any(marker in lowered for marker in BEYOND_LEDGER_MARKERS)


def _scale_mismatch(text: str, spec: FormulaSpec) -> bool:
    """A dollar limit parsed as a fraction, or a percentage parsed as dollars."""
    if spec.threshold is None:
        return False
    money = "$" in text
    if not spec.denominator and money and spec.threshold < Decimal("1"):
        return True
    if spec.denominator and spec.threshold > Decimal("1000"):
        return True
    return False


class SynthesizedFormulaEstimator:
    """Handles covenant wordings the hand-written formula bank does not know."""

    name = "synthesized"

    def __init__(
        self,
        llm: LanguageModel,
        settings: EnsembleSettings,
        library=None,
    ) -> None:
        self._llm = llm
        self._settings = settings
        self._library = library

    def estimate(self, task: CovenantTask, ctx: BorrowerContext) -> Optional[CovenantEstimate]:
        if not task.text.strip():
            return None
        if _beyond_ledger(task.text):
            log.info(
                "%s/%s needs a figure outside the borrower ledger; leaving it to the default",
                task.scenario_id,
                task.covenant_id,
            )
            return None
        grouped = by_category(
            tuple(txn for txn in ctx.transactions if txn.txn_id not in set(ctx.excluded_txn_ids)),
            ctx.category_overrides,
        )
        spec = self._spec_for(task, grouped, ctx)
        if spec is None or not spec.is_usable():
            return None

        value, rows = execute_spec(spec, grouped, ctx.metrics, ctx.related_parties)
        if value is None:
            return None

        status = compare(abs(value), spec.threshold, spec.direction)
        return CovenantEstimate(
            source=self.name,
            status=status,
            actual=abs(Decimal(value)),
            threshold=spec.threshold,
            confidence=0.85,
            notes="synthesized spec: {}".format(spec.note or spec.direction),
        )

    def _spec_for(self, task: CovenantTask, grouped, ctx: BorrowerContext) -> Optional[FormulaSpec]:
        key = covenant_key(task.text)
        if self._library is not None:
            remembered = self._library.get(key)
            if remembered is not None:
                return remembered

        names = vocabulary(grouped, ctx.metrics)
        request = LLMRequest(
            system=SYSTEM_PROMPT,
            prompt="СЛОВАРЬ ИМЁН:\n{}\n\nТЕКСТ ПУНКТА:\n{}\n".format(", ".join(names), task.text),
            schema=SPEC_SCHEMA,
        )
        try:
            response = self._llm.complete(request)
            spec = spec_from_payload(json.loads(response.text))
        except Exception:
            log.warning(
                "spec synthesis failed for %s/%s", task.scenario_id, task.covenant_id, exc_info=True
            )
            return None

        unknown = [
            name
            for name in spec.numerator + spec.numerator_less + spec.denominator + spec.denominator_less
            if name not in names
        ]
        if unknown:
            log.warning("synthesized spec references unknown names %s", unknown)
            return None
        if _wants_ratio(task.text) and not spec.denominator:
            log.warning(
                "synthesized spec for %s/%s has no denominator for a ratio covenant",
                task.scenario_id,
                task.covenant_id,
            )
            return None
        if _scale_mismatch(task.text, spec):
            log.warning(
                "synthesized spec for %s/%s mixes a percentage limit with an absolute total",
                task.scenario_id,
                task.covenant_id,
            )
            return None

        if self._library is not None and spec.is_usable():
            self._library.remember(key, spec)
        return spec
