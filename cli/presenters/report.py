"""Human-readable run diagnostics."""

from __future__ import annotations

import collections
from pathlib import Path
from typing import Mapping, Optional, Sequence

from model import BorrowerContext, Settings, Submission


def print_run_header(
    input_path: Path,
    output_path: Path,
    settings: Settings,
) -> None:
    gemma = settings.ensemble.gemma
    engine = (
        "{} через {}".format(gemma.model_id, gemma.endpoint)
        if gemma.enabled
        else "offline, детерминированная заглушка вместо Gemma"
    )
    cache = str(settings.cache_dir) if settings.cache_dir else "выключен"
    print("Набор данных : {}".format(input_path))
    print("Результат    : {}".format(output_path))
    print("Модель       : {}".format(engine))
    print("Режим Gemma  : {}".format(settings.ensemble.llm_mode))
    print("Кэш ответов  : {}".format(cache))
    print("Потоков      : {}".format(settings.max_workers))
    print("Курс EUR/USD : {}".format(settings.ensemble.fx_eur_usd))


def print_dataset_summary(contexts: Mapping[str, BorrowerContext], total_cells: int) -> None:
    documents = {name for ctx in contexts.values() for name in ctx.document_kinds}
    kinds: collections.Counter = collections.Counter()
    for ctx in contexts.values():
        kinds.update(ctx.document_kinds.values())
    transactions = sum(len(ctx.transactions) for ctx in contexts.values())
    missing_amounts = sum(
        1 for ctx in contexts.values() for txn in ctx.transactions if txn.amount is None
    )
    without_clause = [
        scenario_id
        for scenario_id, ctx in contexts.items()
        if not ctx.covenant_texts
    ]

    print()
    print("Заёмщиков    : {}".format(len(contexts)))
    print("Ячеек        : {}".format(total_cells))
    print("Документов   : {} отобрано по номеру счёта".format(len(documents)))
    interesting = ("active_contract", "final_audit", "kyc", "treasury_memo", "inactive_contract")
    parts = ["{} {}".format(kinds.get(kind, 0), kind) for kind in interesting if kinds.get(kind)]
    if parts:
        print("               {}".format(", ".join(parts)))
    print("Операций     : {}".format(transactions))
    if missing_amounts:
        print("               {} без суммы — восстанавливаются из документов".format(missing_amounts))
    if without_clause:
        print("Без текста ковенантов: {}".format(", ".join(sorted(without_clause))))
    print()


def print_fallback_report(
    submission: Submission,
    reading: Optional[float] = None,
    analysis: Optional[float] = None,
) -> None:
    answers = submission.answers
    fallbacks = [answer for answer in answers if answer.is_fallback]
    disagreements = [answer for answer in answers if answer.has_disagreement]
    statuses: collections.Counter = collections.Counter(answer.status for answer in answers)
    with_evidence = [answer for answer in answers if answer.evidence_txn_id]

    print()
    print("Посчитано    : {} из {}".format(len(answers) - len(fallbacks), len(answers)))
    print("Вердикты     : {} COMPLIANT, {} BREACH".format(
        statuses.get("COMPLIANT", 0),
        statuses.get("BREACH", 0),
    ))
    print("Доказательств: {}".format(len(with_evidence)))
    if reading is not None and analysis is not None:
        print("Время        : {:.1f} c чтение + {:.1f} c анализ = {:.1f} c".format(
            reading, analysis, reading + analysis
        ))

    _print_cells("На дефолте", fallbacks)
    _print_cells("Расхождения участников", disagreements)


def _print_cells(title: str, answers: Sequence[object]) -> None:
    if not answers:
        return
    print()
    print("{} ({}):".format(title, len(answers)))
    for answer in answers:
        reason = getattr(answer, "reasoning", None) or ""
        suffix = " — {}".format(reason.split("|")[0].strip()) if reason else ""
        print("  {}/{}{}".format(answer.scenario_id, answer.covenant_id, suffix))
