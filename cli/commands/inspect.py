"""Single-cell diagnostic command."""

from pathlib import Path

from model import CovenantTask, DatasetRef, Settings, build_pipeline
from cli.runtime import ensure_gemma_runtime


def inspect_command(
    input_path: Path,
    scenario_id: str,
    covenant_id: str,
    settings: Settings,
) -> int:
    try:
        prepared = ensure_gemma_runtime(settings)
    except RuntimeError as exc:
        print("Ошибка запуска модели: {}".format(exc))
        return 2
    if prepared is None:
        print("Запуск отменён.")
        return 2
    settings = prepared
    pipeline = build_pipeline(settings)
    dataset = DatasetRef(input_path)
    contexts = pipeline.build_contexts(dataset)
    if scenario_id not in contexts:
        print("Неизвестный сценарий: {}".format(scenario_id))
        return 2
    context = contexts[scenario_id]
    task = CovenantTask(
        scenario_id=scenario_id,
        covenant_id=covenant_id,
        text=context.covenant_texts.get(covenant_id, ""),
    )
    answer = pipeline.analyze_one(task, context)
    print("{}/{}: {} actual={:.2f}".format(scenario_id, covenant_id, answer.status, answer.actual))
    print("threshold={} evidence={}".format(answer.threshold, answer.evidence_txn_id))
    print("document={} disagreement={}".format(answer.used_document, answer.has_disagreement))
    print("reasoning={}".format(answer.reasoning))
    print("related_parties={}".format(", ".join(context.related_parties)))
    print("covenant={}".format(task.text))
    return 0
