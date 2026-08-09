"""Full dataset run command."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from model import DatasetRef, Settings, build_pipeline
from model.services.pipeline import load_tasks

from cli.presenters.progress import RichProgressSink
from cli.presenters.report import (
    print_dataset_summary,
    print_fallback_report,
    print_run_header,
)
from cli.presenters.writer import write_submission
from cli.runtime import ensure_gemma_runtime
from cli.commands.score import score


def run_command(
    input_path: Path,
    output_path: Path,
    settings: Settings,
    key_path: Optional[Path] = None,
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

    print_run_header(input_path, output_path, settings)
    pipeline = build_pipeline(settings)
    dataset = DatasetRef(input_path)

    print("Читаю документы и леджер...")
    started = time.monotonic()
    contexts = pipeline.build_contexts(dataset)
    reading = time.monotonic() - started
    print_dataset_summary(contexts, len(load_tasks(dataset)))

    started = time.monotonic()
    with RichProgressSink() as progress:
        submission = pipeline.run(dataset, progress=progress, contexts=contexts)
    analysis = time.monotonic() - started

    write_submission(submission, output_path)
    print_fallback_report(submission, reading, analysis)
    print()
    print("Записано: {}".format(output_path))
    if key_path is not None:
        total, rows = score(output_path, key_path)
        percentage = total / len(rows) * 100 if rows else 0.0
        print("Сходство с ground_truth: {:.1f}% ({:.2f} / {})".format(percentage, total, len(rows)))
    return 0
