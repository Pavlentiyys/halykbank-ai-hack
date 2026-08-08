"""Full dataset run command."""

from pathlib import Path

from model import DatasetRef, Settings, build_pipeline

from cli.presenters.progress import RichProgressSink
from cli.presenters.report import print_fallback_report
from cli.presenters.writer import write_submission
from cli.runtime import ensure_gemma_runtime
from cli.commands.score import score


def run_command(
    input_path: Path,
    output_path: Path,
    settings: Settings,
    key_path: Path | None = None,
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
    with RichProgressSink() as progress:
        submission = pipeline.run(DatasetRef(input_path), progress=progress)
    write_submission(submission, output_path)
    print_fallback_report(submission)
    print("Записано: {}".format(output_path))
    if key_path is not None:
        total, rows = score(output_path, key_path)
        percentage = total / len(rows) * 100 if rows else 0.0
        print("Сходство с ground_truth: {:.1f}% ({:.2f} / {})".format(percentage, total, len(rows)))
    return 0
