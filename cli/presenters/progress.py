"""Rich progress port implementation with a plain terminal fallback."""

from __future__ import annotations


class RichProgressSink:
    def __init__(self) -> None:
        self._progress = None
        self._task = None

    def __enter__(self) -> "RichProgressSink":
        try:
            from rich.progress import (
                BarColumn,
                MofNCompleteColumn,
                Progress,
                SpinnerColumn,
                TaskProgressColumn,
                TextColumn,
                TimeElapsedColumn,
            )

            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
            )
            self._progress.start()
        except ImportError:
            self._progress = None
        return self

    def __exit__(self, *args: object) -> None:
        if self._progress is not None:
            self._progress.stop()

    def on_start(self, total: int) -> None:
        if self._progress is not None:
            self._task = self._progress.add_task("Ковенанты", total=total)
        else:
            print("Ковенанты: 0/{}".format(total))

    def on_item(self, completed: int, total: int, item: str) -> None:
        if self._progress is not None and self._task is not None:
            self._progress.update(self._task, completed=completed, description=item)
        else:
            print("Ковенанты: {}/{} {}".format(completed, total, item))
