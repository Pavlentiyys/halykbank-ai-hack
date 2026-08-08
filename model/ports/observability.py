"""Output-neutral observability ports."""

from typing import Mapping, Protocol


class ProgressSink(Protocol):
    def on_start(self, total: int) -> None: ...

    def on_item(self, completed: int, total: int, item: str) -> None: ...


class MetricsSink(Protocol):
    def record(self, name: str, value: float, tags: Mapping[str, str]) -> None: ...

