"""Small inference ports kept independent from covenant implementations."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Protocol, Sequence

from model.domain.types import BorrowerContext, CovenantAnswer, CovenantTask


@dataclass(frozen=True)
class LLMRequest:
    system: str
    prompt: str
    schema: Optional[Mapping[str, Any]] = None

    def fingerprint(self) -> str:
        return json.dumps(
            {"system": self.system, "prompt": self.prompt, "schema": self.schema},
            ensure_ascii=False,
            sort_keys=True,
        )


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str = ""
    usage: Mapping[str, int] = field(default_factory=dict)


class LanguageModel(Protocol):
    def complete(self, request: LLMRequest) -> LLMResponse: ...


class Estimator(Protocol):
    name: str

    def estimate(self, task: CovenantTask, ctx: BorrowerContext) -> object: ...


class ResolutionPolicy(Protocol):
    def resolve(self, task: CovenantTask, estimates: Sequence[object]) -> CovenantAnswer: ...

