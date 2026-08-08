"""Open ensemble orchestration with isolated participant failures."""

from __future__ import annotations

import logging
from typing import Callable, Sequence

from model.domain.types import BorrowerContext, CovenantAnswer, CovenantTask
from model.ensemble.estimate import CovenantEstimate
from model.ports.inference import Estimator, ResolutionPolicy

log = logging.getLogger(__name__)
FallbackFactory = Callable[[CovenantTask], CovenantAnswer]


def make_default_answer(task: CovenantTask) -> CovenantAnswer:
    return CovenantAnswer(
        scenario_id=task.scenario_id,
        covenant_id=task.covenant_id,
        status="COMPLIANT",
        actual=0.01,
        evidence_txn_id=None,
        reasoning="fallback: no usable estimate",
        is_fallback=True,
    )


class CovenantEnsemble:
    def __init__(
        self,
        estimators: Sequence[Estimator],
        policy: ResolutionPolicy,
        fallback: FallbackFactory = make_default_answer,
    ) -> None:
        self._estimators = tuple(estimators)
        self._policy = policy
        self._fallback = fallback

    def analyze(self, task: CovenantTask, ctx: BorrowerContext) -> CovenantAnswer:
        estimates = []
        for estimator in self._estimators:
            try:
                estimate = estimator.estimate(task, ctx)
                if isinstance(estimate, CovenantEstimate):
                    estimates.append(estimate)
            except Exception:
                log.warning(
                    "estimator %s failed on %s/%s",
                    estimator.name,
                    task.scenario_id,
                    task.covenant_id,
                    exc_info=True,
                )
        if not estimates:
            return self._fallback(task)
        try:
            return self._policy.resolve(task, estimates)
        except Exception:
            log.exception("resolution failed on %s/%s", task.scenario_id, task.covenant_id)
            return self._fallback(task)

