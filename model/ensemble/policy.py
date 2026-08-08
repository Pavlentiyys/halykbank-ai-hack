"""Field-level resolution policy: deterministic arithmetic wins."""

from __future__ import annotations

from typing import Optional, Sequence

from model.domain.types import CovenantAnswer, CovenantTask
from model.ensemble.estimate import CovenantEstimate


def _usable(estimates: Sequence[CovenantEstimate], source: str) -> Optional[CovenantEstimate]:
    candidates = [estimate for estimate in estimates if estimate.source == source]
    return max(candidates, key=lambda estimate: estimate.confidence, default=None)


class NumericAuthoritativePolicy:
    def resolve(
        self,
        task: CovenantTask,
        estimates: Sequence[CovenantEstimate],
    ) -> CovenantAnswer:
        numeric = _usable(estimates, "numeric")
        semantic = _usable(estimates, "gemma")
        ranked = sorted(estimates, key=lambda estimate: estimate.confidence, reverse=True)

        actual = numeric.actual if numeric and numeric.actual is not None else None
        status = numeric.status if numeric and numeric.status is not None else None
        evidence = numeric.evidence_txn_id if numeric else None
        if actual is None:
            actual = next((item.actual for item in ranked if item.actual is not None), None)
        if status is None:
            status = next((item.status for item in ranked if item.status is not None), None)
        if evidence is None:
            evidence = next(
                (item.evidence_txn_id for item in ranked if item.evidence_txn_id is not None),
                None,
            )
        if actual is None or status is None:
            raise ValueError("estimates do not contain an answer")

        disagreement = bool(
            numeric
            and semantic
            and numeric.status is not None
            and semantic.status is not None
            and numeric.status != semantic.status
        )
        notes = " | ".join(
            "{}[{:.2f}]: {}".format(item.source, item.confidence, item.notes)
            for item in estimates
            if item.notes
        )
        return CovenantAnswer(
            scenario_id=task.scenario_id,
            covenant_id=task.covenant_id,
            status=status,
            actual=round(float(abs(actual)), 2),
            evidence_txn_id=evidence,
            threshold=float(semantic.threshold) if semantic and semantic.threshold is not None else (
                float(numeric.threshold) if numeric and numeric.threshold is not None else None
            ),
            quote=semantic.quote if semantic else None,
            used_document=semantic.used_document if semantic else None,
            reasoning=notes or None,
            has_disagreement=disagreement,
        )

