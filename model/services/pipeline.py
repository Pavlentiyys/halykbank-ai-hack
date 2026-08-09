"""Invocation-neutral covenant pipeline."""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

from model.config import Settings
from model.domain.types import (
    BorrowerContext,
    CovenantAnswer,
    CovenantTask,
    DatasetRef,
    Submission,
)
from model.ensemble.ensemble import CovenantEnsemble, make_default_answer
from model.ports.extraction import LedgerRepository, TextExtractor
from model.ports.observability import ProgressSink
from model.services.context_builder import BorrowerContextBuilder

log = logging.getLogger(__name__)


def load_tasks(dataset: DatasetRef) -> List[CovenantTask]:
    path = dataset.resolve("submission_template.json")
    with path.open("r", encoding="utf-8") as handle:
        template = json.load(handle)
    tasks = []
    for scenario_id, covenants in template["answers"].items():
        for covenant_id in covenants:
            tasks.append(CovenantTask(scenario_id=scenario_id, covenant_id=covenant_id))
    return tasks


class CovenantPipeline:
    def __init__(
        self,
        documents: TextExtractor,
        ledger: LedgerRepository,
        context_builder: BorrowerContextBuilder,
        ensemble: CovenantEnsemble,
        settings: Settings,
    ) -> None:
        self._documents = documents
        self._ledger = ledger
        self._context_builder = context_builder
        self._ensemble = ensemble
        self._settings = settings

    def run(
        self,
        dataset: DatasetRef,
        progress: Optional[ProgressSink] = None,
        contexts: Optional[Dict[str, BorrowerContext]] = None,
    ) -> Submission:
        raw_tasks = load_tasks(dataset)
        if contexts is None:
            contexts = self._context_builder.build_all(dataset, self._documents, self._ledger)
        tasks = [
            CovenantTask(
                scenario_id=task.scenario_id,
                covenant_id=task.covenant_id,
                text=contexts[task.scenario_id].covenant_texts.get(task.covenant_id, ""),
                period_from=task.period_from,
                period_to=task.period_to,
            )
            for task in raw_tasks
        ]
        answers = [make_default_answer(task) for task in tasks]
        if progress is not None:
            progress.on_start(len(tasks))

        with ThreadPoolExecutor(max_workers=max(1, self._settings.max_workers)) as executor:
            futures = {
                executor.submit(
                    self._ensemble.analyze,
                    task,
                    contexts[task.scenario_id],
                ): index
                for index, task in enumerate(tasks)
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                index = futures[future]
                task = tasks[index]
                try:
                    answers[index] = future.result()
                except Exception:
                    log.exception("cell %s/%s failed", task.scenario_id, task.covenant_id)
                if progress is not None:
                    progress.on_item(
                        completed,
                        len(tasks),
                        "{}/{}".format(task.scenario_id, task.covenant_id),
                    )

        return Submission(
            team=self._settings.team,
            contact_email=self._settings.contact_email,
            model=self._settings.model_name,
            answers=tuple(answers),
        )

    def analyze_one(
        self,
        task: CovenantTask,
        context: BorrowerContext,
    ) -> CovenantAnswer:
        return self._ensemble.analyze(task, context)

    def build_contexts(self, dataset: DatasetRef) -> Dict[str, BorrowerContext]:
        """Expose reusable contexts without coupling callers to file adapters."""
        return self._context_builder.build_all(dataset, self._documents, self._ledger)

