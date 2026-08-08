"""End-to-end offline regression against the published ground truth.

Guards the number quoted in README: a deterministic, network-free run of the
whole pipeline must keep scoring at least as well as it does today.
"""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from cli.commands.score import score
from model import DatasetRef, Settings, build_pipeline

DATASET = Path(__file__).resolve().parent.parent
BASELINE = 35.00  # из 36; ухудшение — регрессия, улучшение — обновите константу


pytestmark = pytest.mark.skipif(
    not (DATASET / "ground_truth.json").exists(),
    reason="открытый набор с ключом рядом с репозиторием отсутствует",
)


def _offline_settings() -> Settings:
    base = Settings()
    gemma = replace(base.ensemble.gemma, enabled=False)
    return replace(
        base,
        ensemble=replace(base.ensemble, gemma=gemma),
        cache_dir=None,
        max_workers=1,
        model_name="offline numeric",
    )


@pytest.fixture(scope="module")
def submission(tmp_path_factory: pytest.TempPathFactory) -> Path:
    pipeline = build_pipeline(_offline_settings())
    result = pipeline.run(DatasetRef(DATASET))
    path = tmp_path_factory.mktemp("regression") / "submission.json"
    path.write_text(
        json.dumps(result.to_submission_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def test_offline_run_does_not_regress(submission: Path) -> None:
    total, rows = score(submission, DATASET / "ground_truth.json")

    assert len(rows) == 36
    assert total >= BASELINE, "offline-прогон просел: {:.2f} < {:.2f}".format(total, BASELINE)


def test_every_cell_is_answered(submission: Path) -> None:
    answers = json.loads(submission.read_text(encoding="utf-8"))["answers"]
    template = json.loads(
        (DATASET / "submission_template.json").read_text(encoding="utf-8")
    )["answers"]

    assert set(answers) == set(template)
    for scenario_id, covenants in template.items():
        assert set(answers[scenario_id]) == set(covenants)
        for covenant_id, cell in answers[scenario_id].items():
            assert set(cell) == {"status", "actual", "evidence_txn_id"}
            assert cell["status"] in ("COMPLIANT", "BREACH")
            assert isinstance(cell["actual"], (int, float))
            assert cell["actual"] >= 0


def test_verdicts_are_not_degenerate(submission: Path) -> None:
    answers = json.loads(submission.read_text(encoding="utf-8"))["answers"]
    statuses = {cell["status"] for covenants in answers.values() for cell in covenants.values()}

    assert statuses == {"COMPLIANT", "BREACH"}
