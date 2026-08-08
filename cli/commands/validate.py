"""Strict structural validation against the organizer template."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List


def validation_errors(submission_path: Path, template_path: Path) -> List[str]:
    with submission_path.open("r", encoding="utf-8") as handle:
        submission = json.load(handle)
    with template_path.open("r", encoding="utf-8") as handle:
        template = json.load(handle)
    errors = []
    if set(submission) != {"team", "contact_email", "model", "answers"}:
        errors.append("top-level keys differ from required schema")
    for field in ("team", "contact_email", "model"):
        if not isinstance(submission.get(field), str) or not submission.get(field):
            errors.append("{} must be a non-empty string".format(field))
    answers = submission.get("answers", {})
    if set(answers) != set(template["answers"]):
        errors.append("scenario keys differ from template")
    for scenario_id, expected_covenants in template["answers"].items():
        actual_covenants = answers.get(scenario_id, {})
        if set(actual_covenants) != set(expected_covenants):
            errors.append("{} covenant keys differ from template".format(scenario_id))
        for covenant_id, cell in actual_covenants.items():
            label = "{}/{}".format(scenario_id, covenant_id)
            if set(cell) != {"status", "actual", "evidence_txn_id"}:
                errors.append("{} has invalid cell keys".format(label))
                continue
            if cell["status"] not in {"COMPLIANT", "BREACH"}:
                errors.append("{} has invalid status".format(label))
            actual = cell["actual"]
            if (
                isinstance(actual, bool)
                or not isinstance(actual, (int, float))
                or not math.isfinite(actual)
                or actual < 0
            ):
                errors.append("{} actual must be finite and non-negative".format(label))
            evidence = cell["evidence_txn_id"]
            if evidence is not None and not isinstance(evidence, str):
                errors.append("{} evidence_txn_id must be a string or null".format(label))
    return errors


def validate_command(submission_path: Path, template_path: Path) -> int:
    errors = validation_errors(submission_path, template_path)
    if errors:
        for error in errors:
            print("ОШИБКА: {}".format(error))
        return 1
    print("OK: submission соответствует шаблону")
    return 0

