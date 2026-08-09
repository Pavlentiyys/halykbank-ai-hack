import json
from pathlib import Path

from model import CovenantAnswer, Submission


def test_serialization_matches_template_keys() -> None:
    template = json.loads(Path("public/submission_template.json").read_text(encoding="utf-8"))
    answers = tuple(
        CovenantAnswer(
            scenario_id=scenario_id,
            covenant_id=covenant_id,
            status="COMPLIANT",
            actual=-1.234,
            reasoning="must not be serialized",
        )
        for scenario_id, covenants in template["answers"].items()
        for covenant_id in covenants
    )
    payload = Submission("team", "team@example.com", "model", answers).to_submission_dict()
    assert set(payload) == {"team", "contact_email", "model", "answers"}
    assert set(payload["answers"]) == set(template["answers"])
    for scenario_id, covenants in payload["answers"].items():
        assert set(covenants) == set(template["answers"][scenario_id])
        for cell in covenants.values():
            assert cell == {
                "status": "COMPLIANT",
                "actual": 1.23,
                "evidence_txn_id": None,
            }
