"""Official unweighted local regression scorer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple


def score(submission_path: Path, key_path: Path) -> Tuple[float, List[Tuple[object, ...]]]:
    with submission_path.open("r", encoding="utf-8") as handle:
        submitted = json.load(handle)["answers"]
    with key_path.open("r", encoding="utf-8") as handle:
        key = json.load(handle)["scenarios"]
    total = 0.0
    rows: List[Tuple[object, ...]] = []
    for scenario_id, scenario in key.items():
        for covenant_id, expected in scenario["covenants"].items():
            answer: Dict[str, object] = submitted.get(scenario_id, {}).get(covenant_id) or {}
            points = 0.0
            if answer.get("status") == expected["status"]:
                points += 0.50
                try:
                    error = abs(float(answer["actual"]) - expected["actual"]) / abs(expected["actual"])
                    scale = max(0.0, 1 - error / 0.05)
                except (TypeError, ValueError, KeyError, ZeroDivisionError):
                    scale = 0.0
                points += 0.30 * scale
                if expected["evidence_txn_id"] is None:
                    points += 0.20 * scale
                elif answer.get("evidence_txn_id") == expected["evidence_txn_id"]:
                    points += 0.20
            total += points
            rows.append(
                (
                    scenario_id,
                    covenant_id,
                    round(points, 3),
                    answer.get("status"),
                    expected["status"],
                )
            )
    rows.sort(key=lambda row: row[2])
    return total, rows


def score_command(submission_path: Path, key_path: Path) -> int:
    total, rows = score(submission_path, key_path)
    for row in rows[:12]:
        print("  {:4} {}  {:.2f}  дали={} ключ={}".format(*row))
    print("\nИТОГО {:.2f} / {} = {:.1f}%".format(total, len(rows), total / len(rows) * 100))
    return 0

