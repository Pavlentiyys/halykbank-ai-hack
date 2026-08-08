"""Submission disk writer; persistence intentionally stays outside core."""

import json
from pathlib import Path

from model import Submission


def write_submission(submission: Submission, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(submission.to_submission_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

