"""Human-readable run diagnostics."""

from model import Submission


def print_fallback_report(submission: Submission) -> None:
    fallbacks = [answer for answer in submission.answers if answer.is_fallback]
    disagreements = [answer for answer in submission.answers if answer.has_disagreement]
    print("Fallbacks: {}/{}".format(len(fallbacks), len(submission.answers)))
    print("Disagreements: {}/{}".format(len(disagreements), len(submission.answers)))
    for answer in disagreements:
        print("  {}/{}".format(answer.scenario_id, answer.covenant_id))

