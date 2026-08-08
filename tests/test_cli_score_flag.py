from pathlib import Path

from cli.__main__ import build_parser


def test_run_score_flag_uses_default_ground_truth() -> None:
    args = build_parser().parse_args(["run", "--score"])

    assert args.show_score is True
    assert args.key == Path("ground_truth.json")


def test_run_show_score_alias_and_custom_key() -> None:
    args = build_parser().parse_args(
        ["run", "--show-score", "--key", "data/private/ground_truth.json"]
    )

    assert args.show_score is True
    assert args.key == Path("data/private/ground_truth.json")
