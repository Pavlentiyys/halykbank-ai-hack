from pathlib import Path

from cli.__main__ import build_parser


def test_run_defaults_to_private_dataset() -> None:
    args = build_parser().parse_args(["run"])

    assert args.dataset_mode == "private"
    assert args.input is None


def test_run_public_selects_public_dataset() -> None:
    args = build_parser().parse_args(["run", "--public"])

    assert args.dataset_mode == "public"
    assert args.input is None


def test_run_private_selects_private_dataset() -> None:
    args = build_parser().parse_args(["run", "--private"])

    assert args.dataset_mode == "private"
    assert args.input is None


def test_run_accepts_custom_dataset() -> None:
    args = build_parser().parse_args(["run", "--input", "data/custom"])

    assert args.input == Path("data/custom")
