"""Command-line entry point."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from .commands.inspect import inspect_command
from .commands.run import run_command
from .commands.score import score_command
from .commands.train import train_command
from .commands.validate import validate_command
from .settings import load_settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="analyze a complete dataset")
    run_parser.add_argument("--input", type=Path, default=Path("."))
    run_parser.add_argument("--output", type=Path, default=Path("submission.json"))
    run_parser.add_argument(
        "--score",
        "--show-score",
        dest="show_score",
        action="store_true",
        help="show submission similarity percentage against ground truth",
    )
    run_parser.add_argument("--key", type=Path, default=Path("ground_truth.json"))
    _add_runtime_arguments(run_parser)

    score_parser = subparsers.add_parser("score", help="score a submission")
    score_parser.add_argument("submission", nargs="?", type=Path, default=Path("submission.json"))
    score_parser.add_argument("--key", type=Path, default=Path("ground_truth.json"))

    validate_parser = subparsers.add_parser("validate", help="validate submission shape")
    validate_parser.add_argument("submission", nargs="?", type=Path, default=Path("submission.json"))
    validate_parser.add_argument("--template", type=Path, default=Path("submission_template.json"))

    inspect_parser = subparsers.add_parser("inspect", help="analyze one covenant")
    inspect_parser.add_argument("--input", type=Path, default=Path("."))
    inspect_parser.add_argument("--scenario", required=True)
    inspect_parser.add_argument("--covenant", required=True, choices=("6.1", "6.2", "6.3"))
    _add_runtime_arguments(inspect_parser)

    train_parser = subparsers.add_parser("train", help="prepare semantic SFT data and start MLX LoRA")
    train_parser.add_argument("--input", type=Path, default=Path("."))
    train_parser.add_argument("--data", type=Path, default=Path("artifacts/training-data"))
    train_parser.add_argument("--adapter", type=Path, default=Path("artifacts/adapters/gemma-covenants"))
    train_parser.add_argument("--base-model", default="mlx-community/gemma-3-1b-it-4bit")
    train_parser.add_argument("--iters", type=int, default=50)
    _add_runtime_arguments(train_parser)
    return parser


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--gemma-model")
    parser.add_argument("--gemma-endpoint")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--fx-eur-usd")
    parser.add_argument("--team")
    parser.add_argument("--contact-email")
    parser.add_argument("--offline", action="store_true", help="use deterministic LLM substitute")
    parser.add_argument("--no-cache", action="store_true")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "score":
        return score_command(args.submission, args.key)
    if args.command == "validate":
        return validate_command(args.submission, args.template)
    settings = load_settings(args)
    if args.command == "run":
        return run_command(
            args.input,
            args.output,
            settings,
            key_path=args.key if args.show_score else None,
        )
    if args.command == "inspect":
        return inspect_command(args.input, args.scenario, args.covenant, settings)
    if args.command == "train":
        return train_command(
            args.input,
            args.data,
            args.adapter,
            args.base_model,
            args.iters,
            settings,
        )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
