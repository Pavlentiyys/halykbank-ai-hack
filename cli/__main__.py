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
    _add_dataset_arguments(run_parser, with_input=True)
    run_parser.add_argument("--output", type=Path, default=Path("submission.json"))
    _add_runtime_arguments(run_parser)

    score_parser = subparsers.add_parser("score", help="score a submission")
    score_parser.add_argument("submission", nargs="?", type=Path, default=Path("submission.json"))
    _add_dataset_arguments(score_parser, with_input=False)
    score_parser.add_argument("--key", type=Path, help="explicit answer key path")

    validate_parser = subparsers.add_parser("validate", help="validate submission shape")
    validate_parser.add_argument("submission", nargs="?", type=Path, default=Path("submission.json"))
    _add_dataset_arguments(validate_parser, with_input=False)
    validate_parser.add_argument("--template", type=Path, help="explicit template path")

    inspect_parser = subparsers.add_parser("inspect", help="analyze one covenant")
    inspect_parser.add_argument("--input", type=Path, default=Path("private"))
    inspect_parser.add_argument("--scenario", required=True)
    inspect_parser.add_argument("--covenant", required=True)
    _add_runtime_arguments(inspect_parser)

    train_parser = subparsers.add_parser("train", help="prepare semantic SFT data and start MLX LoRA")
    train_parser.add_argument("--input", type=Path, default=Path("private"))
    train_parser.add_argument("--data", type=Path, default=Path("artifacts/training-data"))
    train_parser.add_argument("--adapter", type=Path, default=Path("artifacts/adapters/gemma-covenants"))
    train_parser.add_argument("--base-model", default="mlx-community/gemma-3-1b-it-4bit")
    train_parser.add_argument("--iters", type=int, default=50)
    _add_runtime_arguments(train_parser)
    return parser


def _add_dataset_arguments(parser: argparse.ArgumentParser, *, with_input: bool) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--public",
        dest="dataset_mode",
        action="store_const",
        const="public",
        help="use the public dataset, which ships its own answer key",
    )
    group.add_argument(
        "--private",
        dest="dataset_mode",
        action="store_const",
        const="private",
        help="use the private dataset (default)",
    )
    if with_input:
        group.add_argument("--input", type=Path, help="use a custom dataset directory")
    parser.set_defaults(dataset_mode="private")


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--gemma-model")
    parser.add_argument("--gemma-endpoint")
    parser.add_argument(
        "--llm-mode",
        choices=("gaps-only", "always"),
        help="call Gemma only for numeric gaps (default) or for every cell",
    )
    parser.add_argument("--workers", type=int)
    parser.add_argument("--fx-eur-usd")
    parser.add_argument("--team")
    parser.add_argument("--contact-email")
    parser.add_argument("--offline", action="store_true", help="use deterministic LLM substitute")
    parser.add_argument("--no-cache", action="store_true")


def dataset_file(args: argparse.Namespace, filename: str) -> Path:
    """Resolve a dataset file from the selected mode unless overridden explicitly."""
    return Path(args.dataset_mode) / filename


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "score":
        key_path = args.key or dataset_file(args, "ground_truth.json")
        if not key_path.exists():
            print(
                "Ключ {} не найден. У набора {} его нет — укажите --public или --key.".format(
                    key_path, args.dataset_mode
                )
            )
            return 2
        return score_command(args.submission, key_path)
    if args.command == "validate":
        template_path = args.template or dataset_file(args, "submission_template.json")
        return validate_command(args.submission, template_path)
    settings = load_settings(args)
    if args.command == "run":
        input_path = args.input or Path(args.dataset_mode)
        key_path = input_path / "ground_truth.json" if args.dataset_mode == "public" else None
        return run_command(
            input_path,
            args.output,
            settings,
            key_path=key_path,
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
