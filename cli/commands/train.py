"""Prepare leakage-free SFT data and invoke MLX LoRA training."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from model import Settings
from training import prepare_training_data


def train_command(
    input_path: Path,
    data_path: Path,
    adapter_path: Path,
    model_id: str,
    iterations: int,
    settings: Settings,
) -> int:
    manifest = prepare_training_data(input_path, data_path, settings)
    print("Training data: {}".format(manifest["splits"]))
    adapter_path.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    command = [
        sys.executable,
        "-m",
        "mlx_lm",
        "lora",
        "--model",
        model_id,
        "--train",
        "--data",
        str(data_path),
        "--adapter-path",
        str(adapter_path),
        "--fine-tune-type",
        "lora",
        "--mask-prompt",
        "--batch-size",
        "1",
        "--num-layers",
        "4",
        "--iters",
        str(iterations),
        "--learning-rate",
        "1e-5",
        "--max-seq-length",
        "1024",
        "--steps-per-report",
        "1",
        "--steps-per-eval",
        "10",
        "--save-every",
        "10",
        "--grad-checkpoint",
    ]
    print("Starting MLX LoRA: {}".format(model_id))
    return subprocess.run(command, env=environment, check=False).returncode

