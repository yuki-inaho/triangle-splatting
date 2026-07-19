#!/usr/bin/env python3
"""Tune Triangle Splatting hyperparameters with fixed-pose holdout metrics.

This runner intentionally disables densification for every trial.  It is meant
for bounded, reproducible searches on existing COLMAP scenes where a dense
initial point cloud is already available.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import optuna


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EVAL_PATTERN = re.compile(
    r"Evaluating test: L1 (?P<l1>[-+0-9.eE]+) PSNR (?P<psnr>[-+0-9.eE]+) "
    r"SSIM (?P<ssim>[-+0-9.eE]+) LPIPS (?P<lpips>[-+0-9.eE]+)"
)
DISABLE_DENSIFICATION_ITERATION = 999_999


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a bounded Optuna search for a Triangle Splatting scene."
    )
    parser.add_argument("--source", type=Path, required=True, help="Staged COLMAP scene.")
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="New directory for study data."
    )
    parser.add_argument(
        "--tuning-seconds",
        type=int,
        default=900,
        help="Wall-clock budget for short Optuna trials (default: 900).",
    )
    parser.add_argument(
        "--max-trials", type=int, default=24, help="Hard trial cap (default: 24)."
    )
    parser.add_argument(
        "--trial-iterations",
        type=int,
        default=1000,
        help="Iterations per short trial (default: 1000).",
    )
    parser.add_argument(
        "--confirmation-iterations",
        type=int,
        default=2000,
        help="Iterations for each top-configuration confirmation run (default: 2000).",
    )
    parser.add_argument(
        "--top-k", type=int, default=3, help="Configurations to confirm (default: 3)."
    )
    parser.add_argument("--seed", type=int, default=0, help="Optuna sampler seed.")
    return parser.parse_args()


def require_new_directory(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"--output-dir must be new or empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def extract_metrics(output: str) -> dict[str, float]:
    matches = list(EVAL_PATTERN.finditer(output))
    if not matches:
        raise RuntimeError("Training completed without a parseable test evaluation.\n" + output[-4000:])
    return {key: float(value) for key, value in matches[-1].groupdict().items()}


def run_training(
    source: Path,
    model_path: Path,
    iterations: int,
    parameters: dict[str, float],
) -> dict[str, float]:
    command = [
        sys.executable,
        str(REPOSITORY_ROOT / "train.py"),
        "-s",
        str(source),
        "-m",
        str(model_path),
        "--no_dome",
        "--eval",
        "--iterations",
        str(iterations),
        "--test_iterations",
        str(iterations),
        "--save_iterations",
        str(iterations),
        "--densify_from_iter",
        str(DISABLE_DENSIFICATION_ITERATION),
        "--densify_until_iter",
        str(DISABLE_DENSIFICATION_ITERATION),
    ]
    for name, value in parameters.items():
        command.extend([f"--{name}", str(value)])

    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    (model_path / "train.log").write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"Training failed with exit code {completed.returncode}.\n{completed.stdout[-4000:]}")
    metrics = extract_metrics(completed.stdout)
    metrics["elapsed_seconds"] = time.monotonic() - started
    return metrics


def suggest_parameters(trial: optuna.Trial) -> dict[str, float]:
    return {
        "set_sigma": trial.suggest_float("set_sigma", 0.12, 0.9, log=True),
        "triangle_size": trial.suggest_float("triangle_size", 0.35, 2.0, log=True),
        "lr_triangles_points_init": trial.suggest_float(
            "lr_triangles_points_init", 0.0002, 0.002, log=True
        ),
        "lambda_dssim": trial.suggest_float("lambda_dssim", 0.0, 0.35),
    }


def trial_record(trial: optuna.trial.FrozenTrial) -> dict[str, Any]:
    return {
        "number": trial.number,
        "state": trial.state.name,
        "value": trial.value,
        "params": trial.params,
        "user_attrs": trial.user_attrs,
    }


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output_dir = args.output_dir.resolve()
    if not source.is_dir():
        raise ValueError(f"--source is not a directory: {source}")
    if args.tuning_seconds <= 0 or args.trial_iterations <= 0:
        raise ValueError("time and iteration limits must be positive")
    require_new_directory(output_dir)

    trials_dir = output_dir / "trials"
    trials_dir.mkdir()
    sampler = optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=6)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        parameters = suggest_parameters(trial)
        model_path = trials_dir / f"trial_{trial.number:03d}"
        model_path.mkdir()
        metrics = run_training(source, model_path, args.trial_iterations, parameters)
        for key, value in metrics.items():
            trial.set_user_attr(key, value)
        trial.set_user_attr("model_path", str(model_path))
        return metrics["psnr"]

    study.optimize(objective, n_trials=args.max_trials, timeout=args.tuning_seconds, gc_after_trial=True)
    completed_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if not completed_trials:
        raise RuntimeError("No Optuna trial completed successfully")

    ranked_trials = sorted(completed_trials, key=lambda trial: trial.value or float("-inf"), reverse=True)
    confirmation_dir = output_dir / "confirmations"
    confirmation_dir.mkdir()
    confirmations = []
    for rank, trial in enumerate(ranked_trials[: args.top_k], start=1):
        model_path = confirmation_dir / f"rank_{rank:02d}_trial_{trial.number:03d}"
        model_path.mkdir()
        metrics = run_training(source, model_path, args.confirmation_iterations, dict(trial.params))
        confirmations.append(
            {
                "rank_from_short_trials": rank,
                "trial_number": trial.number,
                "params": trial.params,
                "metrics": metrics,
                "model_path": str(model_path),
            }
        )

    best_confirmation = max(confirmations, key=lambda item: item["metrics"]["psnr"])
    best = {
        "selection_metric": "test_psnr",
        "densification": "disabled",
        "best_confirmation": best_confirmation,
        "short_trials": [trial_record(trial) for trial in ranked_trials],
        "confirmations": confirmations,
    }
    (output_dir / "best_config.json").write_text(json.dumps(best, indent=2) + "\n", encoding="utf-8")
    (output_dir / "study_summary.json").write_text(
        json.dumps(
            {
                "source": str(source),
                "tuning_seconds": args.tuning_seconds,
                "trial_iterations": args.trial_iterations,
                "confirmation_iterations": args.confirmation_iterations,
                "trials": [trial_record(trial) for trial in study.trials],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(best_confirmation, indent=2))


if __name__ == "__main__":
    main()
