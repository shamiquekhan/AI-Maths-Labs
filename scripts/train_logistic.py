#!/usr/bin/env python
"""Train the NumPy logistic-regression laboratory from a YAML configuration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from from_zero.core.validation import load_config  # noqa: E402
from from_zero.logistic.training import LogisticConfig, run_logistic_experiment  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/logistic_regression.yaml",
        help="Path to a YAML experiment configuration.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override the epoch budget.")
    parser.add_argument("--dataset", default=None, help="Override the dataset name.")
    parser.add_argument("--output-root", default=None, help="Override the experiment root.")
    parser.add_argument("--no-plots", action="store_true", help="Skip figure generation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = load_config(ROOT / args.config)
    if args.epochs is not None:
        raw.setdefault("model", {})["epochs"] = args.epochs
    if args.dataset is not None:
        raw.setdefault("data", {})["dataset"] = args.dataset
    config = LogisticConfig.from_dict(raw)
    run = run_logistic_experiment(
        config,
        output_root=str(ROOT / args.output_root) if args.output_root else None,
        make_plots=not args.no_plots,
    )
    summary = {
        "experiment": run.config.name,
        "directory": run.result.directory,
        "test_accuracy": run.metrics["test"]["accuracy"],
        "test_loss": run.metrics["test"]["loss"],
        "confusion_matrix": run.metrics["test"]["confusion_matrix"],
        "artifacts": run.artifacts,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
