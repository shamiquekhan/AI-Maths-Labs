#!/usr/bin/env python
"""Train the scalar autodiff MLP on XOR from a YAML configuration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from from_zero.autodiff.training import AutodiffConfig, run_autodiff_experiment  # noqa: E402
from from_zero.core.validation import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/micrograd.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    raw = load_config(ROOT / args.config)
    if args.epochs is not None:
        raw.setdefault("model", {})["epochs"] = args.epochs
    run = run_autodiff_experiment(
        AutodiffConfig.from_dict(raw),
        output_root=str(ROOT / args.output_root) if args.output_root else None,
        make_plots=not args.no_plots,
    )
    print(
        json.dumps(
            {
                "experiment": run.config.name,
                "directory": run.result.directory,
                "final_loss": run.metrics["final_loss"],
                "accuracy": run.metrics["accuracy"],
                "epochs_run": run.metrics["epochs_run"],
                "solved": run.metrics["solved"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
