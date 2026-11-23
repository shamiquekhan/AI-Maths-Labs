#!/usr/bin/env python
"""Train the compact DDPM from a YAML configuration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from from_zero.core.validation import load_config  # noqa: E402
from from_zero.diffusion.training import DiffusionTrainingConfig, train_diffusion  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ddpm_mnist.yaml")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    raw = load_config(ROOT / args.config)
    config = DiffusionTrainingConfig.from_dict(raw)
    run = train_diffusion(
        config,
        output_root=str(ROOT / args.output_root) if args.output_root else None,
        device=args.device,
        max_steps=args.max_steps,
        make_plots=not args.no_plots,
    )
    print(
        json.dumps(
            {
                "experiment": run.config.name,
                "directory": run.result.directory,
                "best_val_loss": run.best_val_loss,
                "parameters": run.metrics["model"]["parameters"],
                "checkpoint": run.artifacts.get("best_checkpoint"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
