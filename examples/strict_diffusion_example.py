#!/usr/bin/env python3
"""Strict diffusion: schedule mathematics, training, and reverse sampling."""

from __future__ import annotations

import sys

sys.path.insert(0, "src")

from from_zero.strict import (
    DiffusionSchedule,
    make_pattern_images,
    sample_diffusion,
    sinusoidal_embedding,
    train_diffusion,
)
from from_zero.tensor import Tensor


def main() -> None:
    print("=" * 66)
    print("FROM ZERO - strict diffusion (MLP denoiser)")
    print("=" * 66)

    print("\n1. The schedule")
    print("-" * 40)
    schedule = DiffusionSchedule.linear(20, beta_start=1e-4, beta_end=0.02)
    print(f"steps       {schedule.steps}")
    print(f"beta[0]     {schedule.betas[0]:.6f}")
    print(f"beta[-1]    {schedule.betas[-1]:.6f}")
    print(f"alpha_bar   {schedule.alpha_bars[0]:.6f} -> {schedule.alpha_bars[-1]:.6f}")
    print(f"posterior variance at t=1: {schedule.posterior_variance[1]:.8f}")

    print("\n2. Closed-form forward process")
    print("-" * 40)
    clean = Tensor([[1.0, -1.0]])
    noise = Tensor([[0.5, 0.5]])
    noisy = schedule.add_noise(clean, [10], noise)
    alpha_bar = schedule.alpha_bars[10]
    expected = [
        1.0 * (alpha_bar**0.5) + 0.5 * ((1.0 - alpha_bar) ** 0.5),
        -1.0 * (alpha_bar**0.5) + 0.5 * ((1.0 - alpha_bar) ** 0.5),
    ]
    print(f"x_t         {[round(value, 6) for value in noisy[0].tolist()]}")
    print(f"expected    {[round(value, 6) for value in expected]}")

    print("\n3. Timestep embedding")
    print("-" * 40)
    print(f"sinusoidal  {[round(value, 4) for value in sinusoidal_embedding(5, 8)]}")

    print("\n4. Training the denoiser")
    print("-" * 40)
    images = make_pattern_images(n_images=8, size=4, seed=0)
    model, schedule, losses = train_diffusion(
        images,
        steps=100,
        batch_size=8,
        learning_rate=0.01,
        schedule_steps=20,
        time_dim=8,
        hidden_dim=32,
        seed=0,
        verbose=True,
    )
    print(f"parameters  {sum(parameter.size for parameter in model.parameters()):,}")
    print(f"first loss  {losses[0]:.4f}")
    print(f"final loss  {losses[-1]:.4f}")

    print("\n5. Reverse sampling")
    print("-" * 40)
    image, frames = sample_diffusion(model, schedule, image_dim=16, seed=0)
    shades = " .:-=+*#%@"
    for row in range(4):
        values = image[row * 4 : (row + 1) * 4]
        line = "".join(
            shades[
                max(
                    0,
                    min(
                        len(shades) - 1,
                        int((value + 1.0) * 0.5 * len(shades)),
                    ),
                )
            ]
            for value in values
        )
        print(f"  |{line}|")
    print(f"reverse steps recorded: {len(frames)}")

    print("\nStrict diffusion example complete.")


if __name__ == "__main__":
    main()
