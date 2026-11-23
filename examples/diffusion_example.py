#!/usr/bin/env python3
"""
Diffusion Example

Demonstrates the PyTorch DDPM implementation:
- Noise schedules (linear, cosine)
- Forward noising process
- U-Net denoiser
- Noise prediction objective
- Reverse sampling
"""

import sys

sys.path.insert(0, "src")

import matplotlib.pyplot as plt
import torch

from from_zero.diffusion import (
    UNet,
    UNetConfig,
    make_schedule,
    make_synthetic_images,
    predict_x0_from_noise,
    q_sample,
    sample_ddpm,
)
from from_zero.diffusion.training import DiffusionTrainingConfig, train_diffusion


def test_schedules():
    """Test noise schedules."""
    print("=" * 60)
    print("Diffusion: Noise Schedules")
    print("=" * 60)

    # Linear schedule
    sched_linear = make_schedule("linear", timesteps=1000, beta_start=1e-4, beta_end=0.02)
    print(f"Linear: β₁={sched_linear.betas[0]:.6f}, β_T={sched_linear.betas[-1]:.4f}")
    print(f"  ᾱ₁={sched_linear.alpha_bars[0]:.6f}, ᾱ_T={sched_linear.alpha_bars[-1]:.6f}")

    # Cosine schedule
    sched_cosine = make_schedule("cosine", timesteps=1000, cosine_s=0.008)
    print(f"Cosine: β₁={sched_cosine.betas[0]:.6f}, β_T={sched_cosine.betas[-1]:.4f}")
    print(f"  ᾱ₁={sched_cosine.alpha_bars[0]:.6f}, ᾱ_T={sched_cosine.alpha_bars[-1]:.6f}")

    # Verify properties
    print("\nLinear schedule validations:")
    print(
        f"  β_t ∈ (0,1): {torch.all(sched_linear.betas > 0) and torch.all(sched_linear.betas < 1)}"
    )
    print(f"  ᾱ_t positive: {torch.all(sched_linear.alpha_bars > 0)}")
    print(
        f"  ᾱ_t non-increasing: {torch.all(sched_linear.alpha_bars[1:] <= sched_linear.alpha_bars[:-1])}"
    )
    print(f"  ᾱ_T = {sched_linear.alpha_bars[-1]:.6f}")


def test_forward_process():
    """Test forward noising process."""
    print("\n" + "=" * 60)
    print("Diffusion: Forward Process")
    print("=" * 60)

    schedule = make_schedule("linear", timesteps=1000)
    x0 = make_synthetic_images(1, image_size=32, seed=42)  # (1, 1, 32, 32)

    timesteps = [0, 100, 250, 500, 750, 999]
    noisy_images = []

    for t in timesteps:
        t_tensor = torch.tensor([t])
        noise = torch.randn_like(x0)
        xt = q_sample(x0, t_tensor, noise, schedule)
        noisy_images.append(xt[0, 0])

        alpha_bar = schedule.alpha_bars[t].item()
        snr = alpha_bar / (1 - alpha_bar)
        print(f"t={t:3d}: ᾱ={alpha_bar:.4f}, SNR={snr:.2f}")

    # Visualize
    _fig, axes = plt.subplots(1, len(timesteps), figsize=(15, 3))
    for ax, img, t in zip(axes, noisy_images, timesteps, strict=False):
        ax.imshow(img, cmap="gray", vmin=-1, vmax=1)
        ax.set_title(f"t = {t}")
        ax.axis("off")
    plt.suptitle("Forward Process: Clean → Noisy")
    plt.tight_layout()
    plt.savefig("/tmp/forward_process.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Forward process visualization saved to /tmp/forward_process.png")


def test_x0_recovery():
    """Test x₀ recovery from noise."""
    print("\n" + "=" * 60)
    print("Diffusion: x₀ Recovery")
    print("=" * 60)

    x0 = make_synthetic_images(1, image_size=32, seed=123)
    noise = torch.randn_like(x0)
    t = torch.tensor([500])
    schedule = make_schedule("cosine", timesteps=1000)

    xt = q_sample(x0, t, noise, schedule)
    x0_recovered = predict_x0_from_noise(xt, t, noise, schedule)

    mse = torch.mean((x0 - x0_recovered) ** 2).item()
    max_err = torch.max(torch.abs(x0 - x0_recovered)).item()
    print(f"MSE: {mse:.6f}")
    print(f"Max absolute error: {max_err:.6f}")

    # Visualize
    _fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    axes[0].imshow(x0[0, 0], cmap="gray", vmin=-1, vmax=1)
    axes[0].set_title("Original x₀")
    axes[1].imshow(xt[0, 0], cmap="gray", vmin=-1, vmax=1)
    axes[1].set_title("Noisy xₜ (t=500)")
    axes[2].imshow(x0_recovered[0, 0], cmap="gray", vmin=-1, vmax=1)
    axes[2].set_title("Recovered x₀")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("/tmp/x0_recovery.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("x₀ recovery visualization saved to /tmp/x0_recovery.png")


def test_denoiser():
    """Test U-Net denoiser."""
    print("\n" + "=" * 60)
    print("Diffusion: U-Net Denoiser")
    print("=" * 60)

    config = UNetConfig(
        in_channels=1,
        base_channels=32,
        channel_mults=(1, 2, 4),
        num_res_blocks=1,
        dropout=0.0,
    )
    model = UNet(config)
    print(f"Model parameters: {model.num_parameters():,}")

    # Forward pass
    images = torch.randn(2, 1, 32, 32)
    timesteps = torch.tensor([0, 500], dtype=torch.long)
    output = model(images, timesteps)
    print(f"Input shape: {images.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Shapes match: {output.shape == images.shape}")

    # Backward pass
    loss = output.square().mean()
    loss.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    print(f"Gradients flow: {has_grad}")


def test_training():
    """Run a tiny training run."""
    print("\n" + "=" * 60)
    print("Diffusion: Training (Tiny Run)")
    print("=" * 60)

    config = DiffusionTrainingConfig(
        name="example_diffusion_training",
        seed=42,
        dataset="synthetic",
        image_size=16,
        max_samples=32,
        timesteps=50,
        schedule="linear",
        beta_start=1e-4,
        beta_end=0.02,
        model=UNetConfig(
            in_channels=1,
            base_channels=16,
            channel_mults=(1, 2),
            num_res_blocks=1,
            dropout=0.0,
        ),
        batch_size=4,
        learning_rate=0.001,
        max_steps=20,
        eval_interval=10,
        eval_batches=2,
        log_interval=5,
        sample_count=4,
        sample_interval=10,
        device="cpu",
        output_dir="artifacts/experiments",
        make_plots=True,
    )

    print("Starting training...")
    run = train_diffusion(config)
    print(f"Training completed: {run.result.directory}")
    print(f"Best val loss: {run.best_val_loss:.4f}")

    # Check checkpoints
    import os

    ckpt_dir = os.path.join(run.result.directory, "checkpoint")
    if os.path.exists(ckpt_dir):
        print(f"Checkpoints: {os.listdir(ckpt_dir)}")


def test_sampling():
    """Test reverse sampling."""
    print("\n" + "=" * 60)
    print("Diffusion: Reverse Sampling")
    print("=" * 60)

    # Create a tiny trained model
    config = UNetConfig(
        in_channels=1,
        base_channels=16,
        channel_mults=(1, 2),
        num_res_blocks=1,
        dropout=0.0,
    )
    model = UNet(config).eval()
    schedule = make_schedule("linear", timesteps=50)

    # Sample from noise
    samples = sample_ddpm(
        model,
        (4, 1, 16, 16),
        schedule,
        device="cpu",
        generator=torch.Generator().manual_seed(42),
    )
    print(f"Sampled shape: {samples.shape}")
    print(f"Range: [{samples.min():.3f}, {samples.max():.3f}]")

    # Visualize
    _fig, axes = plt.subplots(1, 4, figsize=(10, 3))
    for i, ax in enumerate(axes):
        ax.imshow(samples[i, 0], cmap="gray", vmin=-1, vmax=1)
        ax.set_title(f"Sample {i}")
        ax.axis("off")
    plt.suptitle("DDPM Samples (from pure noise)")
    plt.tight_layout()
    plt.savefig("/tmp/ddpm_samples.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Samples saved to /tmp/ddpm_samples.png")


def test_reverse_process_visualization():
    """Visualize the reverse denoising process step by step."""
    print("\n" + "=" * 60)
    print("Diffusion: Reverse Process Visualization")
    print("=" * 60)

    from from_zero.diffusion import p_sample

    model = UNet(
        UNetConfig(
            in_channels=1,
            base_channels=16,
            channel_mults=(1, 2),
            num_res_blocks=1,
            dropout=0.0,
        )
    ).eval()
    schedule = make_schedule("linear", timesteps=50)

    # Start from pure noise
    x = torch.randn(1, 1, 16, 16, generator=torch.Generator().manual_seed(999))

    # Sample timesteps to visualize
    vis_steps = [49, 40, 30, 20, 10, 5, 1, 0]

    _fig, axes = plt.subplots(1, len(vis_steps), figsize=(20, 3))
    for ax, t in zip(axes, vis_steps, strict=False):
        x = p_sample(model, x, t, schedule)
        ax.imshow(x[0, 0], cmap="gray", vmin=-1, vmax=1)
        ax.set_title(f"t = {t}")
        ax.axis("off")

    plt.suptitle("Reverse Process: Noise → Image")
    plt.tight_layout()
    plt.savefig("/tmp/reverse_process.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Reverse process visualization saved to /tmp/reverse_process.png")


def test_schedule_identities():
    """Test mathematical identities of the schedule."""
    print("\n" + "=" * 60)
    print("Diffusion: Schedule Identities")
    print("=" * 60)

    for kind in ["linear", "cosine"]:
        s = make_schedule(kind, timesteps=100)
        # ᾱ_t = cumprod(α_s)
        ab_manual = torch.cumprod(s.alphas, dim=0)
        diff = torch.max(torch.abs(s.alpha_bars - ab_manual)).item()
        print(f"{kind}: ᾱ_t = cumprod(α) max diff = {diff:.2e}")

        # Posterior variance formula
        prev_ab = s.previous_alpha_bars
        expected_var = s.betas * (1 - prev_ab) / (1 - s.alpha_bars)
        expected_var[0] = 0.0
        diff_var = torch.max(torch.abs(s.posterior_variance - expected_var)).item()
        print(f"{kind}: posterior variance formula max diff = {diff_var:.2e}")


def main():
    test_schedules()
    test_forward_process()
    test_x0_recovery()
    test_denoiser()
    test_training()
    test_sampling()
    test_reverse_process_visualization()
    test_schedule_identities()

    print("\n" + "=" * 60)
    print("All diffusion examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
