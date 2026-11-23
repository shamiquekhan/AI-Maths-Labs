"""Diffusion laboratory page."""

from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[3]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import streamlit as st  # noqa: E402

from from_zero.app import common  # noqa: E402
from from_zero.app.services import (  # noqa: E402
    run_diffusion_demo,
    sample_from_diffusion_checkpoint,
)

try:  # PyTorch is optional in the public deployment.
    import torch

    from from_zero.core.plotting import (
        plot_forward_diffusion,
        plot_image_grid,
        plot_noise_schedule,
    )
    from from_zero.diffusion.data import make_synthetic_images
    from from_zero.diffusion.process import q_sample
    from from_zero.diffusion.schedules import make_schedule

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised in the torch-free deployment
    TORCH_AVAILABLE = False

common.setup_page("Diffusion", "◌")
common.sidebar()

common.hero(
    "Noise in, structure out.",
    "A DDPM defines a fixed forward process that gradually destroys an image, then learns "
    "to reverse it one Gaussian step at a time. The model never sees a clean image during "
    "sampling; it starts from pure noise.",
    ("PyTorch", "DDPM", "Linear + cosine schedules", "MNIST"),
)

if not TORCH_AVAILABLE:
    common.callout(
        "This laboratory needs PyTorch.",
        "The public dashboard runs the strict from-scratch engine and the NumPy "
        "laboratories without PyTorch. Install the deep-learning extra to run the "
        "diffusion lab locally.",
    )
    st.code("pip install -e '.[deep-learning]'", language="bash")
    st.markdown(
        "The strict diffusion implementation (a tiny MLP denoiser on the custom tensor "
        "engine) is available on the **Strict Engine** page and needs no third-party "
        "numerical library."
    )
    st.stop()


def _save_figure(figure, name: str) -> str:
    destination = Path("artifacts/ui") / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150, bbox_inches="tight", facecolor="white")
    return str(destination)


def _deterministic_noise(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=generator)


learn, build, visualise, evidence = st.tabs(["Learn", "Build", "Visualise", "Evidence"])

with learn:
    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("### Forward process")
        st.markdown(
            "The forward process is not learned. At each step it adds a small amount of "
            "Gaussian noise controlled by $\\beta_t$:"
        )
        st.latex(r"q(x_t\mid x_{t-1}) = \mathcal{N}\left(\sqrt{1-\beta_t}x_{t-1}, \beta_t I\right)")
        st.markdown("Because Gaussians compose, we can jump directly to any timestep:")
        st.latex(
            r"x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon, "
            r"\qquad \bar\alpha_t = \prod_{s=1}^{t}(1-\beta_s)"
        )
        st.markdown("### Learning to denoise")
        st.markdown(
            "A U-Net receives the noisy image and the timestep, and predicts the noise that "
            "was added. The training target is a simple mean-squared error:"
        )
        st.latex(
            r"\mathcal{L} = \mathbb{E}_{t,x_0,\epsilon}"
            r"\left[\left\|\epsilon - \epsilon_\theta(x_t,t)\right\|_2^2\right]"
        )
        st.markdown("### Reverse process")
        st.latex(
            r"p_\theta(x_{t-1}\mid x_t) = \mathcal{N}\left(\mu_\theta(x_t,t), "
            r"\Sigma_\theta(x_t,t)\right)"
        )
        st.latex(
            r"\mu_\theta = \frac{1}{\sqrt{\alpha_t}}"
            r"\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)\right)"
        )
        st.markdown(
            "At $t=0$ the reverse step is deterministic: there is no $x_{-1}$ to noise. "
            "The implementation handles that boundary explicitly."
        )
    with right:
        st.markdown("### The compact U-Net")
        st.markdown(
            "- Two resolution levels with residual blocks.\n"
            "- Sinusoidal timestep embeddings injected into every block.\n"
            "- Skip connections between down and up paths.\n"
            "- Group normalization and SiLU activations.\n"
            "- A zero-initialized final convolution, so training starts near zero noise."
        )
        st.markdown("### Schedules")
        st.markdown(
            "The linear schedule is the original DDPM choice. The cosine schedule destroys "
            "signal more gently near the end of the chain, which often preserves structure "
            "longer. Both are implemented and validated."
        )
        common.callout(
            "Sampling is genuinely iterative.",
            "Generating one batch requires one model call per timestep — 1,000 calls for "
            "the saved MNIST model. The dashboard warns before starting and uses a small "
            "sample count.",
            tone="warn",
        )
        st.markdown("### What is not claimed")
        st.markdown(
            "- No FID, precision, or recall: a defensible evaluation pipeline is out of scope.\n"
            "- 3,000 training steps and 6,000 MNIST images are a teaching budget.\n"
            "- Sample quality is visibly below modern diffusion systems."
        )

with build:
    st.markdown("### Explore a noise schedule")
    col1, col2, col3 = st.columns(3)
    with col1:
        schedule_kind = st.selectbox("Schedule", ["linear", "cosine"])
    with col2:
        timesteps = st.slider("Diffusion timesteps", 100, 1000, 500, step=100)
    with col3:
        preview_seed = st.number_input("Preview seed", min_value=0, max_value=10_000, value=7)

    schedule = make_schedule(schedule_kind, timesteps=int(timesteps))
    common.figure(
        _save_figure(
            plot_noise_schedule(schedule, title=f"{schedule_kind.title()} schedule"),
            "ui_schedule.png",
        ),
        f"β, α, and ᾱ for the {schedule_kind} schedule",
    )

    st.markdown("### Forward noising of a synthetic shape")
    image = make_synthetic_images(1, image_size=28, seed=int(preview_seed))
    chosen = sorted({0, timesteps // 4, timesteps // 2, (3 * timesteps) // 4, timesteps - 1})
    noise = _deterministic_noise(image.shape, int(preview_seed))

    noisy = torch.stack(
        [
            q_sample(
                image,
                torch.tensor([step], dtype=torch.long),
                noise,
                schedule,
            )[0]
            for step in chosen
        ]
    )
    common.figure(
        _save_figure(
            plot_forward_diffusion(noisy, chosen, title="Forward process: structure becomes noise"),
            "ui_forward.png",
        ),
        "The same image at increasing timesteps",
    )

    st.divider()
    st.markdown("### Sample from a trained checkpoint")
    checkpoints = common.checkpoint_options("pytorch-ddpm-unet")
    if not checkpoints:
        st.info(
            "No DDPM checkpoint was found. Train the saved model with "
            "`python scripts/train_diffusion.py`, or run the short demo below."
        )
    else:
        with st.form("diffusion_sample_form"):
            checkpoint = st.selectbox("Checkpoint", checkpoints, index=0)
            col1, col2 = st.columns(2)
            with col1:
                count = st.slider("Images", 1, 16, 4)
            with col2:
                sample_seed = st.number_input(
                    "Sampling seed", min_value=0, max_value=10_000, value=42
                )
            st.caption("The reverse chain runs 1,000 model calls; this can take a while on CPU.")
            sample = st.form_submit_button("Run reverse diffusion", type="primary")
        if sample:
            with st.spinner("Running the reverse chain from pure noise…"):
                try:
                    st.session_state["diffusion_images"] = sample_from_diffusion_checkpoint(
                        checkpoint, count=int(count), seed=int(sample_seed)
                    )
                except Exception as error:
                    st.error(f"Sampling failed: {error}")
        images = st.session_state.get("diffusion_images")
        if images is not None:
            common.figure(
                _save_figure(
                    plot_image_grid(images, nrow=4, title="Samples from the checkpoint"),
                    "ui_samples.png",
                ),
                "Fresh samples generated from Gaussian noise",
            )

    st.divider()
    st.markdown("### Short dashboard training run")
    st.caption(
        "Synthetic 16×16 shapes, 200 diffusion steps, and a tiny U-Net. This demonstrates "
        "the training loop; it does not produce recognizable digits."
    )
    with st.form("diffusion_train_form"):
        col1, col2 = st.columns(2)
        with col1:
            max_steps = st.slider("Optimizer steps", 20, 300, 60, step=20)
        with col2:
            train_seed = st.number_input("Training seed", min_value=0, max_value=10_000, value=42)
        train = st.form_submit_button("Run demo training", type="primary")
    if train:
        with st.spinner("Training the compact DDPM on synthetic shapes…"):
            try:
                st.session_state["diffusion_run"] = run_diffusion_demo(
                    max_steps=int(max_steps), seed=int(train_seed)
                )
            except Exception as error:
                st.error(f"Demo training failed: {error}")
    run = st.session_state.get("diffusion_run")
    if run is not None:
        st.success(f"Finished in {run.result.directory}")
        col1, col2, col3 = st.columns(3)
        col1.metric("Best validation loss", f"{run.best_val_loss:.4f}")
        col2.metric("Parameters", f"{run.metrics['model']['parameters']:,}")
        col3.metric("Diffusion steps", str(run.config.timesteps))

with visualise:
    st.markdown("### Evidence from the saved MNIST run")
    common.figure(
        "artifacts/experiments/ddpm_mnist/plots/samples/final_samples.png",
        "Final samples from the 540,961-parameter U-Net",
    )
    col1, col2 = st.columns(2)
    with col1:
        common.figure(
            "artifacts/experiments/ddpm_mnist/plots/loss_curve.png",
            "Noise-prediction loss at evaluation points",
        )
    with col2:
        common.figure(
            "artifacts/experiments/ddpm_mnist/plots/forward_process.png",
            "Forward noising of a validation image",
        )
    common.figure(
        "artifacts/experiments/ddpm_mnist/plots/noise_schedule.png",
        "The linear schedule used for training",
    )
    st.markdown("### Sampling progress")
    progress = [
        ("artifacts/experiments/ddpm_mnist/plots/samples/step_001000.png", "Step 1,000"),
        ("artifacts/experiments/ddpm_mnist/plots/samples/step_002000.png", "Step 2,000"),
        ("artifacts/experiments/ddpm_mnist/plots/samples/step_003000.png", "Step 3,000"),
    ]
    columns = st.columns(3)
    for column, (path, caption) in zip(columns, progress, strict=True):
        with column:
            common.figure(path, caption)

with evidence:
    st.markdown("### Saved DDPM run")
    st.write(
        "The checkpoint stores U-Net weights, optimizer state, the diffusion schedule, and "
        "the data configuration. The reverse sampler rebuilds the schedule before sampling."
    )
    metrics = common.read_json("artifacts/experiments/ddpm_mnist/metrics.json")
    if metrics:
        st.json(
            {
                "dataset": metrics.get("dataset"),
                "model": metrics.get("model"),
                "schedule": metrics.get("schedule"),
                "training": metrics.get("training"),
            },
            expanded=False,
        )
    else:
        st.info("Run `python scripts/train_diffusion.py` to create the saved model.")
    notes = common.load_text("artifacts/experiments/ddpm_mnist/notes.md")
    if notes:
        st.markdown("#### Notes")
        st.markdown(notes)

common.footer()
