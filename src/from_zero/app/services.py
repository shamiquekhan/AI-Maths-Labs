"""Application services used by the dashboard.

The Streamlit pages call these functions instead of re-implementing training or
mathematics.  Each training entry point is deliberately bounded so a click in
the browser cannot start an unbounded job.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from from_zero.autodiff.training import AutodiffConfig, AutodiffRun, run_autodiff_experiment
from from_zero.core.experiment import list_experiments, load_experiment
from from_zero.core.paths import project_root
from from_zero.logistic.training import LogisticConfig, LogisticRun, run_logistic_experiment
from from_zero.optim.compare import (
    OptimizerComparison,
    OptimizerComparisonConfig,
    compare_optimizers,
)

if TYPE_CHECKING:
    from from_zero.diffusion.training import DiffusionRun
    from from_zero.gpt.training import GPTRun

try:
    import torch

    from from_zero.diffusion.model import UNetConfig
    from from_zero.diffusion.sampling import sample_ddpm
    from from_zero.diffusion.training import (
        DiffusionTrainingConfig,
        load_diffusion_checkpoint,
        train_diffusion,
    )
    from from_zero.gpt.generation import generate_text
    from from_zero.gpt.model import GPTModelConfig
    from from_zero.gpt.training import GPTTrainingConfig, load_gpt_checkpoint, train_gpt

    TORCH_AVAILABLE = True
    TORCH_IMPORT_ERROR: str | None = None
except ImportError as error:  # pragma: no cover - exercised in the torch-free deployment
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False
    TORCH_IMPORT_ERROR = str(error)


def torch_status() -> tuple[bool, str]:
    """Return whether PyTorch-backed labs are importable, with a reason."""

    if TORCH_AVAILABLE:
        return True, ""
    return False, TORCH_IMPORT_ERROR or "PyTorch is not installed"


def _require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "This laboratory requires PyTorch, which is not installed in this "
            "deployment. Install the deep-learning extra locally with "
            "`pip install -e '.[deep-learning]'` to run it. The strict from-scratch "
            "engine and the NumPy laboratories work without PyTorch."
        )


PROJECT_ROOT = project_root()
EXPERIMENT_ROOT = PROJECT_ROOT / "artifacts" / "experiments"


def run_logistic_demo(
    *,
    dataset: str = "linearly_separable",
    n_samples: int = 600,
    noise: float = 0.15,
    epochs: int = 400,
    learning_rate: float = 0.05,
    seed: int = 42,
) -> LogisticRun:
    config = LogisticConfig(
        name="ui_logistic_demo",
        seed=int(seed),
        dataset=dataset,
        n_samples=int(n_samples),
        noise=float(noise),
        epochs=int(epochs),
        learning_rate=float(learning_rate),
        output_dir=str(EXPERIMENT_ROOT),
        make_plots=True,
        notes="Interactive dashboard run. Bounded controls; not a tuned benchmark.",
    )
    return run_logistic_experiment(config)


def run_optimizer_demo(
    *,
    steps: int = 500,
    optimizers: tuple[str, ...] = ("sgd", "momentum", "rmsprop", "adam"),
    objectives: tuple[str, ...] = ("quadratic", "rosenbrock", "logistic"),
    seed: int = 42,
) -> OptimizerComparison:
    config = OptimizerComparisonConfig(
        name="ui_optimizer_demo",
        seed=int(seed),
        steps=int(steps),
        optimizers=tuple(optimizers),
        objectives=tuple(objectives),
        output_dir=str(EXPERIMENT_ROOT),
        make_plots=True,
        notes="Interactive dashboard comparison on small teaching objectives.",
    )
    return compare_optimizers(config)


def run_autodiff_demo(
    *,
    epochs: int = 4000,
    learning_rate: float = 0.2,
    hidden_sizes: tuple[int, ...] = (4, 4),
    seed: int = 42,
) -> AutodiffRun:
    config = AutodiffConfig(
        name="ui_autodiff_demo",
        seed=int(seed),
        hidden_sizes=tuple(hidden_sizes),
        epochs=int(epochs),
        learning_rate=float(learning_rate),
        output_dir=str(EXPERIMENT_ROOT),
        make_plots=True,
        notes="Interactive dashboard run of the scalar autodiff engine on XOR.",
    )
    return run_autodiff_experiment(config)


def run_strict_logistic_demo(
    *,
    epochs: int = 150,
    learning_rate: float = 0.5,
    seed: int = 7,
) -> dict[str, Any]:
    """Train the strict logistic model on a bounded synthetic dataset."""

    from from_zero.strict import (
        evaluate_logistic,
        fit_logistic_regression,
        make_classification,
    )

    features, targets = make_classification(
        n_samples=160, n_features=2, separation=3.0, noise=0.5, seed=int(seed)
    )
    model, history = fit_logistic_regression(
        features,
        targets,
        epochs=max(1, min(int(epochs), 400)),
        learning_rate=float(learning_rate),
        seed=int(seed),
    )
    report = evaluate_logistic(model, features, targets)
    return {
        "losses": list(history.train_loss),
        "accuracy": float(report["accuracy"]),
        "precision": float(report["precision"]),
        "recall": float(report["recall"]),
        "weights": [row[0] for row in model.weight.tolist()],
        "bias": model.bias.tolist()[0],
        "n_samples": len(features),
        "engine": "strict tensor engine (pure Python)",
    }


def run_strict_gpt_demo(*, steps: int = 12, seed: int = 3) -> dict[str, Any]:
    """Train a tiny character GPT for a bounded number of strict-core steps."""

    from from_zero.strict import GPTConfig, generate_text, train_gpt

    corpus = "the quick brown fox jumps over the lazy dog. " * 3
    config = GPTConfig(
        block_size=8,
        n_embd=8,
        n_head=2,
        n_layer=1,
        dropout=0.0,
        seed=int(seed),
    )
    model, tokenizer, losses = train_gpt(
        corpus,
        config=config,
        steps=max(1, min(int(steps), 40)),
        batch_size=4,
        learning_rate=5e-3,
        seed=int(seed),
    )
    sample = generate_text(
        model,
        tokenizer,
        "the ",
        max_new_tokens=24,
        temperature=0.8,
        seed=int(seed),
    )
    return {
        "losses": list(losses),
        "sample": sample,
        "vocab_size": tokenizer.vocab_size,
        "parameters": sum(parameter.size for parameter in model.parameters()),
        "engine": "strict tensor engine (pure Python)",
    }


def run_strict_diffusion_demo(*, steps: int = 60, seed: int = 2) -> dict[str, Any]:
    """Train and sample the strict MLP denoiser on tiny 4x4 patterns."""

    from from_zero.strict import (
        make_pattern_images,
        sample_diffusion,
        train_diffusion,
    )

    images = make_pattern_images(n_images=6, size=4, seed=int(seed))
    model, schedule, losses = train_diffusion(
        images,
        steps=max(1, min(int(steps), 120)),
        batch_size=6,
        learning_rate=0.01,
        schedule_steps=12,
        time_dim=8,
        hidden_dim=16,
        seed=int(seed),
    )
    sample, frames = sample_diffusion(model, schedule, image_dim=16, seed=int(seed))
    final = [min(1.0, max(0.0, value)) for value in sample]
    grid = [final[row * 4 : (row + 1) * 4] for row in range(4)]
    return {
        "losses": list(losses),
        "grid": grid,
        "frames": len(frames),
        "parameters": sum(parameter.size for parameter in model.parameters()),
        "engine": "strict tensor engine (pure Python)",
    }


def strict_gradient_report() -> list[dict[str, Any]]:
    """Compare analytic and central finite-difference gradients on fixed cases."""

    from from_zero.autograd_tensor import numerical_gradient
    from from_zero.tensor import Tensor

    cases = [
        ("x² + x", lambda x: (x * x + x).sum(), [0.3, -0.7, 1.1]),
        ("tanh(x) + sigmoid(x)", lambda x: (x.tanh() + x.sigmoid()).sum(), [0.2, -0.4, 0.9]),
        ("softplus(x)", lambda x: x.softplus().sum(), [0.0, 0.5, -0.5]),
        ("exp(x) / (1 + x²)", lambda x: (x.exp() / (x * x + 1.5)).sum(), [0.1, 0.6, -0.3]),
        (
            "softmax weighted",
            lambda x: (x.softmax() * Tensor([1.0, 2.0, 3.0])).sum(),
            [0.4, -0.2, 0.7],
        ),
    ]
    report: list[dict[str, Any]] = []
    for name, build, values in cases:
        tensor = Tensor._from_flat(values, (len(values),), requires_grad=True)
        output = build(tensor)
        output.backward()
        analytic = tensor.grad._values()
        numeric = numerical_gradient(lambda b=build, t=tensor: b(t), tensor)
        error = max(abs(left - right) for left, right in zip(analytic, numeric, strict=True))
        report.append(
            {
                "case": name,
                "analytic": analytic,
                "finite_difference": numeric,
                "max_error": error,
            }
        )
    return report


def run_gpt_demo(
    *,
    max_steps: int = 40,
    prompt: str = "\n",
    seed: int = 42,
) -> GPTRun:
    """A deliberately tiny GPT run so the browser can show the full pipeline."""

    _require_torch()
    config = GPTTrainingConfig(
        name="ui_gpt_demo",
        seed=int(seed),
        model=GPTModelConfig(
            vocab_size=0,
            context_length=64,
            embedding_dim=64,
            num_heads=4,
            num_layers=2,
            dropout=0.0,
        ),
        batch_size=8,
        learning_rate=3e-4,
        max_steps=int(max_steps),
        eval_interval=max(1, int(max_steps) // 2),
        eval_batches=2,
        log_interval=max(1, int(max_steps) // 4),
        warmup_steps=5,
        sample_prompt=prompt,
        sample_tokens=80,
        output_dir=str(EXPERIMENT_ROOT),
        make_plots=True,
        notes=(
            "Dashboard demo run. This is a very short training budget; the output is "
            "expected to be mostly noise and is not a quality claim."
        ),
    )
    return train_gpt(config)


def load_gpt_bundle(path: str | Path):
    _require_torch()
    return load_gpt_checkpoint(path, device="cpu")


def generate_gpt(
    checkpoint: str | Path,
    *,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.9,
    top_k: int = 40,
    seed: int = 42,
    greedy: bool = False,
) -> str:
    _require_torch()
    model, tokenizer, _ = load_gpt_checkpoint(checkpoint, device="cpu")
    return generate_text(
        model,
        tokenizer,
        prompt,
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_k=int(top_k),
        greedy=greedy,
        seed=int(seed),
        device="cpu",
    )


def run_diffusion_demo(
    *,
    max_steps: int = 60,
    image_size: int = 16,
    timesteps: int = 200,
    seed: int = 42,
) -> DiffusionRun:
    _require_torch()
    config = DiffusionTrainingConfig(
        name="ui_diffusion_demo",
        seed=int(seed),
        dataset="synthetic",
        image_size=int(image_size),
        max_samples=256,
        timesteps=int(timesteps),
        schedule="linear",
        model=UNetConfig(
            in_channels=1,
            base_channels=16,
            channel_mults=(1, 2),
            num_res_blocks=1,
            dropout=0.0,
        ),
        batch_size=16,
        learning_rate=2e-4,
        max_steps=int(max_steps),
        eval_interval=max(1, int(max_steps) // 2),
        eval_batches=2,
        log_interval=max(1, int(max_steps) // 3),
        sample_count=4,
        sample_interval=max(1, int(max_steps)),
        output_dir=str(EXPERIMENT_ROOT),
        make_plots=True,
        notes=(
            "Dashboard demo on deterministic synthetic shapes with a short budget. "
            "Samples demonstrate the reverse process, not image quality."
        ),
    )
    return train_diffusion(config)


def load_diffusion_bundle(path: str | Path):
    _require_torch()
    return load_diffusion_checkpoint(path, device="cpu")


def sample_from_diffusion_checkpoint(
    path: str | Path,
    *,
    count: int = 4,
    seed: int = 42,
) -> torch.Tensor:
    _require_torch()
    model, schedule, payload = load_diffusion_checkpoint(path, device="cpu")
    config = payload.get("config", {})
    data = config.get("data", {}) if isinstance(config, dict) else {}
    image_size = int(data.get("image_size", 28))
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return sample_ddpm(
        model,
        (int(count), model.config.in_channels, image_size, image_size),
        schedule,
        device="cpu",
        generator=generator,
    )


def experiment_catalog() -> list[dict[str, Any]]:
    return list_experiments(EXPERIMENT_ROOT)


def experiment_detail(name_or_path: str) -> dict[str, Any]:
    return load_experiment(name_or_path, root=EXPERIMENT_ROOT)


def find_checkpoints(model_family: str) -> list[Path]:
    """Find checkpoints whose experiment metadata names a model family.

    Results are sorted to prioritize full training runs over smoke/demo runs:
    1. Runs without "smoke" or "ui_" in the name (full training runs)
    2. Then by creation time (newest first)
    """

    import datetime

    results: list[tuple[int, float, Path]] = []  # (priority, created_at_timestamp, path)
    if not EXPERIMENT_ROOT.exists():
        return []
    for directory in EXPERIMENT_ROOT.iterdir():
        if not directory.is_dir():
            continue
        run_name = directory.name
        # Skip smoke tests and UI demo runs for primary listing
        is_full_run = ("smoke" not in run_name) and ("ui_" not in run_name)
        priority = 0 if is_full_run else 1
        metadata_path = directory / "metadata.json"
        family = ""
        created_at = 0.0
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                family = str(metadata.get("model_family", ""))
                created_str = metadata.get("created_at", "")
                try:
                    dt = datetime.datetime.fromisoformat(created_str.replace("Z", "+00:00"))
                    created_at = dt.timestamp()
                except ValueError:
                    created_at = 0.0
            except (json.JSONDecodeError, ValueError):
                family = ""
        if family != model_family:
            continue
        for name in ("best.pt", "last.pt"):
            candidate = directory / "checkpoint" / name
            if candidate.exists():
                results.append((priority, created_at, candidate))
    # Sort by priority (full runs first), then by created_at descending
    results.sort(key=lambda x: (x[0], -x[1]))
    return [path for _, _, path in results]


def artifact_path(experiment: str, relative: str) -> Path:
    return EXPERIMENT_ROOT / experiment / relative
