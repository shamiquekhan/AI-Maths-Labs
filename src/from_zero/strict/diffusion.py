"""DDPM diffusion with an MLP denoiser on the strict tensor core."""

from __future__ import annotations

import math
from collections.abc import Sequence

from ..nn_tensor import Linear, Module, ReLU
from ..optim_tensor import Adam
from ..tensor import Tensor, cat_last, mse_loss, no_grad
from ..tensor.storage import RNG


class DiffusionSchedule:
    """Linear beta schedule with consistent 0..T-1 indexing."""

    def __init__(self, betas: Sequence[float]) -> None:
        self.betas = [float(beta) for beta in betas]
        if not self.betas:
            raise ValueError("Schedule must contain at least one timestep")
        for beta in self.betas:
            if not 0.0 < beta < 1.0:
                raise ValueError(f"Betas must be in (0, 1), got {beta}")
        self.alphas = [1.0 - beta for beta in self.betas]
        self.alpha_bars: list[float] = []
        running = 1.0
        for alpha in self.alphas:
            running *= alpha
            self.alpha_bars.append(running)

    @classmethod
    def linear(
        cls, steps: int, beta_start: float = 1e-4, beta_end: float = 0.02
    ) -> DiffusionSchedule:
        if steps <= 0:
            raise ValueError("steps must be positive")
        if not 0.0 < beta_start < beta_end < 1.0:
            raise ValueError("Require 0 < beta_start < beta_end < 1")
        if steps == 1:
            return cls([beta_start])
        step = (beta_end - beta_start) / (steps - 1)
        return cls([beta_start + step * index for index in range(steps)])

    @property
    def steps(self) -> int:
        return len(self.betas)

    @property
    def posterior_variance(self) -> list[float]:
        variances = [0.0]
        for t in range(1, self.steps):
            value = self.betas[t] * (1.0 - self.alpha_bars[t - 1]) / (1.0 - self.alpha_bars[t])
            variances.append(value)
        return variances

    def add_noise(self, clean: Tensor, timesteps, noise: Tensor) -> Tensor:
        """Closed form ``sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * eps``."""
        if clean.shape != noise.shape:
            raise ValueError(f"clean {clean.shape} and noise {noise.shape} must match")
        batch = clean.shape[0]
        if isinstance(timesteps, int):
            timesteps = [timesteps] * batch
        timesteps = [int(t) for t in timesteps]
        if len(timesteps) != batch:
            raise ValueError("timesteps must match the batch size")
        signal = []
        noise_scale = []
        for t in timesteps:
            if not 0 <= t < self.steps:
                raise IndexError(f"Timestep {t} out of range")
            alpha_bar = self.alpha_bars[t]
            signal.append([math.sqrt(alpha_bar)])
            noise_scale.append([math.sqrt(1.0 - alpha_bar)])
        return clean * Tensor(signal) + noise * Tensor(noise_scale)

    def reverse_mean(
        self, x_t: Sequence[float], t: int, predicted_noise: Sequence[float]
    ) -> list[float]:
        beta = self.betas[t]
        alpha = self.alphas[t]
        alpha_bar = self.alpha_bars[t]
        factor = beta / math.sqrt(1.0 - alpha_bar)
        return [
            (value - factor * noise) / math.sqrt(alpha)
            for value, noise in zip(x_t, predicted_noise, strict=False)
        ]


def sinusoidal_embedding(timestep: int, dimension: int) -> list[float]:
    """Sinusoidal timestep embedding (``dimension`` must be even)."""
    if dimension <= 0 or dimension % 2 != 0:
        raise ValueError("Embedding dimension must be a positive even number")
    values: list[float] = []
    half = dimension // 2
    for index in range(half):
        frequency = math.exp(-math.log(10000.0) * index / max(half - 1, 1))
        angle = timestep * frequency
        values.append(math.sin(angle))
        values.append(math.cos(angle))
    return values


class MLPDenoiser(Module):
    """Predicts the noise added to a flattened image at a given timestep."""

    def __init__(
        self,
        image_dim: int,
        time_dim: int = 8,
        hidden_dim: int = 32,
        generator: RNG | None = None,
    ) -> None:
        super().__init__()
        if image_dim <= 0 or hidden_dim <= 0:
            raise ValueError("Dimensions must be positive")
        if time_dim % 2 != 0:
            raise ValueError("time_dim must be even")
        self.image_dim = image_dim
        self.time_dim = time_dim
        self.fc1 = Linear(image_dim + time_dim, hidden_dim, generator=generator)
        self.activation = ReLU()
        self.fc2 = Linear(hidden_dim, hidden_dim, generator=generator)
        self.fc3 = Linear(hidden_dim, image_dim, generator=generator)

    def forward(self, x_t: Tensor, timesteps) -> Tensor:
        if isinstance(timesteps, int):
            timesteps = [timesteps]
        timesteps = [int(t) for t in timesteps]
        if x_t.ndim != 2 or x_t.shape[0] != len(timesteps):
            raise ValueError("x_t must be (batch, image_dim) and match timesteps")
        embeddings = Tensor([sinusoidal_embedding(t, self.time_dim) for t in timesteps])
        hidden = cat_last(x_t, embeddings)
        hidden = self.activation(self.fc1(hidden))
        hidden = self.activation(self.fc2(hidden))
        return self.fc3(hidden)

    def extra_repr(self) -> str:
        return f"image_dim={self.image_dim}, time_dim={self.time_dim}"


def train_diffusion(
    images: Sequence[Sequence[float]],
    steps: int = 200,
    batch_size: int = 8,
    learning_rate: float = 0.01,
    schedule_steps: int = 20,
    time_dim: int = 8,
    hidden_dim: int = 32,
    seed: int = 0,
    verbose: bool = False,
) -> tuple[MLPDenoiser, DiffusionSchedule, list[float]]:
    """Train the noise-prediction objective on a tiny image set."""
    if not images:
        raise ValueError("images must not be empty")
    image_dim = len(images[0])
    if any(len(image) != image_dim for image in images):
        raise ValueError("All images must have the same flattened dimension")

    schedule = DiffusionSchedule.linear(schedule_steps)
    rng = RNG(seed)
    model = MLPDenoiser(image_dim, time_dim=time_dim, hidden_dim=hidden_dim, generator=rng)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    losses: list[float] = []

    for step in range(steps):
        indices = [rng.randint(0, len(images) - 1) for _ in range(batch_size)]
        clean = Tensor([list(images[index]) for index in indices])
        timesteps = [rng.randint(0, schedule.steps - 1) for _ in range(batch_size)]
        noise = Tensor(
            [[rng.normal(0.0, 1.0) for _ in range(image_dim)] for _ in range(batch_size)]
        )
        noisy = schedule.add_noise(clean, timesteps, noise)
        optimizer.zero_grad()
        predicted = model(noisy, timesteps)
        loss = mse_loss(predicted, noise)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
        if verbose and (step % max(1, steps // 10) == 0):
            print(f"step {step:4d} loss {losses[-1]:.4f}")
    return model, schedule, losses


def sample_diffusion(
    model: MLPDenoiser,
    schedule: DiffusionSchedule,
    image_dim: int,
    seed: int = 0,
) -> tuple[list[float], list[list[float]]]:
    """Reverse diffusion; returns the final image and every intermediate frame."""
    rng = RNG(seed)
    x = [rng.normal(0.0, 1.0) for _ in range(image_dim)]
    frames = [list(x)]
    variances = schedule.posterior_variance
    model.eval()
    with no_grad():
        for t in range(schedule.steps - 1, -1, -1):
            predicted = model(Tensor([x]), [t])._values()
            mean = schedule.reverse_mean(x, t, predicted)
            if t > 0:
                standard_deviation = math.sqrt(max(variances[t], 0.0))
                x = [value + standard_deviation * rng.normal(0.0, 1.0) for value in mean]
            else:
                x = mean
            frames.append(list(x))
    return x, frames


def make_pattern_images(n_images: int = 8, size: int = 4, seed: int = 0) -> list[list[float]]:
    """Tiny structured grayscale images in [0, 1] for diffusion smoke tests."""
    if size <= 0:
        raise ValueError("size must be positive")
    rng = RNG(seed)
    images: list[list[float]] = []
    for _ in range(n_images):
        mode = rng.randint(0, 3)
        values: list[float] = []
        for row in range(size):
            for column in range(size):
                if mode == 0:
                    value = 1.0 if column > row else 0.1
                elif mode == 1:
                    value = 1.0 if column < row else 0.1
                elif mode == 2:
                    value = 1.0 if (row + column) % 2 == 0 else 0.1
                else:
                    value = 1.0 if row == size // 2 else 0.1
                values.append(value + rng.normal(0.0, 0.02))
        images.append([min(1.0, max(0.0, value)) for value in values])
    return images


__all__ = [
    "DiffusionSchedule",
    "MLPDenoiser",
    "make_pattern_images",
    "sample_diffusion",
    "sinusoidal_embedding",
    "train_diffusion",
]
