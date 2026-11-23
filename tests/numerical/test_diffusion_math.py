"""Numerical identities for the diffusion schedules and forward process."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from from_zero.diffusion import (  # noqa: E402
    make_schedule,
    posterior_mean,
    posterior_variance,
    predict_noise_from_x0,
    predict_x0_from_noise,
    q_sample,
)


@pytest.mark.parametrize("kind", ["linear", "cosine"])
def test_schedule_alpha_bar_is_cumulative_product(kind: str) -> None:
    schedule = make_schedule(kind, timesteps=40)
    assert torch.allclose(schedule.alpha_bars, torch.cumprod(schedule.alphas, dim=0), atol=1e-6)
    assert torch.allclose(schedule.alphas, 1.0 - schedule.betas)
    previous = schedule.previous_alpha_bars
    assert previous.shape == schedule.alpha_bars.shape
    assert previous[0] == 1.0
    assert torch.allclose(previous[1:], schedule.alpha_bars[:-1])


def test_posterior_variance_matches_closed_form() -> None:
    schedule = make_schedule("linear", timesteps=20)
    previous = schedule.previous_alpha_bars
    expected = schedule.betas * (1.0 - previous) / (1.0 - schedule.alpha_bars)
    expected[0] = 0.0
    assert torch.allclose(schedule.posterior_variance, expected, atol=1e-7)
    timesteps = torch.arange(20)
    values = posterior_variance(timesteps, schedule)
    assert torch.allclose(values, expected)
    assert values[0] == 0.0


def test_forward_process_empirical_moments() -> None:
    torch.manual_seed(0)
    schedule = make_schedule("linear", timesteps=100)
    timestep = 50
    x0 = torch.full((20_000, 1, 1, 1), 0.5)
    timesteps = torch.full((20_000,), timestep, dtype=torch.long)
    noise = torch.randn_like(x0)
    xt = q_sample(x0, timesteps, noise, schedule)
    expected_mean = torch.sqrt(schedule.alpha_bars[timestep]) * 0.5
    expected_variance = 1.0 - schedule.alpha_bars[timestep]
    assert abs(float(xt.mean()) - float(expected_mean)) < 0.01
    assert abs(float(xt.var(unbiased=False)) - float(expected_variance)) < 0.01


def test_noise_and_clean_image_recovery_are_consistent() -> None:
    schedule = make_schedule("cosine", timesteps=50)
    x0 = torch.randn(4, 1, 8, 8)
    noise = torch.randn_like(x0)
    timesteps = torch.tensor([3, 10, 25, 49])
    xt = q_sample(x0, timesteps, noise, schedule)
    recovered_x0 = predict_x0_from_noise(xt, timesteps, noise, schedule)
    recovered_noise = predict_noise_from_x0(x0, xt, timesteps, schedule)
    # The identity is exact in real arithmetic.  At t=49 the cosine schedule has
    # alpha_bar ~ 1e-3, so recovering x0 divides by sqrt(alpha_bar) ~ 0.03 and
    # float32 rounding is amplified.  The noise recovery stays well-conditioned.
    assert torch.allclose(recovered_x0, x0, atol=1e-3)
    assert torch.allclose(recovered_noise, noise, atol=1e-5)


def test_posterior_mean_is_convex_combination_at_interior_steps() -> None:
    schedule = make_schedule("linear", timesteps=25)
    x0 = torch.randn(3, 1, 4, 4)
    noise = torch.randn_like(x0)
    timesteps = torch.tensor([1, 12, 24])
    xt = q_sample(x0, timesteps, noise, schedule)
    mean = posterior_mean(x0, xt, timesteps, schedule)
    alpha_bar = schedule.alpha_bars[timesteps].view(3, 1, 1, 1)
    alpha_bar_prev = schedule.previous_alpha_bars[timesteps].view(3, 1, 1, 1)
    beta = schedule.betas[timesteps].view(3, 1, 1, 1)
    alpha = schedule.alphas[timesteps].view(3, 1, 1, 1)
    coefficient_x0 = torch.sqrt(alpha_bar_prev) * beta / (1.0 - alpha_bar)
    coefficient_xt = torch.sqrt(alpha) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
    assert torch.all(coefficient_x0 > 0) and torch.all(coefficient_xt > 0)
    # Exact in real arithmetic; float32 rounding of 1 - alpha_bar leaves ~1e-5.
    assert torch.allclose(
        coefficient_x0 + coefficient_xt, torch.ones_like(coefficient_x0), atol=1e-3
    )
    assert torch.allclose(mean, coefficient_x0 * x0 + coefficient_xt * xt)


def test_sampling_boundary_has_no_added_noise() -> None:
    schedule = make_schedule("linear", timesteps=5)
    x0 = torch.randn(2, 1, 4, 4)
    noise = torch.randn_like(x0)
    timesteps = torch.zeros(2, dtype=torch.long)
    xt = q_sample(x0, timesteps, noise, schedule)
    mean = posterior_mean(x0, xt, timesteps, schedule)
    # At t=0, the posterior mean is x0 and the variance is exactly zero.
    assert torch.allclose(mean, x0, atol=1e-5)
    assert torch.all(posterior_variance(timesteps, schedule) == 0.0)
