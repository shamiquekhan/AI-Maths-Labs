"""Learning-rate schedules operating on an optimizer's ``defaults['lr']``."""

from __future__ import annotations

import math

from .base import Optimizer


class LRScheduler:
    def __init__(self, optimizer: Optimizer, last_epoch: int = -1) -> None:
        self.optimizer = optimizer
        self.base_lr = float(optimizer.defaults["lr"])
        self.last_epoch = last_epoch
        self.step()

    def get_lr(self) -> float:
        raise NotImplementedError

    def step(self, epoch: int | None = None) -> float:
        self.last_epoch = self.last_epoch + 1 if epoch is None else int(epoch)
        value = float(self.get_lr())
        if value < 0:
            raise ValueError("Learning rate schedule produced a negative value")
        self.optimizer.defaults["lr"] = value
        return value

    def state_dict(self) -> dict:
        return {
            "base_lr": self.base_lr,
            "last_epoch": self.last_epoch,
            "type": type(self).__name__,
        }

    def load_state_dict(self, state: dict) -> None:
        self.base_lr = float(state.get("base_lr", self.base_lr))
        self.last_epoch = int(state.get("last_epoch", self.last_epoch))
        self.optimizer.defaults["lr"] = float(self.get_lr())


class ConstantSchedule(LRScheduler):
    def __init__(self, optimizer: Optimizer, lr: float | None = None, last_epoch: int = -1) -> None:
        if lr is not None:
            optimizer.defaults["lr"] = float(lr)
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> float:
        return self.base_lr


class StepDecay(LRScheduler):
    def __init__(
        self, optimizer: Optimizer, step_size: int, gamma: float = 0.1, last_epoch: int = -1
    ) -> None:
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> float:
        return self.base_lr * (self.gamma ** (self.last_epoch // self.step_size))


class ExponentialDecay(LRScheduler):
    def __init__(self, optimizer: Optimizer, gamma: float, last_epoch: int = -1) -> None:
        if gamma <= 0:
            raise ValueError("gamma must be positive")
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> float:
        return self.base_lr * (self.gamma**self.last_epoch)


class CosineDecay(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        T_max: int,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        if T_max <= 0:
            raise ValueError("T_max must be positive")
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> float:
        progress = min(self.last_epoch / self.T_max, 1.0)
        return (
            self.eta_min
            + (self.base_lr - self.eta_min) * (1.0 + math.cos(math.pi * progress)) / 2.0
        )


class LinearWarmup(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int | None = None,
        start_lr: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.start_lr = start_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> float:
        if self.warmup_steps and self.last_epoch < self.warmup_steps:
            progress = self.last_epoch / self.warmup_steps
            return self.start_lr + (self.base_lr - self.start_lr) * progress
        if self.total_steps and self.last_epoch >= self.total_steps:
            return self.start_lr
        return self.base_lr


class PolynomialDecay(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        power: float = 1.0,
        end_lr: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        if total_steps <= 0:
            raise ValueError("total_steps must be positive")
        self.total_steps = total_steps
        self.power = power
        self.end_lr = end_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> float:
        if self.last_epoch >= self.total_steps:
            return self.end_lr
        progress = self.last_epoch / self.total_steps
        return (self.base_lr - self.end_lr) * (1.0 - progress) ** self.power + self.end_lr


# Backwards-compatible aliases
CosineAnnealingWarmRestarts = CosineDecay


__all__ = [
    "ConstantSchedule",
    "CosineAnnealingWarmRestarts",
    "CosineDecay",
    "ExponentialDecay",
    "LRScheduler",
    "LinearWarmup",
    "PolynomialDecay",
    "StepDecay",
]
