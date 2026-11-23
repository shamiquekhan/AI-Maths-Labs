"""Teaching examples for the scalar autodiff laboratory."""

from __future__ import annotations

from dataclasses import dataclass, field

from .nn import MLP
from .operations import binary_cross_entropy_with_logits
from .optim import SGD
from .value import Value


def xor_dataset() -> list[tuple[tuple[float, float], int]]:
    """The classic four-point XOR problem (not linearly separable)."""

    return [
        ((0.0, 0.0), 0),
        ((0.0, 1.0), 1),
        ((1.0, 0.0), 1),
        ((1.0, 1.0), 0),
    ]


def demo_expression() -> Value:
    """A small shared-subexpression graph used in documentation and the app."""

    a = Value(2.0)
    b = Value(-3.0)
    c = Value(10.0)
    f = Value(-2.0)
    e = a * b
    d = e + c
    loss = d * f
    loss.backward()
    return loss


@dataclass
class XORTrainingResult:
    model: MLP
    history: list[dict[str, float]] = field(default_factory=list)
    final_loss: float = float("nan")
    accuracy: float = 0.0
    epochs_run: int = 0
    solved: bool = False


def train_xor(
    hidden_sizes: tuple[int, ...] = (4, 4),
    *,
    epochs: int = 2000,
    learning_rate: float = 0.1,
    seed: int = 42,
    target_loss: float = 0.02,
    record_every: int = 10,
) -> XORTrainingResult:
    """Train a scalar MLP on XOR and record the loss trajectory."""

    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    data = xor_dataset()
    model = MLP(
        n_inputs=2,
        hidden_sizes=hidden_sizes,
        n_outputs=1,
        activation="tanh",
        output_activation="linear",
        seed=seed,
    )
    optimizer = SGD(model.parameters(), learning_rate=learning_rate)
    history: list[dict[str, float]] = []
    final_loss = float("nan")
    accuracy = 0.0
    solved = False

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        total_loss = Value(0.0)
        for features, target in data:
            logit = model(list(features))
            assert isinstance(logit, Value)
            total_loss = total_loss + binary_cross_entropy_with_logits(logit, target)
        loss = total_loss / float(len(data))
        loss.backward()
        optimizer.step()
        final_loss = loss.data
        accuracy = _xor_accuracy(model)
        if epoch == 1 or epoch % record_every == 0:
            history.append(
                {
                    "epoch": float(epoch),
                    "loss": float(loss.data),
                    "accuracy": float(accuracy),
                }
            )
        if final_loss <= target_loss and accuracy == 1.0:
            solved = True
            break

    if not history or history[-1]["epoch"] != float(epoch):
        history.append(
            {"epoch": float(epoch), "loss": float(final_loss), "accuracy": float(accuracy)}
        )
    return XORTrainingResult(
        model=model,
        history=history,
        final_loss=float(final_loss),
        accuracy=float(accuracy),
        epochs_run=epoch,
        solved=solved,
    )


def _xor_accuracy(model: MLP) -> float:
    correct = 0
    for features, target in xor_dataset():
        logit = model(list(features))
        assert isinstance(logit, Value)
        prediction = 1 if logit.data >= 0.0 else 0
        correct += int(prediction == target)
    return correct / 4.0
