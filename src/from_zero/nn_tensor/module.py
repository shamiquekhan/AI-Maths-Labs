"""Module and Parameter abstractions for the custom neural-network engine."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from ..tensor import Tensor


class Parameter(Tensor):
    """A trainable tensor registered with its owning module."""

    def __init__(
        self,
        data: Any,
        requires_grad: bool = True,
        name: str | None = None,
    ) -> None:
        if isinstance(data, Tensor):
            values = data._values()
            shape = data.shape
        else:
            reference = Tensor(data)
            values = reference._values()
            shape = reference.shape
        super().__init__(values, shape=shape, requires_grad=requires_grad)
        self.name = name

    def __repr__(self) -> str:
        return f"Parameter(shape={self.shape}, requires_grad={self.requires_grad})"


class Module:
    """Base class for all neural-network modules."""

    def __init__(self) -> None:
        object.__setattr__(self, "_parameters", {})
        object.__setattr__(self, "_modules", {})
        object.__setattr__(self, "_training", True)

    # -- attribute registration -------------------------------------------
    def __setattr__(self, name: str, value: Any) -> None:
        parameters = self.__dict__.get("_parameters")
        modules = self.__dict__.get("_modules")
        if parameters is not None and modules is not None:
            if isinstance(value, Parameter):
                modules.pop(name, None)
                parameters[name] = value
            elif isinstance(value, Module):
                parameters.pop(name, None)
                modules[name] = value
            else:
                parameters.pop(name, None)
                modules.pop(name, None)
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        self.__dict__.get("_parameters", {}).pop(name, None)
        self.__dict__.get("_modules", {}).pop(name, None)
        object.__delattr__(self, name)

    def register_parameter(self, name: str, parameter: Parameter | None) -> None:
        if parameter is None:
            self._parameters.pop(name, None)
            object.__setattr__(self, name, None)
        elif not isinstance(parameter, Parameter):
            raise TypeError("register_parameter expects a Parameter or None")
        else:
            self._parameters[name] = parameter
            object.__setattr__(self, name, parameter)

    def add_module(self, name: str, module: Module | None) -> None:
        if module is None:
            self._modules.pop(name, None)
            object.__setattr__(self, name, None)
        elif not isinstance(module, Module):
            raise TypeError("add_module expects a Module or None")
        else:
            self._modules[name] = module
            object.__setattr__(self, name, module)

    # -- calling -----------------------------------------------------------
    def forward(self, *inputs: Any) -> Any:
        raise NotImplementedError(f"{type(self).__name__}.forward is not implemented")

    def __call__(self, *inputs: Any) -> Any:
        return self.forward(*inputs)

    # -- parameter discovery ----------------------------------------------
    def named_parameters(self, prefix: str = "") -> Iterator[tuple[str, Parameter]]:
        for name, parameter in self._parameters.items():
            full = f"{prefix}.{name}" if prefix else name
            yield full, parameter
        for name, module in self._modules.items():
            full = f"{prefix}.{name}" if prefix else name
            yield from module.named_parameters(full)

    def parameters(self) -> list[Parameter]:
        seen = set()
        result: list[Parameter] = []
        for _, parameter in self.named_parameters():
            if id(parameter) not in seen:
                seen.add(id(parameter))
                result.append(parameter)
        return result

    def modules(self) -> Iterator[Module]:
        seen = set()

        def walk(module: Module):
            if id(module) in seen:
                return
            seen.add(id(module))
            yield module
            for child in module._modules.values():
                yield from walk(child)

        yield from walk(self)

    def children(self) -> Iterator[Module]:
        yield from self._modules.values()

    # -- modes -------------------------------------------------------------
    def train(self, mode: bool = True) -> Module:
        object.__setattr__(self, "_training", bool(mode))
        for module in self._modules.values():
            module.train(mode)
        return self

    def eval(self) -> Module:
        return self.train(False)

    @property
    def training(self) -> bool:
        return self._training

    # -- gradients ---------------------------------------------------------
    def zero_grad(self) -> None:
        for parameter in self.parameters():
            parameter.grad = None

    # -- state -------------------------------------------------------------
    def state_dict(self) -> dict[str, Tensor]:
        return {name: parameter.clone() for name, parameter in self.named_parameters()}

    def load_state_dict(self, state: dict[str, Tensor], strict: bool = True) -> None:
        own = dict(self.named_parameters())
        missing = [name for name in own if name not in state]
        unexpected = [name for name in state if name not in own]
        if strict and (missing or unexpected):
            raise KeyError(f"State dict mismatch. Missing={missing}, unexpected={unexpected}")
        for name, parameter in own.items():
            if name not in state:
                continue
            value = state[name]
            values = value._values() if isinstance(value, Tensor) else list(value)
            if len(values) != parameter.size:
                raise ValueError(
                    f"Shape mismatch for {name}: expected {parameter.size} values, got {len(values)}"
                )
            parameter._data[parameter._offset : parameter._offset + parameter.size] = values

    def to(self, device: str) -> Module:
        return self

    def cpu(self) -> Module:
        return self

    def cuda(self) -> Module:
        raise RuntimeError("The FROM ZERO engine is CPU-only")

    def extra_repr(self) -> str:
        return ""

    def __repr__(self) -> str:
        lines = [f"{type(self).__name__}({self.extra_repr()})"]
        for name, module in self._modules.items():
            rendered = repr(module).splitlines()
            lines.append(f"  ({name}): {rendered[0]}")
            lines.extend("  " + line for line in rendered[1:])
        return "\n".join(lines)


__all__ = ["Module", "Parameter"]
