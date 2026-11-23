"""Experiment records: config, environment, metrics, notes, and artifacts.

A run directory is intentionally simple and inspectable:

``artifacts/experiments/<name>/``
    ``config.yaml``, ``metadata.json``, ``metrics.csv``, ``metrics.json``,
    ``notes.md``, ``plots/``, ``samples/``, ``checkpoint/``.
"""

from __future__ import annotations

import csv
import json
import platform
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .paths import experiments_dir, project_root


@dataclass
class ExperimentResult:
    """The serialisable result of one experiment run."""

    name: str
    directory: str
    config: dict[str, Any]
    metrics: dict[str, Any]
    history: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(asdict(self))


class ExperimentRecorder:
    """Create and update a local, human-readable experiment record."""

    def __init__(
        self,
        name: str,
        config: dict[str, Any] | None = None,
        *,
        root: str | Path | None = None,
        seed: int | None = None,
        notes: str = "",
        unique: bool = False,
    ) -> None:
        self.name = _slugify(name)
        base = Path(root) if root is not None else experiments_dir()
        directory = base / self.name
        if unique:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            directory = base / f"{self.name}_{stamp}"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "plots").mkdir(exist_ok=True)
        (directory / "samples").mkdir(exist_ok=True)
        (directory / "checkpoint").mkdir(exist_ok=True)

        self.directory = directory
        self.config = to_jsonable(config or {})
        self.history: list[dict[str, Any]] = []
        self.artifacts: dict[str, str] = {}
        self.notes = notes
        self.started_at = _utc_now()
        self.metadata: dict[str, Any] = {
            "name": self.name,
            "run_id": self.name if not unique else directory.name,
            "created_at": self.started_at,
            "git": git_info(),
            "environment": environment_info(),
            "seed": seed,
            "command": " ".join(sys.argv),
            "cwd": str(Path.cwd()),
            "known_limitations": [],
        }
        self._write_config()
        self._write_metadata()

    @property
    def checkpoint_dir(self) -> Path:
        path = self.directory / "checkpoint"
        path.mkdir(exist_ok=True)
        return path

    def log(self, metrics: dict[str, Any], *, step: int | None = None) -> None:
        row = to_jsonable(metrics)
        if step is not None:
            row.setdefault("step", int(step))
        self.history.append(row)
        self._write_metrics_csv()

    def log_many(
        self,
        rows: list[dict[str, Any]],
        *,
        start_step: int | None = None,
    ) -> None:
        """Append several history rows with one CSV rewrite."""

        for offset, metrics in enumerate(rows):
            row = to_jsonable(metrics)
            if start_step is not None:
                row.setdefault("step", int(start_step) + offset)
            self.history.append(row)
        self._write_metrics_csv()

    def log_epoch(self, epoch: int, metrics: dict[str, Any]) -> None:
        self.log(metrics, step=epoch)

    def save_figure(self, figure: Any, filename: str, *, dpi: int = 150) -> str:
        from .plotting import save_figure

        destination = self.directory / "plots" / filename
        save_figure(figure, destination, dpi=dpi)
        return self._relative(destination)

    def save_text(self, text: str, filename: str) -> str:
        destination = self.directory / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8")
        return self._relative(destination)

    def save_json(self, payload: Any, filename: str) -> str:
        destination = self.directory / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return self._relative(destination)

    def save_npz(self, filename: str, **arrays: Any) -> str:
        destination = self.directory / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez(destination, **{key: np.asarray(value) for key, value in arrays.items()})
        return self._relative(destination)

    def finish(
        self,
        metrics: dict[str, Any] | None = None,
        *,
        notes: str | None = None,
        limitations: list[str] | None = None,
    ) -> ExperimentResult:
        if metrics is not None:
            self.save_json(metrics, "metrics.json")
        if notes is not None:
            self.notes = notes
        if limitations:
            self.metadata["known_limitations"] = limitations
        self.metadata["finished_at"] = _utc_now()
        self.metadata["history_points"] = len(self.history)
        self.metadata["artifacts"] = dict(self.artifacts)
        self._write_metrics_csv()
        self._write_metadata()
        self.save_text(self.notes, "notes.md")
        if metrics is not None:
            self.save_json(metrics, "metrics.json")
        return ExperimentResult(
            name=self.name,
            directory=str(self.directory),
            config=self.config,
            metrics=to_jsonable(metrics or {}),
            history=to_jsonable(self.history),
            metadata=to_jsonable(self.metadata),
            artifacts=dict(self.artifacts),
            notes=self.notes,
        )

    def _write_config(self) -> None:
        path = self.directory / "config.yaml"
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.config, handle, sort_keys=False)

    def _write_metadata(self) -> None:
        path = self.directory / "metadata.json"
        path.write_text(
            json.dumps(self.metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def _write_metrics_csv(self) -> None:
        path = self.directory / "metrics.csv"
        if not self.history:
            path.write_text("", encoding="utf-8")
            return
        fieldnames: list[str] = []
        for row in self.history:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.history:
                writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})

    def _relative(self, path: Path) -> str:
        try:
            return str(path.relative_to(project_root()))
        except ValueError:
            return str(path)


def list_experiments(root: str | Path | None = None) -> list[dict[str, Any]]:
    """Return summaries of all experiment directories, newest first."""

    base = Path(root) if root is not None else experiments_dir()
    if not base.exists():
        return []
    summaries: list[dict[str, Any]] = []
    for directory in sorted(path for path in base.iterdir() if path.is_dir()):
        metadata_path = directory / "metadata.json"
        metrics_path = directory / "metrics.json"
        if not metadata_path.exists():
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        metrics: dict[str, Any] = {}
        if metrics_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                metrics = {}
        summaries.append(
            {
                "name": metadata.get("name", directory.name),
                "directory": str(directory),
                "created_at": metadata.get("created_at", ""),
                "config": _load_yaml(directory / "config.yaml"),
                "metrics": metrics,
                "metadata": metadata,
            }
        )
    summaries.sort(key=lambda item: item.get("created_at", ""), reverse=True)
    return summaries


def load_experiment(path_or_name: str | Path, root: str | Path | None = None) -> dict[str, Any]:
    """Load one experiment record by directory name or path."""

    base = Path(root) if root is not None else experiments_dir()
    candidate = Path(path_or_name)
    directory = candidate if candidate.is_dir() else base / str(path_or_name)
    if not directory.exists():
        raise FileNotFoundError(f"Experiment not found: {directory}")
    metadata = _load_json(directory / "metadata.json")
    metrics = _load_json(directory / "metrics.json")
    return {
        "name": metadata.get("name", directory.name),
        "directory": str(directory),
        "metadata": metadata,
        "metrics": metrics,
        "config": _load_yaml(directory / "config.yaml"),
        "notes": (directory / "notes.md").read_text(encoding="utf-8")
        if (directory / "notes.md").exists()
        else "",
        "history": _load_history(directory / "metrics.csv"),
    }


def git_info() -> dict[str, Any]:
    """Best-effort commit and dirty-state information."""

    info: dict[str, Any] = {"commit": None, "dirty": None, "branch": None}
    try:
        info["commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root(),
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
        info["branch"] = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=project_root(),
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=project_root(),
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout
        info["dirty"] = bool(status.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return info


def environment_info() -> dict[str, Any]:
    """Record the software and hardware environment without importing torch eagerly."""

    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "torch": None,
        "torch_cuda": None,
        "device": "cpu",
    }
    try:
        import scipy

        info["scipy"] = scipy.__version__
    except ImportError:
        pass
    try:
        import torch

        info["torch"] = torch.__version__
        info["torch_cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            info["device"] = "cuda"
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return info


def to_jsonable(value: Any) -> Any:
    """Recursively convert arrays, tensors, paths, and dataclasses to JSON-safe data."""

    if is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().tolist()
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", name.strip()).strip("-._")
    if not slug:
        raise ValueError("Experiment name must contain at least one alphanumeric character")
    return slug


def _csv_value(value: Any) -> Any:
    if isinstance(value, dict | list | tuple):
        return json.dumps(to_jsonable(value), sort_keys=True)
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data if isinstance(data, dict) else {}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _load_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]
