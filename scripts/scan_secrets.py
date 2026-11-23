#!/usr/bin/env python3
"""Fail if credential-like material appears in the repository.

This is a conservative, dependency-free scanner. It is not a replacement for a
full secret-scanning service, but it catches the common patterns that must never
reach a public repository: private keys, cloud tokens, and literal assignments
to secret-looking names.

Placeholders in documentation and ``.example`` files are ignored. The real
``.streamlit/secrets.toml`` must exist only locally and must never be tracked.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "node_modules",
    "artifacts",
    "data",
    "htmlcov",
    "dist",
    "build",
}

SKIP_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".ico",
    ".pdf",
    ".pt",
    ".pth",
    ".ckpt",
    ".npz",
    ".npy",
    ".woff",
    ".woff2",
    ".pyc",
    ".zip",
}

# Values containing these markers are documentation placeholders, not secrets.
ALLOW_MARKERS = (
    "example",
    "placeholder",
    "your-",
    "your_",
    "changeme",
    "change-me",
    "<",
    "xxx",
    "redacted",
    "dummy",
    "fake",
    "not-a-real",
    "invalid",
    "localhost",
    "test",
)

PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("private key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----")),
    ("AWS access key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b")),
    ("Slack token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b")),
    ("OpenAI-style key", re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")),
    ("Google API key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("JWT", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b")),
    (
        "secret-like assignment",
        re.compile(
            r"(?i)\b(?:api[_-]?key|secret|password|passwd|token|client[_-]?secret)\b"
            r"\s*[:=]\s*[\"']([^\"']{8,})[\"']"
        ),
    ),
)


def iter_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.relative_to(ROOT).parts):
            continue
        if path.suffix.lower() in SKIP_SUFFIXES:
            continue
        if path.stat().st_size > 2_000_000:
            continue
        files.append(path)
    return sorted(files)


def is_placeholder(value: str) -> bool:
    lowered = value.lower()
    return any(marker in lowered for marker in ALLOW_MARKERS)


def scan() -> list[str]:
    findings: list[str] = []
    for path in iter_files():
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        relative = path.relative_to(ROOT)
        for label, pattern in PATTERNS:
            for match in pattern.finditer(text):
                captured = match.group(1) if match.lastindex else match.group(0)
                if is_placeholder(captured):
                    continue
                line = text.count("\n", 0, match.start()) + 1
                findings.append(f"{relative}:{line}: {label}")

    secrets_file = ROOT / ".streamlit" / "secrets.toml"
    if secrets_file.exists():
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(secrets_file.relative_to(ROOT))],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if tracked.returncode == 0:
            findings.append(".streamlit/secrets.toml is tracked by git")
    return findings


def main() -> int:
    findings = scan()
    if findings:
        print("Secret scan FAILED:")
        for finding in findings:
            print(f"  {finding}")
        return 1
    print(f"Secret scan passed: no credential patterns in {len(iter_files())} files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
