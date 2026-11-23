"""A character-level tokenizer with an explicit, inspectable vocabulary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class CharTokenizer:
    """Map characters to integer IDs and back.

    Character tokenization keeps the first language model small and makes the
    vocabulary easy to inspect.  It is not an efficient production tokenizer.
    """

    def __init__(self, chars: list[str] | tuple[str, ...]) -> None:
        unique = tuple(dict.fromkeys(str(char) for char in chars))
        if not unique:
            raise ValueError("Vocabulary must contain at least one character")
        if any(len(char) != 1 for char in unique):
            raise ValueError("Every vocabulary item must be exactly one character")
        self.chars = unique
        self._stoi = {char: index for index, char in enumerate(self.chars)}

    @classmethod
    def from_text(cls, text: str) -> CharTokenizer:
        if not text:
            raise ValueError("Cannot build a tokenizer from empty text")
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str, *, strict: bool = True) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        unknown = sorted({char for char in text if char not in self._stoi})
        if unknown and strict:
            raise ValueError(
                f"Text contains characters outside the vocabulary: {unknown!r}. "
                "Use strict=False to skip them or extend the tokenizer."
            )
        return [self._stoi[char] for char in text if char in self._stoi]

    def decode(self, ids: list[int] | tuple[int, ...]) -> str:
        characters = []
        for token in ids:
            index = int(token)
            if index < 0 or index >= self.vocab_size:
                raise ValueError(f"Token id {index} is outside [0, {self.vocab_size})")
            characters.append(self.chars[index])
        return "".join(characters)

    def filter_text(self, text: str) -> str:
        return "".join(char for char in text if char in self._stoi)

    def to_dict(self) -> dict[str, Any]:
        return {"type": "char", "chars": list(self.chars)}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CharTokenizer:
        if not isinstance(payload, dict) or "chars" not in payload:
            raise ValueError("Tokenizer payload must contain a 'chars' list")
        chars = payload["chars"]
        if not isinstance(chars, list | tuple):
            raise ValueError("'chars' must be a list")
        return cls(chars)

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return destination

    @classmethod
    def load(cls, path: str | Path) -> CharTokenizer:
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {source}")
        return cls.from_dict(json.loads(source.read_text(encoding="utf-8")))

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:
        preview = "".join(char if char.isprintable() else repr(char) for char in self.chars[:20])
        return f"CharTokenizer(vocab_size={self.vocab_size}, preview={preview!r})"
