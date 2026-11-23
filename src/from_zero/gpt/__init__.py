"""A small decoder-only transformer language model in PyTorch."""

from __future__ import annotations

from .attention import CausalSelfAttention
from .dataset import TextCorpus, get_batch, load_text_corpus
from .generation import generate, generate_text
from .model import GPTModelConfig, MiniGPT
from .tokenizer import CharTokenizer
from .training import GPTRun, GPTTrainingConfig, load_gpt_checkpoint, train_gpt

__all__ = [
    "CausalSelfAttention",
    "CharTokenizer",
    "GPTModelConfig",
    "GPTRun",
    "GPTTrainingConfig",
    "MiniGPT",
    "TextCorpus",
    "generate",
    "generate_text",
    "get_batch",
    "load_gpt_checkpoint",
    "load_text_corpus",
    "train_gpt",
]
