from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from core.algorithm_enum import StegoAlgorithm
from transformers import PreTrainedModel, PreTrainedTokenizerBase


@dataclass
class StegoEncodeContext:
    algorithm: StegoAlgorithm | str
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    secret_bits: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    max_new_tokens: int = 128
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    precision: int = 52
    prg: Any | None = None
    stop_on_eos: bool | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class StegoDecodeContext:
    algorithm: StegoAlgorithm | str
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    generated_token_ids: Sequence[int]
    messages: list[dict[str, Any]] = field(default_factory=list)
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    precision: int = 52
    prg: Any | None = None
    max_bits: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)
