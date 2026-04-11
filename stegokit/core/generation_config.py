from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    precision: int = 52
    stop_on_eos: bool | None = None
    max_new_tokens: int = 128
