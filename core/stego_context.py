from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from core.algorithm_enum import StegoAlgorithm


@dataclass
class EncodeContext:
    algorithm: StegoAlgorithm | str
    prob_table: Sequence[float]
    indices: Sequence[int]
    bit_stream: str
    bit_index: int = 0
    precision: int = 52
    prg: Any | None = None
    cur_interval: list[int] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class DecodeContext:
    algorithm: StegoAlgorithm | str
    prob_table: Sequence[float]
    indices: Sequence[int]
    prev_token_id: int
    precision: int = 52
    prg: Any | None = None
    cur_interval: list[int] | None = None
    extra: dict[str, Any] = field(default_factory=dict)
