from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from core.stego_context import DecodeContext, EncodeContext


@dataclass
class EncodeResult:
    sampled_token_id: int | None = None
    bits_consumed: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DecodeResult:
    bits: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class StegoStrategy(Protocol):
    def encode(self, context: EncodeContext) -> EncodeResult:
        ...

    def decode(self, context: DecodeContext) -> DecodeResult:
        ...
