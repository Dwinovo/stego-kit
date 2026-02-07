from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from core.stego_context import StegoDecodeContext, StegoEncodeContext


@dataclass
class StegoEncodeResult:
    generated_token_ids: list[int] = field(default_factory=list)
    consumed_bits: int = 0
    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StegoDecodeResult:
    bits: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class StegoStrategy(Protocol):
    def encode(self, context: StegoEncodeContext) -> StegoEncodeResult:
        ...

    def decode(self, context: StegoDecodeContext) -> StegoDecodeResult:
        ...
