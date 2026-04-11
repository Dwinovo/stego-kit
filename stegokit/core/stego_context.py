from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from stegokit.core.algorithm_config import NoConfig
from stegokit.core.algorithm_enum import StegoAlgorithm
from stegokit.core.generation_config import GenerationConfig
from stegokit.core.runtime_context import RuntimeContext
from stegokit.core.security_material import NoMaterial


@dataclass(slots=True)
class StegoEncodeContext:
    algorithm: StegoAlgorithm | str
    runtime: RuntimeContext
    secret_bits: str
    config: Any = field(default_factory=NoConfig)
    material: Any = field(default_factory=NoMaterial)

    @property
    def model(self):
        return self.runtime.model

    @property
    def tokenizer(self):
        return self.runtime.tokenizer

    @property
    def messages(self) -> list[dict[str, Any]]:
        return self.runtime.messages

    @property
    def generation(self) -> GenerationConfig:
        return self.runtime.generation

    @property
    def temperature(self) -> float:
        return self.runtime.generation.temperature

    @property
    def top_k(self) -> int | None:
        return self.runtime.generation.top_k

    @property
    def top_p(self) -> float | None:
        return self.runtime.generation.top_p

    @property
    def precision(self) -> int:
        return self.runtime.generation.precision

    @property
    def stop_on_eos(self) -> bool | None:
        return self.runtime.generation.stop_on_eos

    @property
    def max_new_tokens(self) -> int:
        return self.runtime.generation.max_new_tokens


@dataclass(slots=True)
class StegoDecodeContext:
    algorithm: StegoAlgorithm | str
    runtime: RuntimeContext
    generated_token_ids: Sequence[int]
    max_bits: int | None = None
    config: Any = field(default_factory=NoConfig)
    material: Any = field(default_factory=NoMaterial)

    @property
    def model(self):
        return self.runtime.model

    @property
    def tokenizer(self):
        return self.runtime.tokenizer

    @property
    def messages(self) -> list[dict[str, Any]]:
        return self.runtime.messages

    @property
    def generation(self) -> GenerationConfig:
        return self.runtime.generation

    @property
    def temperature(self) -> float:
        return self.runtime.generation.temperature

    @property
    def top_k(self) -> int | None:
        return self.runtime.generation.top_k

    @property
    def top_p(self) -> float | None:
        return self.runtime.generation.top_p

    @property
    def precision(self) -> int:
        return self.runtime.generation.precision
