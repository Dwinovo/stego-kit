from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence


class SupportsGenerateBits(Protocol):
    def generate_bits(self, n: int) -> Sequence[int]:
        ...


class SupportsGenerateRandom(Protocol):
    def generate_random(self, n: int) -> float:
        ...


@dataclass(frozen=True, slots=True)
class NoMaterial:
    pass


@dataclass(frozen=True, slots=True)
class RandomnessMaterial:
    prg: SupportsGenerateRandom | None = None


@dataclass(frozen=True, slots=True)
class BitMaskMaterial:
    prg: SupportsGenerateBits | None = None


def require_randomness_material(material: Any, strategy_name: str) -> SupportsGenerateRandom:
    if not isinstance(material, RandomnessMaterial):
        raise TypeError(f"{strategy_name} requires RandomnessMaterial")
    if material.prg is None or not hasattr(material.prg, "generate_random"):
        raise ValueError(f"{strategy_name} requires RandomnessMaterial.prg with generate_random(n)")
    return material.prg


def require_bitmask_material(material: Any, strategy_name: str) -> SupportsGenerateBits:
    if not isinstance(material, BitMaskMaterial):
        raise TypeError(f"{strategy_name} requires BitMaskMaterial")
    if material.prg is None or not hasattr(material.prg, "generate_bits"):
        raise ValueError(f"{strategy_name} requires BitMaskMaterial.prg with generate_bits(n)")
    return material.prg


def validate_material_instance(material: Any, expected_type: type[Any], strategy_name: str) -> None:
    if not isinstance(material, expected_type):
        raise TypeError(f"{strategy_name} requires material type {expected_type.__name__}")
    if isinstance(material, RandomnessMaterial):
        require_randomness_material(material, strategy_name)
    elif isinstance(material, BitMaskMaterial):
        require_bitmask_material(material, strategy_name)
