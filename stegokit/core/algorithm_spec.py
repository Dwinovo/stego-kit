from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .stego_algorithm import StegoStrategy
from .stego_paradigm import StegoParadigm


Validator = Callable[[Any], None]


@dataclass(frozen=True, slots=True)
class AlgorithmSpec:
    name: str
    paradigm: StegoParadigm
    strategy: StegoStrategy
    encode_config_type: type[Any]
    decode_config_type: type[Any]
    encode_material_type: type[Any]
    decode_material_type: type[Any]
    notes: str = ""
    encode_config_validator: Validator | None = None
    decode_config_validator: Validator | None = None
    encode_material_validator: Validator | None = None
    decode_material_validator: Validator | None = None
