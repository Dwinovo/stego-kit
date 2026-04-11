from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ACConfig:
    pass


@dataclass(frozen=True, slots=True)
class ADGConfig:
    epsilon: float = 0.01
    max_bit: int = 15

    def __post_init__(self) -> None:
        if self.epsilon <= 0:
            raise ValueError("ADGConfig.epsilon must be > 0")
        if self.max_bit < 1:
            raise ValueError("ADGConfig.max_bit must be >= 1")


@dataclass(frozen=True, slots=True)
class DiscopConfig:
    pass


@dataclass(frozen=True, slots=True)
class DiscopBaseConfig:
    pass


@dataclass(frozen=True, slots=True)
class FDPSSDifferentialBasedConfig:
    pass


@dataclass(frozen=True, slots=True)
class FDPSSBinaryBasedConfig:
    pass


@dataclass(frozen=True, slots=True)
class FDPSSStabilityBasedConfig:
    pass


@dataclass(frozen=True, slots=True)
class MeteorConfig:
    pass


@dataclass(frozen=True, slots=True)
class HuffmanConfig:
    candidate_count: int | None = None
    bit_num: int | None = None

    def __post_init__(self) -> None:
        if self.candidate_count is not None and self.candidate_count < 1:
            raise ValueError("HuffmanConfig.candidate_count must be >= 1")
        if self.bit_num is not None and self.bit_num < 1:
            raise ValueError("HuffmanConfig.bit_num must be >= 1")


@dataclass(frozen=True, slots=True)
class ARSEncodeConfig:
    seed: str = "12345"
    secure_parameter: int = 32
    func_type: int = 0

    def __post_init__(self) -> None:
        if self.secure_parameter < 1:
            raise ValueError("ARSEncodeConfig.secure_parameter must be >= 1")
        if self.func_type not in {0, 1, 2}:
            raise ValueError("ARSEncodeConfig.func_type must be one of {0, 1, 2}")


@dataclass(frozen=True, slots=True)
class ARSDecodeConfig(ARSEncodeConfig):
    decode_mode: str = "regular"
    robust_search_window: int = 1000

    def __post_init__(self) -> None:
        ARSEncodeConfig.__post_init__(self)
        if self.decode_mode not in {"regular", "robust"}:
            raise ValueError("ARSDecodeConfig.decode_mode must be 'regular' or 'robust'")
        if self.robust_search_window < 1:
            raise ValueError("ARSDecodeConfig.robust_search_window must be >= 1")


@dataclass(frozen=True, slots=True)
class NoConfig:
    pass
