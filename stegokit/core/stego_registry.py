from __future__ import annotations

from typing import TypeAlias

from stegokit.algo.ac.ac import ACStrategy
from stegokit.algo.adg.adg import ADGStrategy
from stegokit.algo.discop.discop import DiscopStrategy
from stegokit.algo.discop.discop_base import DiscopBaseStrategy
from stegokit.algo.fdpss.binary_based import BinaryBasedStrategy
from stegokit.algo.fdpss.differential_based import DifferentialBasedStrategy
from stegokit.algo.fdpss.stability_based import StabilityBasedStrategy
from stegokit.algo.huffman.huffman import HuffmanStrategy
from stegokit.algo.meteor.meteor import MeteorStrategy
from stegokit.algo.ars.ars import ARSStrategy
from stegokit.core.algorithm_config import (
    ACConfig,
    ADGConfig,
    ARSDecodeConfig,
    ARSEncodeConfig,
    DiscopBaseConfig,
    DiscopConfig,
    FDPSSBinaryBasedConfig,
    FDPSSDifferentialBasedConfig,
    FDPSSStabilityBasedConfig,
    HuffmanConfig,
    MeteorConfig,
    NoConfig,
)
from stegokit.core.algorithm_enum import StegoAlgorithm
from stegokit.core.algorithm_spec import AlgorithmSpec
from stegokit.core.security_material import (
    AsymmetricDecodeMaterial,
    AsymmetricEncodeMaterial,
    BitMaskMaterial,
    NoMaterial,
    RandomnessMaterial,
)
from stegokit.core.stego_algorithm import StegoStrategy
from stegokit.core.stego_paradigm import StegoParadigm

AlgorithmRef: TypeAlias = StegoAlgorithm | str


class StegoAlgorithmRegistry:
    def __init__(self) -> None:
        self._builtin_specs: dict[StegoAlgorithm, AlgorithmSpec] = {}
        self._custom_specs: dict[str, AlgorithmSpec] = {}

    def _register_builtin_spec(self, algorithm: StegoAlgorithm, spec: AlgorithmSpec) -> None:
        algo = StegoAlgorithm(algorithm)
        if algo in self._builtin_specs:
            raise ValueError(f"Algorithm already registered: {algo.value}")
        self._builtin_specs[algo] = spec

    def register(self, name: str, strategy: StegoStrategy) -> None:
        self.register_spec(
            AlgorithmSpec(
                name=name,
                paradigm=StegoParadigm.SYMMETRIC,
                strategy=strategy,
                encode_config_type=NoConfig,
                decode_config_type=NoConfig,
                encode_material_type=NoMaterial,
                decode_material_type=NoMaterial,
            )
        )

    def register_spec(self, spec: AlgorithmSpec) -> None:
        if not isinstance(spec, AlgorithmSpec):
            raise TypeError("spec must be an AlgorithmSpec instance")

        name = spec.name
        if not isinstance(name, str):
            raise TypeError("AlgorithmSpec.name must be a string")
        key = name.strip().lower()
        if not key:
            raise ValueError("Algorithm name cannot be empty")
        if key in {algo.value for algo in StegoAlgorithm}:
            raise ValueError(f"Algorithm name is reserved by built-in algorithm: {key}")
        if key in self._custom_specs:
            raise ValueError(f"Custom algorithm already registered: {key}")
        self._custom_specs[key] = AlgorithmSpec(
            name=key,
            paradigm=spec.paradigm,
            strategy=spec.strategy,
            encode_config_type=spec.encode_config_type,
            decode_config_type=spec.decode_config_type,
            encode_material_type=spec.encode_material_type,
            decode_material_type=spec.decode_material_type,
            notes=spec.notes,
            encode_config_validator=spec.encode_config_validator,
            decode_config_validator=spec.decode_config_validator,
            encode_material_validator=spec.encode_material_validator,
            decode_material_validator=spec.decode_material_validator,
        )

    def get_spec(self, algorithm: AlgorithmRef) -> AlgorithmSpec:
        if isinstance(algorithm, StegoAlgorithm):
            spec = self._builtin_specs.get(algorithm)
            if spec is None:
                raise KeyError(f"Built-in algorithm not registered: {algorithm.value}")
            return spec

        key = algorithm.strip().lower()
        try:
            builtin = StegoAlgorithm(key)
        except ValueError:
            builtin = None

        if builtin is not None:
            spec = self._builtin_specs.get(builtin)
            if spec is None:
                raise KeyError(f"Built-in algorithm not registered: {builtin.value}")
            return spec

        spec = self._custom_specs.get(key)
        if spec is None:
            raise KeyError(f"Custom algorithm not registered: {key}")
        return spec

    def get(self, algorithm: AlgorithmRef) -> StegoStrategy:
        return self.get_spec(algorithm).strategy

    def specs(self) -> dict[str, AlgorithmSpec]:
        data: dict[str, AlgorithmSpec] = {algo.value: spec for algo, spec in self._builtin_specs.items()}
        data.update(self._custom_specs)
        return data

    @classmethod
    def default(cls) -> "StegoAlgorithmRegistry":
        registry = cls()
        registry._register_builtin_spec(
            StegoAlgorithm.AC,
            AlgorithmSpec(
                name=StegoAlgorithm.AC.value,
                paradigm=StegoParadigm.SYMMETRIC,
                strategy=ACStrategy(),
                encode_config_type=ACConfig,
                decode_config_type=ACConfig,
                encode_material_type=NoMaterial,
                decode_material_type=NoMaterial,
            ),
        )
        registry._register_builtin_spec(
            StegoAlgorithm.ADG,
            AlgorithmSpec(
                name=StegoAlgorithm.ADG.value,
                paradigm=StegoParadigm.SYMMETRIC,
                strategy=ADGStrategy(),
                encode_config_type=ADGConfig,
                decode_config_type=ADGConfig,
                encode_material_type=NoMaterial,
                decode_material_type=NoMaterial,
            ),
        )
        registry._register_builtin_spec(
            StegoAlgorithm.DISCOP,
            AlgorithmSpec(
                name=StegoAlgorithm.DISCOP.value,
                paradigm=StegoParadigm.SYMMETRIC,
                strategy=DiscopStrategy(),
                encode_config_type=DiscopConfig,
                decode_config_type=DiscopConfig,
                encode_material_type=RandomnessMaterial,
                decode_material_type=RandomnessMaterial,
            ),
        )
        registry._register_builtin_spec(
            StegoAlgorithm.DISCOP_BASE,
            AlgorithmSpec(
                name=StegoAlgorithm.DISCOP_BASE.value,
                paradigm=StegoParadigm.SYMMETRIC,
                strategy=DiscopBaseStrategy(),
                encode_config_type=DiscopBaseConfig,
                decode_config_type=DiscopBaseConfig,
                encode_material_type=RandomnessMaterial,
                decode_material_type=RandomnessMaterial,
            ),
        )
        registry._register_builtin_spec(
            StegoAlgorithm.FDPSS_DIFFERENTIAL_BASED,
            AlgorithmSpec(
                name=StegoAlgorithm.FDPSS_DIFFERENTIAL_BASED.value,
                paradigm=StegoParadigm.SYMMETRIC,
                strategy=DifferentialBasedStrategy(),
                encode_config_type=FDPSSDifferentialBasedConfig,
                decode_config_type=FDPSSDifferentialBasedConfig,
                encode_material_type=RandomnessMaterial,
                decode_material_type=RandomnessMaterial,
            ),
        )
        registry._register_builtin_spec(
            StegoAlgorithm.FDPSS_BINARY_BASED,
            AlgorithmSpec(
                name=StegoAlgorithm.FDPSS_BINARY_BASED.value,
                paradigm=StegoParadigm.SYMMETRIC,
                strategy=BinaryBasedStrategy(),
                encode_config_type=FDPSSBinaryBasedConfig,
                decode_config_type=FDPSSBinaryBasedConfig,
                encode_material_type=RandomnessMaterial,
                decode_material_type=RandomnessMaterial,
            ),
        )
        registry._register_builtin_spec(
            StegoAlgorithm.FDPSS_STABILITY_BASED,
            AlgorithmSpec(
                name=StegoAlgorithm.FDPSS_STABILITY_BASED.value,
                paradigm=StegoParadigm.SYMMETRIC,
                strategy=StabilityBasedStrategy(),
                encode_config_type=FDPSSStabilityBasedConfig,
                decode_config_type=FDPSSStabilityBasedConfig,
                encode_material_type=RandomnessMaterial,
                decode_material_type=RandomnessMaterial,
            ),
        )
        registry._register_builtin_spec(
            StegoAlgorithm.METEOR,
            AlgorithmSpec(
                name=StegoAlgorithm.METEOR.value,
                paradigm=StegoParadigm.SYMMETRIC,
                strategy=MeteorStrategy(),
                encode_config_type=MeteorConfig,
                decode_config_type=MeteorConfig,
                encode_material_type=BitMaskMaterial,
                decode_material_type=BitMaskMaterial,
            ),
        )
        registry._register_builtin_spec(
            StegoAlgorithm.ARS,
            AlgorithmSpec(
                name=StegoAlgorithm.ARS.value,
                paradigm=StegoParadigm.ASYMMETRIC,
                strategy=ARSStrategy(),
                encode_config_type=ARSEncodeConfig,
                decode_config_type=ARSDecodeConfig,
                encode_material_type=AsymmetricEncodeMaterial,
                decode_material_type=AsymmetricDecodeMaterial,
            ),
        )
        registry._register_builtin_spec(
            StegoAlgorithm.HUFFMAN,
            AlgorithmSpec(
                name=StegoAlgorithm.HUFFMAN.value,
                paradigm=StegoParadigm.SYMMETRIC,
                strategy=HuffmanStrategy(),
                encode_config_type=HuffmanConfig,
                decode_config_type=HuffmanConfig,
                encode_material_type=NoMaterial,
                decode_material_type=NoMaterial,
            ),
        )
        return registry
