from __future__ import annotations

import logging
from typing import TypeAlias

from algo.ac.ac import ACStrategy
from algo.artifacts.binary_based import BinaryBasedStrategy
from algo.artifacts.differential_based import DifferentialBasedStrategy
from algo.artifacts.stability_based import StabilityBasedStrategy
from algo.discop.discop import DiscopStrategy
from algo.discop.discop_base import DiscopBaseStrategy
from algo.meteor.meteor import MeteorStrategy
from core.algorithm_enum import StegoAlgorithm
from core.stego_algorithm import DecodeResult, EncodeResult, StegoStrategy
from core.stego_context import DecodeContext, EncodeContext

logger = logging.getLogger(__name__)
AlgorithmRef: TypeAlias = StegoAlgorithm | str


class PlaceholderStrategy:
    def __init__(self, name: str) -> None:
        self._name = name

    def encode(self, context: EncodeContext) -> EncodeResult:
        algo_name = context.algorithm.value if isinstance(context.algorithm, StegoAlgorithm) else str(context.algorithm)
        logger.info(f"[{self._name}] encode placeholder only, algorithm not implemented yet.")
        return EncodeResult(metadata={"algorithm": algo_name, "placeholder": True})

    def decode(self, context: DecodeContext) -> DecodeResult:
        algo_name = context.algorithm.value if isinstance(context.algorithm, StegoAlgorithm) else str(context.algorithm)
        logger.info(f"[{self._name}] decode placeholder only, algorithm not implemented yet.")
        return DecodeResult(metadata={"algorithm": algo_name, "placeholder": True})


class AlgorithmRegistry:
    def __init__(self) -> None:
        self._builtin_strategies: dict[StegoAlgorithm, StegoStrategy] = {}
        self._custom_strategies: dict[str, StegoStrategy] = {}

    def _register_builtin(self, algorithm: StegoAlgorithm, strategy: StegoStrategy) -> None:
        algo = StegoAlgorithm(algorithm)
        if algo in self._builtin_strategies:
            raise ValueError(f"Algorithm already registered: {algo.value}")
        self._builtin_strategies[algo] = strategy

    def register(self, name: str, strategy: StegoStrategy) -> None:
        key = name.strip().lower()
        if not key:
            raise ValueError("Algorithm name cannot be empty")
        if key in {algo.value for algo in StegoAlgorithm}:
            raise ValueError(f"Algorithm name is reserved by built-in algorithm: {key}")
        if key in self._custom_strategies:
            raise ValueError(f"Custom algorithm already registered: {key}")
        self._custom_strategies[key] = strategy

    def get(self, algorithm: AlgorithmRef) -> StegoStrategy:
        if isinstance(algorithm, StegoAlgorithm):
            strategy = self._builtin_strategies.get(algorithm)
            if strategy is None:
                raise KeyError(f"Built-in algorithm not registered: {algorithm.value}")
            return strategy

        key = algorithm.strip().lower()
        try:
            builtin = StegoAlgorithm(key)
        except ValueError:
            builtin = None

        if builtin is not None:
            strategy = self._builtin_strategies.get(builtin)
            if strategy is None:
                raise KeyError(f"Built-in algorithm not registered: {builtin.value}")
            return strategy

        strategy = self._custom_strategies.get(key)
        if strategy is None:
            raise KeyError(f"Custom algorithm not registered: {key}")
        return strategy

    @classmethod
    def default(cls) -> "AlgorithmRegistry":
        registry = cls()
        registry._register_builtin(StegoAlgorithm.AC, ACStrategy())
        registry._register_builtin(StegoAlgorithm.DISCOP, DiscopStrategy())
        registry._register_builtin(StegoAlgorithm.DISCOP_BASE, DiscopBaseStrategy())
        registry._register_builtin(StegoAlgorithm.DIFFERENTIAL_BASED, DifferentialBasedStrategy())
        registry._register_builtin(StegoAlgorithm.BINARY_BASED, BinaryBasedStrategy())
        registry._register_builtin(StegoAlgorithm.STABILITY_BASED, StabilityBasedStrategy())
        registry._register_builtin(StegoAlgorithm.METEOR, MeteorStrategy())
        return registry
