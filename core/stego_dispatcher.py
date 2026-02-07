from __future__ import annotations

import logging
from typing import Any

from core.algorithm_enum import StegoAlgorithm
from core.stego_algorithm import DecodeResult, EncodeResult
from core.stego_context import DecodeContext, EncodeContext
from core.stego_registry import AlgorithmRegistry

logger = logging.getLogger(__name__)


class StegoDispatcher:
    """Thin orchestrator: validate context, resolve strategy, execute."""

    def __init__(self, registry: AlgorithmRegistry | None = None, verbose: bool = True) -> None:
        self._registry = registry or AlgorithmRegistry.default()
        self._verbose = verbose

    @property
    def registry(self) -> AlgorithmRegistry:
        return self._registry

    def dispatch_encode(self, context: EncodeContext) -> EncodeResult:
        self._validate_encode_context(context)
        strategy = self._registry.get(context.algorithm)
        builtin_algo = self._resolve_builtin_algorithm(context.algorithm)
        algo_name = self._algorithm_name(context.algorithm, builtin_algo)
        self._info_if_prg_ignored(builtin_algo, context.prg)

        if self._verbose:
            preview = list(context.prob_table[:5])
            logger.info(
                f"[Encode Dispatcher] algorithm={algo_name}, prob_table_size={len(context.prob_table)}, "
                f"indices_size={len(context.indices)}, bit_index={context.bit_index}, precision={context.precision}, "
                f"preview={preview}"
            )

        return strategy.encode(context)

    def dispatch_decode(self, context: DecodeContext) -> DecodeResult:
        self._validate_decode_context(context)
        strategy = self._registry.get(context.algorithm)
        builtin_algo = self._resolve_builtin_algorithm(context.algorithm)
        algo_name = self._algorithm_name(context.algorithm, builtin_algo)
        self._info_if_prg_ignored(builtin_algo, context.prg)

        if self._verbose:
            preview = list(context.prob_table[:5])
            logger.info(
                f"[Decode Dispatcher] algorithm={algo_name}, prob_table_size={len(context.prob_table)}, "
                f"indices_size={len(context.indices)}, prev_token_id={context.prev_token_id}, "
                f"precision={context.precision}, preview={preview}"
            )

        return strategy.decode(context)

    @staticmethod
    def _validate_encode_context(context: EncodeContext) -> None:
        if not context.prob_table:
            raise ValueError("prob_table cannot be empty")
        if not context.indices:
            raise ValueError("indices cannot be empty")

    @staticmethod
    def _validate_decode_context(context: DecodeContext) -> None:
        if not context.prob_table:
            raise ValueError("prob_table cannot be empty")
        if not context.indices:
            raise ValueError("indices cannot be empty")

    def _info_if_prg_ignored(self, algo: StegoAlgorithm | None, prg: Any) -> None:
        if self._verbose and algo == StegoAlgorithm.AC and prg is not None:
            logger.info("[INFO] AC does not use PRG; `context.prg` is ignored.")

    @staticmethod
    def _resolve_builtin_algorithm(algorithm: StegoAlgorithm | str) -> StegoAlgorithm | None:
        if isinstance(algorithm, StegoAlgorithm):
            return algorithm
        try:
            return StegoAlgorithm(algorithm.strip().lower())
        except ValueError:
            return None

    @staticmethod
    def _algorithm_name(algorithm: StegoAlgorithm | str, builtin: StegoAlgorithm | None) -> str:
        if builtin is not None:
            return builtin.value
        return algorithm.strip().lower() if isinstance(algorithm, str) else str(algorithm)


if __name__ == "__main__":
    class DemoPRG:
        def __init__(self, seed: int = 1) -> None:
            self.v = seed

        def generate_random(self, n):
            self.v = (1103515245 * self.v + 12345) % (2**31)
            return (self.v % 1000000) / 1000000.0

    dispatcher = StegoDispatcher()
    demo_prob_table = [0.42, 0.26, 0.18, 0.09, 0.05]
    demo_indices = [10, 20, 30, 40, 50]

    logger.info("=== Dispatcher Demo Start ===")
    for algo in (
        StegoAlgorithm.AC,
        StegoAlgorithm.DISCOP,
        StegoAlgorithm.DIFFERENTIAL_BASED,
        StegoAlgorithm.METEOR,
    ):
        logger.info(f"\n--- encode test {algo.value} ---")
        dispatcher.dispatch_encode(
            EncodeContext(
                algorithm=algo,
                prob_table=demo_prob_table,
                indices=demo_indices,
                bit_stream="010101011001",
                bit_index=0,
                precision=16,
                prg=DemoPRG(seed=7),
            )
        )
        logger.info(f"--- decode test {algo.value} ---")
        dispatcher.dispatch_decode(
            DecodeContext(
                algorithm=algo,
                prob_table=demo_prob_table,
                indices=demo_indices,
                prev_token_id=20,
                precision=16,
                prg=DemoPRG(seed=7),
            )
        )
    logger.info("\n=== Dispatcher Demo End ===")
