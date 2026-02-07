from __future__ import annotations

from algo.common import require_prg_method, to_sorted_tensors


class DiscopCommonMixin:
    @staticmethod
    def _prepare_inputs(prob_table, indices) -> tuple[list[float], list[int]]:
        prob, token_indices = to_sorted_tensors(prob_table, indices)
        return prob.tolist(), token_indices.tolist()

    @staticmethod
    def _require_prg(prg):
        return require_prg_method(prg, "generate_random", "Discop strategy")
