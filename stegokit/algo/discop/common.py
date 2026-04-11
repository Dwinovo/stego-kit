from __future__ import annotations

from stegokit.core.security_material import require_randomness_material
from stegokit.algo.common import to_sorted_tensors


class DiscopCommonMixin:
    @staticmethod
    def _prepare_inputs(prob_table, indices) -> tuple[list[float], list[int]]:
        prob, token_indices = to_sorted_tensors(prob_table, indices)
        return prob.tolist(), token_indices.tolist()

    @staticmethod
    def _require_prg(material):
        return require_randomness_material(material, "Discop strategy")
