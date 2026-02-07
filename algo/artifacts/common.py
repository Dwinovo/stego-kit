from __future__ import annotations

import math
from typing import Sequence

import torch

from algo.common import lsb_bits2int, lsb_int2bits, require_prg_method, to_tensors


class ArtifactsCommonMixin:
    @staticmethod
    def _require_prg(prg):
        return require_prg_method(prg, "generate_random", "Artifacts strategies")

    @staticmethod
    def _to_tensors(prob_table: Sequence[float], indices: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
        return to_tensors(prob_table, indices)

    def _uni_cyclic_shift_enc(self, bit_stream: str, n: int, prg, precision: int) -> tuple[int, str]:
        if n == 1:
            prg.generate_random(n=precision)
            return 0, ""

        ptr = prg.generate_random(n=precision)
        r = math.floor(ptr * n)
        k = math.floor(math.log2(n))
        t = n - 2 ** k

        bits = bit_stream[:k]
        bits_res = bit_stream[k] if k < len(bit_stream) else "0"
        idx_sort = lsb_bits2int([int(b) for b in bits])

        if idx_sort < 2 ** k - t:
            return (idx_sort + r) % n, bits
        idx = (2 * (idx_sort - (2 ** k - t)) + (2 ** k - t) + r + int(bits_res)) % n
        return idx, bits + bits_res

    def _uni_cyclic_shift_dec(self, idx: int, n: int, prg, precision: int) -> str:
        if n == 1:
            prg.generate_random(n=precision)
            return ""

        ptr = prg.generate_random(n=precision)
        r = math.floor(ptr * n)
        k = math.floor(math.log2(n))
        t = n - 2 ** k
        idx_sort = (idx - r) % n

        if idx_sort < 2 ** k - t:
            bits = lsb_int2bits(idx_sort, k)
            return "".join(str(_) for _ in bits)

        s1 = idx_sort - 2 ** k + t
        s_last = s1 % 2
        bits = lsb_int2bits((s1 - s_last) // 2 + 2 ** k - t, k)
        return "".join(str(_) for _ in bits) + ("0" if s_last == 0 else "1")

    @staticmethod
    def _differential_based_recombination(prob: torch.Tensor, indices: torch.Tensor):
        prob, sorted_indices = torch.sort(prob, descending=False)
        indices = indices[sorted_indices]
        mask = prob > 0
        prob_nonzero = prob[mask]
        indices_nonzero = indices[mask]

        diff = torch.cat((prob_nonzero[:1], torch.diff(prob_nonzero, n=1)))
        n = len(prob_nonzero)
        weights = torch.arange(n, 0, -1, device=prob.device)
        diff_positive = diff > 0
        prob_new = diff[diff_positive] * weights[diff_positive]
        bins = torch.arange(n, device=prob.device)[diff_positive]
        return indices_nonzero, bins, prob_new

    @staticmethod
    def _binary_based_recombination(prob: torch.Tensor, indices: torch.Tensor, precision: int):
        mask = prob > 0
        prob_nonzero = prob[mask]
        indices_nonzero = indices[mask]

        scale = 2 ** precision
        scaled_probs = (prob_nonzero * scale).to(torch.int64)
        masks = (1 << torch.arange(precision)).to(scaled_probs.device)
        masked_probs = scaled_probs.unsqueeze(1) & masks
        nonzero_mask_indices = masked_probs > 0

        bins = [indices_nonzero[nonzero_mask_indices[:, k]] for k in range(precision)]
        bin_sizes = torch.tensor([len(g) for g in bins], dtype=torch.long)
        power_i = 2 ** (-torch.arange(precision, 0, -1).float())
        prob_new = bin_sizes * power_i
        return bins, prob_new
