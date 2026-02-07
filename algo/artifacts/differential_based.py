from __future__ import annotations

import torch

from .common import ArtifactsCommonMixin
from core.stego_algorithm import DecodeResult, EncodeResult
from core.stego_context import DecodeContext, EncodeContext


class DifferentialBasedStrategy(ArtifactsCommonMixin):
    def encode(self, context: EncodeContext) -> EncodeResult:
        prg = self._require_prg(context.prg)
        prob, indices = self._to_tensors(context.prob_table, context.indices)

        indices_nonzero, bins, prob_new = self._differential_based_recombination(prob, indices)
        prob_new = prob_new / prob_new.sum()

        random_p = prg.generate_random(n=context.precision)
        cdf = torch.cumsum(prob_new, dim=0)
        bin_idx = torch.searchsorted(cdf, random_p).item()
        bin_vals = indices_nonzero[bins[bin_idx]:]

        idx, bits = self._uni_cyclic_shift_enc(
            bit_stream=context.bit_stream[context.bit_index:],
            n=len(bin_vals),
            prg=prg,
            precision=context.precision,
        )
        sampled_token_id = int(bin_vals[idx].item())
        bits_used = len(bits)
        return EncodeResult(
            sampled_token_id=sampled_token_id,
            bits_consumed=bits_used,
            metadata={"next_bit_index": context.bit_index + bits_used},
        )

    def decode(self, context: DecodeContext) -> DecodeResult:
        prg = self._require_prg(context.prg)
        prob, indices = self._to_tensors(context.prob_table, context.indices)

        indices_nonzero, bins, prob_new = self._differential_based_recombination(prob, indices)
        prob_new = prob_new / prob_new.sum()

        random_p = prg.generate_random(n=context.precision)
        cdf = torch.cumsum(prob_new, dim=0)
        bin_idx = torch.searchsorted(cdf, random_p).item()
        bin_vals = indices_nonzero[bins[bin_idx]:]

        prev = int(context.prev_token_id)
        pos = (bin_vals == prev).nonzero()
        if len(pos) == 0:
            return DecodeResult(bits="", metadata={"bits_len": 0})
        idx = int(pos.item())
        bits = self._uni_cyclic_shift_dec(idx=idx, n=len(bin_vals), prg=prg, precision=context.precision)
        return DecodeResult(bits=bits, metadata={"bits_len": len(bits)})
