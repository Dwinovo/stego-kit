from __future__ import annotations

import torch

from .common import ArtifactsCommonMixin
from core.stego_algorithm import DecodeResult, EncodeResult
from core.stego_context import DecodeContext, EncodeContext


class StabilityBasedStrategy(ArtifactsCommonMixin):
    @staticmethod
    def _sample_bin(p_sum: torch.Tensor, q_sum: torch.Tensor, t, device) -> torch.Tensor:
        assert torch.allclose(p_sum[-1], q_sum[-1])
        q_sum2 = torch.concatenate([torch.tensor([0], device=device), q_sum])
        i = torch.searchsorted(q_sum, t, side="right")
        s = t - q_sum2[i]
        l = q_sum2[:-1] + s
        l = l[l < q_sum]
        return torch.searchsorted(p_sum, l, side="right")

    def _sample_method2_encode(self, p: torch.Tensor, t, device) -> torch.Tensor:
        p, _ = torch.sort(p, descending=True)
        q2 = p[1] + min(p[0], 1 - p[0] - 1e-8) - p[1]
        p_sum = torch.cumsum(p, dim=0)
        q2s = p_sum[0] + q2
        q_sum = torch.concatenate((torch.tensor([p_sum[0], q2s], device=device), p_sum[p_sum > q2s]), axis=0)
        return self._sample_bin(p_sum, q_sum, t, device)

    def _sample_method2_decode(self, p: torch.Tensor, t, device) -> torch.Tensor:
        p, _ = torch.sort(p, descending=True)
        q2 = min(p[0], 1 - p[0] - 1e-8)
        p_sum = torch.cumsum(p, dim=0)
        q2s = p_sum[0] + q2
        q_sum = torch.concatenate((torch.tensor([p_sum[0], q2s], device=device), p_sum[p_sum > q2s]), axis=0)
        return self._sample_bin(p_sum, q_sum, t, device)

    def encode(self, context: EncodeContext) -> EncodeResult:
        prg = self._require_prg(context.prg)
        prob, indices = self._to_tensors(context.prob_table, context.indices)
        device = prob.device
        prob, sorted_indices = torch.sort(prob, descending=True)
        indices = indices[sorted_indices]

        random_p = prg.generate_random(n=context.precision)
        bin_vals = indices[self._sample_method2_encode(prob, random_p, device)]

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
        device = prob.device
        prob, sorted_indices = torch.sort(prob, descending=True)
        indices = indices[sorted_indices]

        random_p = prg.generate_random(n=context.precision)
        bin_vals = indices[self._sample_method2_decode(prob, random_p, device)]

        prev = int(context.prev_token_id)
        pos = (bin_vals == prev).nonzero()
        if len(pos) == 0:
            return DecodeResult(bits="", metadata={"bits_len": 0})
        idx = int(pos.item())
        bits = self._uni_cyclic_shift_dec(idx=idx, n=len(bin_vals), prg=prg, precision=context.precision)
        return DecodeResult(bits=bits, metadata={"bits_len": len(bits)})
