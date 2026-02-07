from __future__ import annotations

from typing import Sequence

import torch

from algo.common import (
    bit_slice_with_padding,
    bits2int,
    int2bits,
    num_same_from_beg,
    require_prg_method,
    to_sorted_tensors,
)
from core.stego_algorithm import DecodeResult, EncodeResult
from core.stego_context import DecodeContext, EncodeContext


class MeteorStrategy:
    @staticmethod
    def _require_prg(prg):
        return require_prg_method(prg, "generate_bits", "Meteor strategy")

    @staticmethod
    def _prepare_inputs(prob_table: Sequence[float], indices: Sequence[int], precision: int) -> tuple[torch.Tensor, torch.Tensor]:
        prob, token_indices = to_sorted_tensors(prob_table, indices)

        topk = len(prob)
        epsilon = 2 ** (-precision)
        nonzero_indices = (prob < epsilon).nonzero().squeeze()
        if nonzero_indices.numel() > 0:
            if nonzero_indices.numel() == 1:
                first = nonzero_indices.item()
            else:
                first = nonzero_indices[0].item()
            topk = min(max(2, first), topk)

        prob = prob[:topk]
        token_indices = token_indices[:topk]
        return prob, token_indices

    @staticmethod
    def _build_cum_probs(prob: torch.Tensor, precision: int) -> torch.Tensor:
        cur_int_range = 2 ** precision
        prob = (prob / torch.sum(prob) * cur_int_range).round().long()
        cum_probs = prob.cumsum(0)
        overfill_index = (cum_probs > cur_int_range).nonzero()
        if len(overfill_index) > 0:
            cum_probs = cum_probs[:overfill_index[0]]
        cum_probs += cur_int_range - cum_probs[-1]
        return cum_probs

    def encode(self, context: EncodeContext) -> EncodeResult:
        prg = self._require_prg(context.prg)
        precision = context.precision
        prob, indices = self._prepare_inputs(context.prob_table, context.indices, precision)
        cum_probs = self._build_cum_probs(prob, precision)

        bits_slice = bit_slice_with_padding(context.bit_stream, context.bit_index, precision)
        message_bits = [int(bit) for bit in bits_slice]

        mask_bits = prg.generate_bits(precision)
        for i in range(len(message_bits)):
            message_bits[i] = message_bits[i] ^ int(mask_bits[i])

        message_idx = bits2int(reversed(message_bits))
        selection = (cum_probs > message_idx).nonzero()[0].item()

        new_int_bottom = cum_probs[selection - 1] if selection > 0 else 0
        new_int_top = cum_probs[selection]

        new_int_bottom_bits = list(reversed(int2bits(int(new_int_bottom), precision)))
        new_int_top_bits = list(reversed(int2bits(int(new_int_top - 1), precision)))
        bits_consumed = num_same_from_beg(new_int_bottom_bits, new_int_top_bits)

        sampled_token_id = int(indices[selection].item())
        return EncodeResult(
            sampled_token_id=sampled_token_id,
            bits_consumed=bits_consumed,
            metadata={"next_bit_index": context.bit_index + bits_consumed},
        )

    def decode(self, context: DecodeContext) -> DecodeResult:
        prg = self._require_prg(context.prg)
        precision = context.precision
        prob, indices = self._prepare_inputs(context.prob_table, context.indices, precision)
        cum_probs = self._build_cum_probs(prob, precision)

        prev = int(context.prev_token_id)
        positions = (indices == prev).nonzero()
        if len(positions) == 0:
            return DecodeResult(bits="", metadata={"bits_len": 0})
        selection = positions[0].item()

        new_int_bottom = cum_probs[selection - 1] if selection > 0 else 0
        new_int_top = cum_probs[selection]
        new_int_bottom_bits = list(reversed(int2bits(int(new_int_bottom), precision)))
        new_int_top_bits = list(reversed(int2bits(int(new_int_top - 1), precision)))
        num = num_same_from_beg(new_int_bottom_bits, new_int_top_bits)

        new_bits = new_int_top_bits[:num]
        mask_bits = prg.generate_bits(precision)
        for i in range(len(new_bits)):
            new_bits[i] = new_bits[i] ^ int(mask_bits[i])
        bits = "".join(str(b) for b in new_bits)
        return DecodeResult(bits=bits, metadata={"bits_len": len(bits)})
