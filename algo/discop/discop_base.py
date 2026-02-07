from __future__ import annotations

import math

import torch

from algo.common import bit_slice_with_padding
from .common import DiscopCommonMixin
from core.stego_algorithm import DecodeResult, EncodeResult
from core.stego_context import DecodeContext, EncodeContext


class DiscopBaseStrategy(DiscopCommonMixin):
    """Discop-base strategy."""

    @staticmethod
    def _baseline_encode_step(indices: list[int], probs: list[float], message_bits: str, bit_index: int, prg, precision: int):
        probs_cumsum = torch.tensor(probs).cumsum(dim=0)
        interval_begin = torch.cat((torch.tensor([0], device=probs_cumsum.device), probs_cumsum[:-1]), dim=0)

        capacity = int(math.log2(1 / probs[0]))
        capacity_upper_bound = capacity + 1
        tbl = {}
        ptr = prg.generate_random(n=precision)
        n_bits = 0

        while capacity <= capacity_upper_bound:
            if capacity == 0:
                capacity += 1
                continue

            rotate_step_size = 2.0 ** (-capacity)
            is_available = True
            tbl_new = {}
            for i in range(int(2 ** capacity)):
                ptr_i = ptr + i * rotate_step_size
                if ptr_i >= 1.0:
                    ptr_i -= 1
                index_idx = (ptr_i >= interval_begin).nonzero()[-1].item()
                index = indices[index_idx]
                if index in tbl_new.values():
                    is_available = False
                    break
                tbl_new[i] = index
            if not is_available:
                break
            tbl = tbl_new
            n_bits = capacity
            capacity += 1

        if n_bits < 1:
            sampled_index = indices[(ptr >= interval_begin).nonzero()[-1].item()]
        else:
            cur_message_bits_decimal = 0
            base = 1
            for d in range(n_bits - 1, -1, -1):
                if message_bits[bit_index + d] == "1":
                    cur_message_bits_decimal += base
                base *= 2
            sampled_index = tbl[cur_message_bits_decimal]

        return sampled_index, n_bits

    @staticmethod
    def _baseline_decode_step(indices: list[int], probs: list[float], stego_t: int, prg, precision: int) -> str:
        probs_cumsum = torch.tensor(probs).cumsum(dim=0)
        interval_begin = torch.cat((torch.tensor([0], device=probs_cumsum.device), probs_cumsum[:-1]), dim=0)

        capacity = int(math.log2(1 / probs[0]))
        capacity_upper_bound = capacity + 1
        tbl = {}
        ptr = prg.generate_random(n=precision)
        n_bits = 0

        while capacity <= capacity_upper_bound:
            if capacity == 0:
                capacity += 1
                continue

            rotate_step_size = 2.0 ** (-capacity)
            is_available = True
            tbl_new = {}
            for i in range(int(2 ** capacity)):
                ptr_i = ptr + i * rotate_step_size
                if ptr_i >= 1.0:
                    ptr_i -= 1
                index_idx = (ptr_i >= interval_begin).nonzero()[-1].item()
                index = indices[index_idx]
                if index in tbl_new.values():
                    is_available = False
                    break
                tbl_new[i] = index
            if not is_available:
                break
            tbl = tbl_new
            n_bits = capacity
            capacity += 1

        if n_bits < 1:
            return ""
        if stego_t not in tbl.values():
            return ""
        tbl_swapped = dict(zip(tbl.values(), tbl.keys()))
        return bin(tbl_swapped[stego_t])[2:].zfill(n_bits)

    def encode(self, context: EncodeContext) -> EncodeResult:
        prg = self._require_prg(context.prg)
        probs_list, indices_list = self._prepare_inputs(context.prob_table, context.indices)

        bits_slice = bit_slice_with_padding(context.bit_stream, context.bit_index, context.precision)

        sampled_index, n_bits = self._baseline_encode_step(
            indices_list, probs_list, bits_slice, 0, prg, context.precision
        )

        return EncodeResult(
            sampled_token_id=int(sampled_index),
            bits_consumed=int(n_bits),
            metadata={"next_bit_index": context.bit_index + int(n_bits)},
        )

    def decode(self, context: DecodeContext) -> DecodeResult:
        prg = self._require_prg(context.prg)
        probs_list, indices_list = self._prepare_inputs(context.prob_table, context.indices)
        stego_t = int(context.prev_token_id)
        bits = self._baseline_decode_step(indices_list, probs_list, stego_t, prg, context.precision)
        return DecodeResult(bits=bits, metadata={"bits_len": len(bits)})
