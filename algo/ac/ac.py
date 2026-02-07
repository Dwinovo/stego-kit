from __future__ import annotations

from algo.common import (
    bit_slice_with_padding,
    msb_bits2int,
    msb_int2bits,
    num_same_from_beg,
    to_sorted_tensors,
)
from core.stego_algorithm import DecodeResult, EncodeResult
from core.stego_context import DecodeContext, EncodeContext


class ACStrategy:
    """Arithmetic-coding-based steganography strategy."""

    def encode(self, context: EncodeContext) -> EncodeResult:
        prob, indices = to_sorted_tensors(context.prob_table, context.indices)
        precision = context.precision
        cur_interval = list(context.cur_interval or [0, 2 ** precision])

        cur_int_range = cur_interval[1] - cur_interval[0]
        cur_threshold = 1 / cur_int_range
        if prob[-1] < cur_threshold:
            k = max(2, (prob < cur_threshold).nonzero()[0].item())
            prob = prob[:k]
            indices = indices[:k]

        prob = prob / prob.sum()
        prob = (prob * cur_int_range).round().long()

        cum_probs = prob.cumsum(0)
        overfill_index = (cum_probs > cur_int_range).nonzero()
        if len(overfill_index) > 0:
            cum_probs = cum_probs[:overfill_index[0]]
        cum_probs += cur_int_range - cum_probs[-1]
        cum_probs += cur_interval[0]

        bits_slice = bit_slice_with_padding(context.bit_stream, context.bit_index, precision)
        message_idx = msb_bits2int([int(ch) for ch in bits_slice])
        selection = (cum_probs > message_idx).nonzero()[0].item()

        new_int_bottom = cum_probs[selection - 1] if selection > 0 else cur_interval[0]
        new_int_top = cum_probs[selection]

        new_int_bottom_bits_inc = msb_int2bits(int(new_int_bottom), precision)
        new_int_top_bits_inc = msb_int2bits(int(new_int_top - 1), precision)
        num_bits_encoded = num_same_from_beg(new_int_bottom_bits_inc, new_int_top_bits_inc)

        new_int_bottom_bits = new_int_bottom_bits_inc[num_bits_encoded:] + [0] * num_bits_encoded
        new_int_top_bits = new_int_top_bits_inc[num_bits_encoded:] + [1] * num_bits_encoded

        next_interval = [msb_bits2int(new_int_bottom_bits), msb_bits2int(new_int_top_bits) + 1]
        sampled_token_id = int(indices[selection].item())

        return EncodeResult(
            sampled_token_id=sampled_token_id,
            bits_consumed=num_bits_encoded,
            metadata={
                "cur_interval": next_interval,
                "next_bit_index": context.bit_index + num_bits_encoded,
            },
        )

    def decode(self, context: DecodeContext) -> DecodeResult:
        prob, indices = to_sorted_tensors(context.prob_table, context.indices)
        precision = context.precision
        cur_interval = list(context.cur_interval or [0, 2 ** precision])

        cur_int_range = cur_interval[1] - cur_interval[0]
        cur_threshold = 1 / cur_int_range
        if prob[-1] < cur_threshold:
            k = max(2, (prob < cur_threshold).nonzero()[0].item())
            prob = prob[:k]
            indices = indices[:k]

        prob = prob / prob.sum()
        prob = (prob * cur_int_range).round().long()

        cum_probs = prob.cumsum(0)
        overfill_index = (cum_probs > cur_int_range).nonzero()
        if len(overfill_index) > 0:
            cum_probs = cum_probs[:overfill_index[0]]

        prev = int(context.prev_token_id)
        if prev not in indices:
            return DecodeResult(bits="", metadata={"cur_interval": cur_interval})
        if len(overfill_index) > 0 and prev in indices[overfill_index]:
            return DecodeResult(bits="", metadata={"cur_interval": cur_interval})

        cum_probs += cur_int_range - cum_probs[-1]
        cum_probs += cur_interval[0]
        selection = (indices == prev).nonzero()[0].item()

        new_int_bottom = cum_probs[selection - 1] if selection > 0 else cur_interval[0]
        new_int_top = cum_probs[selection]

        new_int_bottom_bits_inc = msb_int2bits(int(new_int_bottom), precision)
        new_int_top_bits_inc = msb_int2bits(int(new_int_top - 1), precision)
        num_bits_decoded = num_same_from_beg(new_int_bottom_bits_inc, new_int_top_bits_inc)
        bits = "".join(str(b) for b in new_int_bottom_bits_inc[:num_bits_decoded])

        new_int_bottom_bits = new_int_bottom_bits_inc[num_bits_decoded:] + [0] * num_bits_decoded
        new_int_top_bits = new_int_top_bits_inc[num_bits_decoded:] + [1] * num_bits_decoded
        next_interval = [msb_bits2int(new_int_bottom_bits), msb_bits2int(new_int_top_bits) + 1]

        return DecodeResult(bits=bits, metadata={"cur_interval": next_interval})
