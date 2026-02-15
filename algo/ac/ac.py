from __future__ import annotations

import time
from typing import Any, Sequence

import torch

from algo.common import (
    _filter_distribution,
    _prepare_prefix_ids,
    _stop_on_eos,
    bit_slice_with_padding,
    msb_bits2int,
    msb_int2bits,
    num_same_from_beg,
    to_sorted_tensors,
)
from core.stego_algorithm import StegoDecodeResult, StegoEncodeResult
from core.stego_context import StegoDecodeContext, StegoEncodeContext
from utils.entropy import shannon_entropy


class ACStrategy:
    """Arithmetic-coding-based steganography strategy."""

    def encode(self, context: StegoEncodeContext) -> StegoEncodeResult:
        encode_started_at = time.perf_counter()
        prefix_ids = _prepare_prefix_ids(context.messages, context.model, context.tokenizer)
        x = prefix_ids
        past_key_values = None
        bit_index = 0
        cur_interval = None
        eos_token_id = getattr(context.tokenizer, "eos_token_id", None)
        stop_on_eos = _stop_on_eos(context, default=True)
        generated_ids: list[int] = []
        entropy_sum = 0.0
        entropy_steps = 0

        for _ in range(context.max_new_tokens):
            with torch.no_grad():
                output = context.model(input_ids=x, past_key_values=past_key_values, use_cache=True)
            logits = output.logits[0, -1, :]
            past_key_values = getattr(output, "past_key_values", None)
            probs, token_indices = _filter_distribution(logits, context.temperature, context.top_k, context.top_p)
            prob_table = probs.tolist()
            entropy_sum += shannon_entropy(prob_table)
            entropy_steps += 1

            er = self._encode_token_step(
                prob_table=prob_table,
                indices=token_indices.tolist(),
                bit_stream=context.secret_bits,
                bit_index=bit_index,
                precision=context.precision,
                prg=context.prg,
                cur_interval=cur_interval,
                extra=context.extra,
            )
            sampled_token_id = er.get("sampled_token_id")
            if sampled_token_id is None:
                raise RuntimeError("ACStrategy._encode_token_step returned sampled_token_id=None")
            token_id = int(sampled_token_id)
            generated_ids.append(token_id)
            bits_consumed = int(er.get("bits_consumed", 0))
            bit_index = int(er.get("next_bit_index", bit_index + bits_consumed))
            cur_interval = er.get("cur_interval", cur_interval)
            x = torch.tensor([[token_id]], device=prefix_ids.device, dtype=torch.long)

            if stop_on_eos and eos_token_id is not None and token_id == int(eos_token_id):
                break

        text = context.tokenizer.decode(generated_ids)
        effective_consumed_bits = min(bit_index, len(context.secret_bits))
        generated_steps = len(generated_ids)
        average_entropy = entropy_sum / entropy_steps if entropy_steps > 0 else 0.0
        encode_time_seconds = time.perf_counter() - encode_started_at
        embedding_capacity = (effective_consumed_bits / generated_steps) if generated_steps > 0 else 0.0
        return StegoEncodeResult(
            generated_token_ids=generated_ids,
            consumed_bits=effective_consumed_bits,
            text=text,
            average_entropy=average_entropy,
            encode_time_seconds=encode_time_seconds,
            embedding_capacity=embedding_capacity,
            metadata={
                "algorithm": context.algorithm,
                "final_bit_index": bit_index,
                "cur_interval": cur_interval,
                "generated_steps": len(generated_ids),
                "embedded_bits": context.secret_bits[:effective_consumed_bits],
            },
        )

    def decode(self, context: StegoDecodeContext) -> StegoDecodeResult:
        decode_started_at = time.perf_counter()
        if not context.generated_token_ids:
            return StegoDecodeResult(
                bits="",
                decode_time_seconds=time.perf_counter() - decode_started_at,
                metadata={"decoded_steps": 0},
            )

        prefix_ids = _prepare_prefix_ids(context.messages, context.model, context.tokenizer)
        x = prefix_ids
        past_key_values = None
        cur_interval = None
        recovered_parts: list[str] = []
        recovered_len = 0
        decoded_steps = 0

        for token_id in context.generated_token_ids:
            with torch.no_grad():
                output = context.model(input_ids=x, past_key_values=past_key_values, use_cache=True)
            logits = output.logits[0, -1, :]
            past_key_values = getattr(output, "past_key_values", None)
            probs, token_indices = _filter_distribution(logits, context.temperature, context.top_k, context.top_p)

            dr = self._decode_token_step(
                prob_table=probs.tolist(),
                indices=token_indices.tolist(),
                prev_token_id=int(token_id),
                precision=context.precision,
                prg=context.prg,
                cur_interval=cur_interval,
                extra=context.extra,
            )
            bits = str(dr.get("bits", ""))
            recovered_parts.append(bits)
            recovered_len += len(bits)
            decoded_steps += 1
            cur_interval = dr.get("cur_interval", cur_interval)
            x = torch.tensor([[int(token_id)]], device=prefix_ids.device, dtype=torch.long)

            if context.max_bits is not None and recovered_len >= context.max_bits:
                break

        bits = "".join(recovered_parts)
        if context.max_bits is not None:
            bits = bits[: context.max_bits]
        return StegoDecodeResult(
            bits=bits,
            decode_time_seconds=time.perf_counter() - decode_started_at,
            metadata={"algorithm": context.algorithm, "cur_interval": cur_interval, "decoded_steps": decoded_steps},
        )

    def _encode_token_step(
        self,
        *,
        prob_table: Sequence[float],
        indices: Sequence[int],
        bit_stream: str,
        bit_index: int,
        precision: int,
        prg: Any | None,
        cur_interval: list[int] | None,
        extra: dict[str, Any] | None,
    ) -> dict[str, Any]:
        del prg, extra
        prob, indices = to_sorted_tensors(prob_table, indices)
        cur_interval = list(cur_interval or [0, 2 ** precision])

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

        bits_slice = bit_slice_with_padding(bit_stream, bit_index, precision)
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

        return {
            "sampled_token_id": sampled_token_id,
            "bits_consumed": num_bits_encoded,
            "cur_interval": next_interval,
            "next_bit_index": bit_index + num_bits_encoded,
        }

    def _decode_token_step(
        self,
        *,
        prob_table: Sequence[float],
        indices: Sequence[int],
        prev_token_id: int,
        precision: int,
        prg: Any | None,
        cur_interval: list[int] | None,
        extra: dict[str, Any] | None,
    ) -> dict[str, Any]:
        del prg, extra
        prob, indices = to_sorted_tensors(prob_table, indices)
        cur_interval = list(cur_interval or [0, 2 ** precision])

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

        prev = int(prev_token_id)
        if prev not in indices:
            return {"bits": "", "bits_len": 0, "cur_interval": cur_interval}
        if len(overfill_index) > 0 and prev in indices[overfill_index]:
            return {"bits": "", "bits_len": 0, "cur_interval": cur_interval}

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

        return {"bits": bits, "bits_len": len(bits), "cur_interval": next_interval}
