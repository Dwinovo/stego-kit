from __future__ import annotations

import math
import time
from typing import Any, Sequence

import torch

from algo.common import _filter_distribution, _prepare_prefix_ids, _stop_on_eos, bit_slice_with_padding
from .common import DiscopCommonMixin
from core.stego_algorithm import StegoDecodeResult, StegoEncodeResult
from core.stego_context import StegoDecodeContext, StegoEncodeContext
from utils.entropy import shannon_entropy


class DiscopBaseStrategy(DiscopCommonMixin):
    """Discop-base strategy."""

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
                raise RuntimeError("DiscopBaseStrategy._encode_token_step returned sampled_token_id=None")
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
        del cur_interval, extra
        prg = self._require_prg(prg)
        probs_list, indices_list = self._prepare_inputs(prob_table, indices)

        bits_slice = bit_slice_with_padding(bit_stream, bit_index, precision)

        sampled_index, n_bits = self._baseline_encode_step(
            indices_list, probs_list, bits_slice, 0, prg, precision
        )

        return {
            "sampled_token_id": int(sampled_index),
            "bits_consumed": int(n_bits),
            "next_bit_index": bit_index + int(n_bits),
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
        del cur_interval, extra
        prg = self._require_prg(prg)
        probs_list, indices_list = self._prepare_inputs(prob_table, indices)
        stego_t = int(prev_token_id)
        bits = self._baseline_decode_step(indices_list, probs_list, stego_t, prg, precision)
        return {"bits": bits, "bits_len": len(bits)}
