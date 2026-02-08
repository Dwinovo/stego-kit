from __future__ import annotations

from typing import Any, Sequence

import torch

from algo.common import (
    _filter_distribution,
    _prepare_prefix_ids,
    bit_slice_with_padding,
    bits2int,
    int2bits,
    num_same_from_beg,
    require_prg_method,
    to_sorted_tensors,
)
from core.stego_algorithm import StegoDecodeResult, StegoEncodeResult
from core.stego_context import StegoDecodeContext, StegoEncodeContext

class MeteorStrategy:
    def encode(self, context: StegoEncodeContext) -> StegoEncodeResult:
        prefix_ids = _prepare_prefix_ids(context.messages, context.model, context.tokenizer)
        x = prefix_ids
        past_key_values = None
        bit_index = 0
        cur_interval = None
        eos_token_id = getattr(context.tokenizer, "eos_token_id", None)
        generated_ids: list[int] = []

        for _ in range(context.max_new_tokens):
            with torch.no_grad():
                output = context.model(input_ids=x, past_key_values=past_key_values, use_cache=True)
            logits = output.logits[0, -1, :]
            past_key_values = getattr(output, "past_key_values", None)
            probs, token_indices = _filter_distribution(logits, context.temperature, context.top_k, context.top_p)

            er = self._encode_token_step(
                prob_table=probs.tolist(),
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
                raise RuntimeError("MeteorStrategy._encode_token_step returned sampled_token_id=None")
            token_id = int(sampled_token_id)
            generated_ids.append(token_id)
            bits_consumed = int(er.get("bits_consumed", 0))
            bit_index = int(er.get("next_bit_index", bit_index + bits_consumed))
            cur_interval = er.get("cur_interval", cur_interval)
            x = torch.tensor([[token_id]], device=prefix_ids.device, dtype=torch.long)

            if eos_token_id is not None and token_id == int(eos_token_id):
                break

        text = context.tokenizer.decode(generated_ids)
        effective_consumed_bits = min(bit_index, len(context.secret_bits))
        return StegoEncodeResult(
            generated_token_ids=generated_ids,
            consumed_bits=effective_consumed_bits,
            text=text,
            metadata={
                "algorithm": context.algorithm,
                "final_bit_index": bit_index,
                "cur_interval": cur_interval,
                "generated_steps": len(generated_ids),
                "embedded_bits": context.secret_bits[:effective_consumed_bits],
            },
        )

    def decode(self, context: StegoDecodeContext) -> StegoDecodeResult:
        if not context.generated_token_ids:
            return StegoDecodeResult(bits="", metadata={"decoded_steps": 0})

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
            metadata={"algorithm": context.algorithm, "cur_interval": cur_interval, "decoded_steps": decoded_steps},
        )
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
        prob, indices = self._prepare_inputs(prob_table, indices, precision)
        cum_probs = self._build_cum_probs(prob, precision)

        bits_slice = bit_slice_with_padding(bit_stream, bit_index, precision)
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
        return {
            "sampled_token_id": sampled_token_id,
            "bits_consumed": bits_consumed,
            "next_bit_index": bit_index + bits_consumed,
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
        prob, indices = self._prepare_inputs(prob_table, indices, precision)
        cum_probs = self._build_cum_probs(prob, precision)

        prev = int(prev_token_id)
        positions = (indices == prev).nonzero()
        if len(positions) == 0:
            return {"bits": "", "bits_len": 0}
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
        return {"bits": bits, "bits_len": len(bits)}
