from __future__ import annotations

import time
from typing import Any, Sequence

import torch

from stegokit.algo.common import _filter_distribution, _prepare_prefix_ids, _stop_on_eos, bit_slice_with_padding
from stegokit.core.stego_algorithm import StegoDecodeResult, StegoEncodeResult
from stegokit.core.stego_context import StegoDecodeContext, StegoEncodeContext


def _find_nearest_list(prob: list[float], delta: float) -> list[int]:
    if not prob:
        return []

    diff = [(item - delta) ** 2 for item in prob]
    tmp_idx = min(range(len(prob)), key=lambda idx: diff[idx])

    if prob[tmp_idx] < delta:
        result = [tmp_idx]
        if tmp_idx != len(prob) - 1:
            tmp_sum = prob[tmp_idx]
            for i in range(tmp_idx + 1, len(prob) - 1):
                if delta > (tmp_sum + prob[i]):
                    tmp_sum += prob[i]
                    result.append(i)
        return result

    if tmp_idx >= len(prob) - 2:
        return [tmp_idx]

    new_idx = tmp_idx + 1
    idx = [new_idx]
    idx.extend(item + new_idx + 1 for item in _find_nearest_list(prob[new_idx + 1 :], delta - prob[new_idx]))
    if (delta - sum(prob[item] for item in idx)) ** 2 > diff[tmp_idx]:
        return [tmp_idx]
    return idx


def _group_distribution(prob: torch.Tensor) -> tuple[tuple[float, list[int]], tuple[float, list[int]]]:
    if prob.numel() == 0:
        raise ValueError("ADG grouping requires a non-empty probability tensor")

    sorted_prob, sorted_positions = torch.sort(prob, descending=True)
    prob_list = [float(item) for item in sorted_prob.tolist()]
    positions = [int(item) for item in sorted_positions.tolist()]

    mean = 0.5
    group0_prob = prob_list[0]
    group0_positions = [positions[0]]

    if group0_prob > mean:
        return (group0_prob, group0_positions), (max(0.0, 1.0 - group0_prob), positions[1:])

    del prob_list[0]
    del positions[0]
    delta = mean - group0_prob

    while prob_list and prob_list[-1] < 2 * delta:
        idx_list = _find_nearest_list(prob_list, delta)
        for idx in sorted(idx_list, reverse=True):
            group0_prob += prob_list[idx]
            group0_positions.append(positions[idx])
            del prob_list[idx]
            del positions[idx]
        delta = mean - group0_prob

    return (group0_prob, group0_positions), (max(0.0, 1.0 - group0_prob), positions)


class ADGStrategy:
    """Adaptive Dynamic Grouping strategy adapted from adg.py."""

    def encode(self, context: StegoEncodeContext) -> StegoEncodeResult:
        encode_started_at = time.perf_counter()
        prefix_ids = _prepare_prefix_ids(context.messages, context.model, context.tokenizer)
        x = prefix_ids
        past_key_values = None
        bit_index = 0
        eos_token_id = getattr(context.tokenizer, "eos_token_id", None)
        stop_on_eos = _stop_on_eos(context, default=True)
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
                cur_interval=None,
                extra=context.extra,
            )
            sampled_token_id = er.get("sampled_token_id")
            if sampled_token_id is None:
                raise RuntimeError("ADGStrategy._encode_token_step returned sampled_token_id=None")
            token_id = int(sampled_token_id)
            generated_ids.append(token_id)
            bits_consumed = int(er.get("bits_consumed", 0))
            bit_index = int(er.get("next_bit_index", bit_index + bits_consumed))
            x = torch.tensor([[token_id]], device=prefix_ids.device, dtype=torch.long)

            if stop_on_eos and eos_token_id is not None and token_id == int(eos_token_id):
                break

        text = context.tokenizer.decode(generated_ids)
        effective_consumed_bits = min(bit_index, len(context.secret_bits))
        generated_steps = len(generated_ids)
        encode_time_seconds = time.perf_counter() - encode_started_at
        embedding_capacity = (effective_consumed_bits / generated_steps) if generated_steps > 0 else 0.0
        return StegoEncodeResult(
            generated_token_ids=generated_ids,
            consumed_bits=effective_consumed_bits,
            text=text,
            encode_time_seconds=encode_time_seconds,
            embedding_capacity=embedding_capacity,
            metadata={
                "algorithm": context.algorithm,
                "final_bit_index": bit_index,
                "generated_steps": generated_steps,
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
                cur_interval=None,
                extra=context.extra,
            )
            bits = str(dr.get("bits", ""))
            recovered_parts.append(bits)
            recovered_len += len(bits)
            decoded_steps += 1
            x = torch.tensor([[int(token_id)]], device=prefix_ids.device, dtype=torch.long)

            if context.max_bits is not None and recovered_len >= context.max_bits:
                break

        bits = "".join(recovered_parts)
        if context.max_bits is not None:
            bits = bits[: context.max_bits]
        return StegoDecodeResult(
            bits=bits,
            decode_time_seconds=time.perf_counter() - decode_started_at,
            metadata={"algorithm": context.algorithm, "decoded_steps": decoded_steps},
        )

    @staticmethod
    def _resolve_params(extra: dict[str, Any] | None) -> tuple[float, int]:
        extra = extra or {}
        epsilon = float(extra.get("epsilon", 0.01))
        max_bit = int(extra.get("max_bit", 15))
        if epsilon <= 0:
            raise ValueError("ADG strategy requires extra['epsilon'] > 0")
        if max_bit < 1:
            raise ValueError("ADG strategy requires extra['max_bit'] >= 1")
        return epsilon, max_bit

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
        del precision, prg, cur_interval
        epsilon, max_bit = self._resolve_params(extra)

        prob = torch.tensor(prob_table, dtype=torch.float64)
        token_indices = torch.tensor(indices, dtype=torch.long)
        if prob.numel() == 0:
            raise ValueError("ADG strategy requires a non-empty token distribution")

        bits_slice = bit_slice_with_padding(bit_stream, bit_index, max_bit)
        bit_tmp = 0

        while bit_tmp < max_bit:
            (group0_prob, group0_positions), (_, group1_positions) = _group_distribution(prob)
            if not (abs(group0_prob - 0.5) <= epsilon * (2 ** bit_tmp) and abs(group0_prob - 0.5) < 0.5):
                break

            next_positions = group0_positions if bits_slice[bit_tmp] == "0" else group1_positions
            if not next_positions:
                break

            prob = prob[next_positions]
            token_indices = token_indices[next_positions]
            prob = prob / prob.sum()
            bit_tmp += 1

        selection = int(torch.multinomial(prob.to(dtype=torch.float32), 1).item())
        sampled_token_id = int(token_indices[selection].item())
        return {
            "sampled_token_id": sampled_token_id,
            "bits_consumed": bit_tmp,
            "next_bit_index": bit_index + bit_tmp,
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
        del precision, prg, cur_interval
        epsilon, max_bit = self._resolve_params(extra)

        prob = torch.tensor(prob_table, dtype=torch.float64)
        token_indices = torch.tensor(indices, dtype=torch.long)
        prev = int(prev_token_id)
        if prob.numel() == 0:
            return {"bits": "", "bits_len": 0}
        if not bool((token_indices == prev).any()):
            return {"bits": "", "bits_len": 0}

        recovered_bits: list[str] = []
        bit_tmp = 0
        while bit_tmp < max_bit:
            (group0_prob, group0_positions), (_, group1_positions) = _group_distribution(prob)
            if not (abs(group0_prob - 0.5) <= epsilon * (2 ** bit_tmp) and abs(group0_prob - 0.5) < 0.5):
                break

            if group0_positions and bool((token_indices[group0_positions] == prev).any()):
                next_positions = group0_positions
                recovered_bits.append("0")
            elif group1_positions and bool((token_indices[group1_positions] == prev).any()):
                next_positions = group1_positions
                recovered_bits.append("1")
            else:
                break

            prob = prob[next_positions]
            token_indices = token_indices[next_positions]
            prob = prob / prob.sum()
            bit_tmp += 1

        bits = "".join(recovered_bits)
        return {"bits": bits, "bits_len": len(bits)}
