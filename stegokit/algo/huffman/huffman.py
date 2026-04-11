from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Sequence

import torch

from stegokit.algo.common import _filter_distribution, _prepare_prefix_ids, _stop_on_eos, bit_slice_with_padding
from stegokit.core.algorithm_config import HuffmanConfig
from stegokit.core.stego_algorithm import StegoDecodeResult, StegoEncodeResult
from stegokit.core.stego_context import StegoDecodeContext, StegoEncodeContext


@dataclass
class _Node:
    weight: float
    order: int
    left: "_Node | None" = None
    right: "_Node | None" = None
    parent: "_Node | None" = None


def _build_huffman_codes(weights: Sequence[float]) -> list[str]:
    if not weights:
        return []

    nodes = [_Node(weight=float(weight), order=i) for i, weight in enumerate(weights)]
    if len(nodes) == 1:
        return [""]

    queue = nodes[:]
    next_order = len(queue)
    while len(queue) > 1:
        queue.sort(key=lambda item: (item.weight, item.order))
        left = queue.pop(0)
        right = queue.pop(0)
        parent = _Node(weight=left.weight + right.weight, order=next_order, left=left, right=right)
        next_order += 1
        left.parent = parent
        right.parent = parent
        queue.append(parent)

    root = queue[0]
    codes = [""] * len(nodes)
    for i, node in enumerate(nodes):
        cur = node
        while cur is not root:
            if cur.parent is None:
                raise RuntimeError("Invalid Huffman tree: node without parent before reaching root")
            codes[i] = ("0" if cur.parent.left is cur else "1") + codes[i]
            cur = cur.parent
    return codes


def _select_top_candidates(
    prob_table: Sequence[float],
    indices: Sequence[int],
    candidate_count: int,
) -> tuple[list[int], list[float]]:
    ranked = sorted(
        enumerate(zip(indices, prob_table)),
        key=lambda item: (-float(item[1][1]), item[0]),
    )
    selected = ranked[:candidate_count]
    selected_indices = [int(item[1][0]) for item in selected]
    selected_probs = [float(item[1][1]) for item in selected]
    return selected_indices, selected_probs


class HuffmanStrategy:
    """Legacy Huffman top-k coding adapted from RNN-Stega."""

    @staticmethod
    def _require_config(config: Any) -> HuffmanConfig:
        if not isinstance(config, HuffmanConfig):
            raise TypeError("Huffman strategy requires context.config to be a HuffmanConfig")
        return config

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
                config=self._require_config(context.config),
            )
            sampled_token_id = er.get("sampled_token_id")
            if sampled_token_id is None:
                raise RuntimeError("HuffmanStrategy._encode_token_step returned sampled_token_id=None")
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
                config=self._require_config(context.config),
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
    def _resolve_candidate_count(available: int, config: HuffmanConfig) -> int:
        if config.candidate_count is not None:
            candidate_count = int(config.candidate_count)
        elif config.bit_num is not None:
            candidate_count = 2 ** int(config.bit_num)
        else:
            candidate_count = 8

        if candidate_count < 1:
            raise ValueError("Huffman strategy candidate_count must be >= 1")
        return min(candidate_count, available)

    def _encode_token_step(
        self,
        *,
        prob_table: Sequence[float],
        indices: Sequence[int],
        bit_stream: str,
        bit_index: int,
        precision: int,
        config: HuffmanConfig,
    ) -> dict[str, Any]:
        del precision
        available = min(len(prob_table), len(indices))
        if available == 0:
            raise ValueError("Huffman strategy requires a non-empty token distribution")

        candidate_count = self._resolve_candidate_count(available, config)
        candidate_ids, candidate_probs = _select_top_candidates(prob_table, indices, candidate_count)
        if len(candidate_ids) == 1:
            return {
                "sampled_token_id": int(candidate_ids[0]),
                "bits_consumed": 0,
                "next_bit_index": bit_index,
            }

        codes = _build_huffman_codes(candidate_probs)
        max_code_len = max(len(code) for code in codes)
        prefix_bits = bit_slice_with_padding(bit_stream, bit_index, max_code_len)
        code_to_position = {code: pos for pos, code in enumerate(codes)}

        for width in range(1, max_code_len + 1):
            code = prefix_bits[:width]
            if code in code_to_position:
                pos = code_to_position[code]
                return {
                    "sampled_token_id": int(candidate_ids[pos]),
                    "bits_consumed": width,
                    "next_bit_index": bit_index + width,
                }

        return {
            "sampled_token_id": int(candidate_ids[0]),
            "bits_consumed": 0,
            "next_bit_index": bit_index,
        }

    def _decode_token_step(
        self,
        *,
        prob_table: Sequence[float],
        indices: Sequence[int],
        prev_token_id: int,
        precision: int,
        config: HuffmanConfig,
    ) -> dict[str, Any]:
        del precision
        available = min(len(prob_table), len(indices))
        if available == 0:
            return {"bits": "", "bits_len": 0}

        candidate_count = self._resolve_candidate_count(available, config)
        candidate_ids, candidate_probs = _select_top_candidates(prob_table, indices, candidate_count)
        codes = _build_huffman_codes(candidate_probs)
        token_to_code = {int(token_id): code for token_id, code in zip(candidate_ids, codes)}
        bits = token_to_code.get(int(prev_token_id), "")
        return {"bits": bits, "bits_len": len(bits)}
