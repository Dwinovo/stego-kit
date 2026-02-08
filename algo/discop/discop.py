from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Sequence

import torch

from algo.common import _filter_distribution, _prepare_prefix_ids, bit_slice_with_padding
from .common import DiscopCommonMixin
from core.stego_algorithm import StegoDecodeResult, StegoEncodeResult
from core.stego_context import StegoDecodeContext, StegoEncodeContext


@dataclass
class _Node:
    prob: float
    left: "_Node | None"
    right: "_Node | None"
    index: int
    search_path: int


def _is_leaf(node: _Node) -> bool:
    return node.index != -1


def _pop_min(q1: deque[_Node], q2: deque[_Node]) -> _Node:
    if q1 and q2 and q1[0].prob < q2[0].prob:
        return q1.popleft()
    if not q1:
        return q2.popleft()
    if not q2:
        return q1.popleft()
    return q2.popleft()


def _create_huffman_tree(indices: list[int], probs: list[float], search_for: int) -> _Node:
    q1: deque[_Node] = deque()
    q2: deque[_Node] = deque()

    for i in range(len(indices) - 1, -1, -1):
        search_path = 0 if search_for == indices[i] else 9
        q1.append(_Node(probs[i], None, None, indices[i], search_path))

    while len(q1) + len(q2) > 1:
        first = _pop_min(q1, q2)
        second = _pop_min(q1, q2)
        search_path = 9
        if first.search_path != 9:
            search_path = -1
        elif second.search_path != 9:
            search_path = 1
        q2.append(_Node(first.prob + second.prob, first, second, -1, search_path))

    return q2[0] if q2 else q1[0]


class DiscopStrategy(DiscopCommonMixin):
    """Discop strategy."""

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
                raise RuntimeError("DiscopStrategy._encode_token_step returned sampled_token_id=None")
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
    def _encode_step(indices: list[int], probs: list[float], message_bits: str, bit_index: int, prg, precision: int):
        node = _create_huffman_tree(indices, probs, -1)
        n_bits = 0

        while not _is_leaf(node):
            prob_sum = node.prob
            ptr = prg.generate_random(n=precision)
            ptr_0 = ptr * prob_sum
            ptr_1 = (ptr + 0.5) * prob_sum
            if ptr_1 > prob_sum:
                ptr_1 -= prob_sum

            partition = node.left.prob
            path_0 = -1 if ptr_0 < partition else 1
            path_1 = -1 if ptr_1 < partition else 1

            bit = int(message_bits[n_bits + bit_index])
            path = path_0 if bit == 0 else path_1
            node = node.right if path == 1 else node.left

            if path_0 != path_1:
                n_bits += 1

        return node.index, n_bits

    @staticmethod
    def _decode_step(indices: list[int], probs: list[float], stego_t: int, prg, precision: int) -> str:
        node = _create_huffman_tree(indices, probs, stego_t)
        message_decoded_t = ""

        while not _is_leaf(node):
            prob_sum = node.prob
            ptr = prg.generate_random(n=precision)
            ptr_0 = ptr * prob_sum
            ptr_1 = (ptr + 0.5) * prob_sum
            if ptr_1 > prob_sum:
                ptr_1 -= prob_sum

            partition = node.left.prob
            path_0 = -1 if ptr_0 < partition else 1
            path_1 = -1 if ptr_1 < partition else 1

            if path_0 != path_1:
                if node.search_path == 9:
                    return ""
                if path_0 == -1:
                    path_table_swap = {-1: "0", 1: "1"}
                else:
                    path_table_swap = {-1: "1", 1: "0"}
                message_decoded_t += path_table_swap[node.search_path]
                node = node.left if node.search_path == -1 else node.right
            else:
                node = node.left if path_0 == -1 else node.right

        if node.search_path != 0:
            return ""
        return message_decoded_t

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

        sampled_index, n_bits = self._encode_step(
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
        bits = self._decode_step(indices_list, probs_list, stego_t, prg, precision)
        return {"bits": bits, "bits_len": len(bits)}
