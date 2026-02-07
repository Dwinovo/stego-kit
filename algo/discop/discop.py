from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from algo.common import bit_slice_with_padding
from .common import DiscopCommonMixin
from core.stego_algorithm import DecodeResult, EncodeResult
from core.stego_context import DecodeContext, EncodeContext


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

    def encode(self, context: EncodeContext) -> EncodeResult:
        prg = self._require_prg(context.prg)
        probs_list, indices_list = self._prepare_inputs(context.prob_table, context.indices)

        bits_slice = bit_slice_with_padding(context.bit_stream, context.bit_index, context.precision)

        sampled_index, n_bits = self._encode_step(
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
        bits = self._decode_step(indices_list, probs_list, stego_t, prg, context.precision)
        return DecodeResult(bits=bits, metadata={"bits_len": len(bits)})
