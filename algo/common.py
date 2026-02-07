from __future__ import annotations

from typing import Sequence

import torch


def require_prg_method(prg, method: str, strategy_name: str):
    if prg is None or not hasattr(prg, method):
        raise ValueError(f"{strategy_name} requires context.prg with method {method}(n)")
    return prg


def to_tensors(prob_table: Sequence[float], indices: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
    prob = torch.tensor(prob_table, dtype=torch.float64)
    token_indices = torch.tensor(indices, dtype=torch.long)
    return prob, token_indices


def to_sorted_tensors(prob_table: Sequence[float], indices: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
    prob, token_indices = to_tensors(prob_table, indices)
    prob, sorted_indices = torch.sort(prob, descending=True)
    token_indices = token_indices[sorted_indices]
    return prob, token_indices


def bit_slice_with_padding(bit_stream: str, bit_index: int, width: int) -> str:
    bits = bit_stream[bit_index: bit_index + width]
    if len(bits) < width:
        bits += "0" * (width - len(bits))
    return bits


def num_same_from_beg(bits1: Sequence[int], bits2: Sequence[int]) -> int:
    if len(bits1) != len(bits2):
        raise ValueError("bits length mismatch")
    for i in range(len(bits1)):
        if bits1[i] != bits2[i]:
            return i
    return len(bits1)


def msb_bits2int(bits: Sequence[int]) -> int:
    res = 0
    for i, bit in enumerate(bits[::-1]):
        res += int(bit) * (2 ** i)
    return res


def msb_int2bits(value: int, num_bits: int) -> list[int]:
    if num_bits == 0:
        return []
    return [int(ch) for ch in f"{value:0{num_bits}b}"]


def lsb_bits2int(bits: Sequence[int]) -> int:
    res = 0
    for i, bit in enumerate(bits):
        res += int(bit) * (2 ** i)
    return res


def lsb_int2bits(value: int, num_bits: int) -> list[int]:
    if num_bits == 0:
        return []
    return [int(ch) for ch in reversed(f"{value:0{num_bits}b}")]


def bits2int(bits: Sequence[int]) -> int:
    return lsb_bits2int(bits)


def int2bits(value: int, num_bits: int) -> list[int]:
    return lsb_int2bits(value, num_bits)
