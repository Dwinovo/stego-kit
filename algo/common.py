from __future__ import annotations

from typing import Any, Sequence

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


def _model_device(model) -> torch.device:
    if hasattr(model, "device"):
        return torch.device(model.device)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _prepare_prefix_ids(context_messages: Sequence[dict[str, Any]], model, tokenizer) -> torch.Tensor:
    device = _model_device(model)
    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError("tokenizer must support apply_chat_template")
    input_ids = tokenizer.apply_chat_template(
        list(context_messages),
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.tensor(input_ids, dtype=torch.long)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    return input_ids.to(device)


def _filter_distribution(
    logits: torch.Tensor,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    t = max(float(temperature), 1e-8)
    probs = torch.softmax(logits / t, dim=-1)
    token_indices = torch.arange(probs.shape[-1], device=probs.device)

    if top_k is not None and top_k > 0 and top_k < probs.shape[-1]:
        probs, top_idx = torch.topk(probs, k=top_k)
        token_indices = token_indices[top_idx]

    if top_p is not None and 0 < top_p < 1:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=0)
        keep = cumsum <= top_p
        if keep.numel() > 0:
            keep[0] = True
        probs = sorted_probs[keep]
        token_indices = token_indices[sorted_idx[keep]]

    probs = probs / probs.sum()
    return probs, token_indices
