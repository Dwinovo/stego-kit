from __future__ import annotations

import math
import random
from typing import Any, Sequence

import numpy as np
import torch

from algo.common import _prepare_prefix_ids
from core.stego_algorithm import StegoDecodeResult, StegoEncodeResult
from core.stego_context import StegoDecodeContext, StegoEncodeContext


class AsymmetricStrategy:
    """Asymmetric steganography strategy adapted from demo.ipynb/fast_decode.py."""

    @staticmethod
    def _sampling_function(x, func_type: int):
        if func_type == 0:
            return np.cos(np.pi * x)
        if func_type == 1:
            return -np.sign(x - 0.5)
        if func_type == 2:
            return np.log2(2 - x) - 0.5573
        raise ValueError(f"Unsupported func_type: {func_type}")

    @staticmethod
    def _get_extra(context, key: str, default):
        return context.extra.get(key, default) if context.extra else default

    @staticmethod
    def _bit_length(tokenizer) -> int:
        return int(math.ceil(math.log2(int(tokenizer.vocab_size))))

    @staticmethod
    def _int_to_bin(number: int, length: int) -> str:
        return format(int(number), f"0{length}b")

    @staticmethod
    def _prepare_prompt_ids(context: StegoEncodeContext | StegoDecodeContext) -> torch.Tensor:
        use_chat_template = bool(context.extra.get("use_chat_template", False)) if context.extra else False
        prompt = context.prompt or ""
        if use_chat_template and hasattr(context.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            return context.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            ).to(context.model.device)
        return _prepare_prefix_ids(prompt, context.model, context.tokenizer)

    @staticmethod
    def _interval_sum(probs: torch.Tensor, start: int, end: int) -> float:
        n = int(probs.shape[0])
        if start >= n:
            return 0.0
        e = min(end, n)
        if e <= start:
            return 0.0
        return float(probs[start:e].sum().item())

    def encode(self, context: StegoEncodeContext) -> StegoEncodeResult:
        secure_parameter = int(self._get_extra(context, "secure_parameter", 32))
        func_type = int(self._get_extra(context, "func_type", 0))
        seed = str(self._get_extra(context, "seed", "12345"))

        threshold = 2 ** (-secure_parameter)
        length = self._bit_length(context.tokenizer)
        vocab_size = int(context.tokenizer.vocab_size)

        prompt_ids = self._prepare_prompt_ids(context)
        x = prompt_ids
        past_key_values = None
        eos_token_id = getattr(context.tokenizer, "eos_token_id", None)

        generated_ids: list[int] = []
        bit_cnt = 0
        cnt_v = 0
        sum_v = 0.0
        segment = ""
        segments: list[str] = []
        step_scores: list[float] = []
        entropy_acc = 0.0

        if len(context.secret_bits) == 0:
            return StegoEncodeResult(generated_token_ids=[], consumed_bits=0, text="", metadata={"algorithm": context.algorithm})

        current_bit = context.secret_bits[0]

        with torch.no_grad():
            for _ in range(context.max_new_tokens):
                if bit_cnt >= len(context.secret_bits):
                    break

                output = context.model(input_ids=x, past_key_values=past_key_values, use_cache=True)
                logits = output.logits[0, -1, :]
                past_key_values = getattr(output, "past_key_values", None)
                probs = torch.softmax(logits / max(float(context.temperature), 1e-8), dim=-1)

                group_left = 0
                group_right = 2 ** length
                split = 2 ** (length - 1)

                for _ in range(length):
                    if bit_cnt >= len(context.secret_bits):
                        break

                    random.seed(seed + str(cnt_v))
                    random_number = random.random()

                    left_mass = self._interval_sum(probs, group_left, split)
                    right_mass = self._interval_sum(probs, split, group_right)
                    total_mass = left_mass + right_mass
                    if total_mass <= 0:
                        break

                    current_bit = context.secret_bits[bit_cnt]
                    left_ratio = left_mass / total_mass
                    right_ratio = right_mass / total_mass

                    if current_bit == "0":
                        if random_number <= left_ratio:
                            next_bit = "0"
                            sum_v += float(self._sampling_function(random_number, func_type))
                            entropy_acc += -math.log(max(left_ratio, 1e-12))
                            group_right = split
                        else:
                            next_bit = "1"
                            sum_v += float(self._sampling_function(1 - random_number, func_type))
                            entropy_acc += -math.log(max(right_ratio, 1e-12))
                            group_left = split
                    else:
                        if random_number <= right_ratio:
                            next_bit = "1"
                            sum_v += float(self._sampling_function(1 - random_number, func_type))
                            entropy_acc += -math.log(max(right_ratio, 1e-12))
                            group_left = split
                        else:
                            next_bit = "0"
                            sum_v += float(self._sampling_function(random_number, func_type))
                            entropy_acc += -math.log(max(left_ratio, 1e-12))
                            group_right = split

                    cnt_v += 1
                    split = int(math.ceil((group_left + group_right) / 2))
                    delta = sum_v / max(cnt_v, 1)
                    crit = math.exp(-2 * (delta ** 2) * cnt_v)
                    segment += next_bit
                    step_scores.append(entropy_acc)

                    # Same stop rule as demo: confirm one secret bit and move to next segment.
                    if crit <= threshold and delta * (0.5 - int(current_bit)) >= 0:
                        seed = seed + "a"
                        sum_v = 0.0
                        cnt_v = 0
                        bit_cnt += 1
                        segments.append(segment)
                        segment = ""

                token_id = max(0, min(int(group_left), vocab_size - 1))
                generated_ids.append(token_id)
                x = torch.tensor([[token_id]], device=prompt_ids.device, dtype=torch.long)

                if eos_token_id is not None and token_id == int(eos_token_id):
                    break

        text = context.tokenizer.decode(generated_ids)
        consumed = min(bit_cnt, len(context.secret_bits))
        return StegoEncodeResult(
            generated_token_ids=generated_ids,
            consumed_bits=consumed,
            text=text,
            metadata={
                "algorithm": context.algorithm,
                "decode_mode_default": "regular",
                "secure_parameter": secure_parameter,
                "func_type": func_type,
                "segments": segments,
                "step_scores": step_scores,
            },
        )

    def decode(self, context: StegoDecodeContext) -> StegoDecodeResult:
        mode = str(self._get_extra(context, "decode_mode", "regular")).lower()
        if mode == "robust":
            bits, spans = self._decode_robust(context)
            if context.max_bits is not None:
                bits = bits[: context.max_bits]
            return StegoDecodeResult(bits=bits, metadata={"algorithm": context.algorithm, "decode_mode": "robust", "spans": spans})

        bits, strings_per_bit = self._decode_regular(context)
        if context.max_bits is not None:
            bits = bits[: context.max_bits]
        return StegoDecodeResult(
            bits=bits,
            metadata={"algorithm": context.algorithm, "decode_mode": "regular", "strings_per_bit": strings_per_bit},
        )

    def _decode_regular(self, context: StegoDecodeContext) -> tuple[str, list[str]]:
        secure_parameter = int(self._get_extra(context, "secure_parameter", 32))
        func_type = int(self._get_extra(context, "func_type", 0))
        seed = str(self._get_extra(context, "seed", "12345"))
        threshold = 2 ** (-secure_parameter)
        length = self._bit_length(context.tokenizer)

        output_bits = "".join(self._int_to_bin(token_id, length) for token_id in context.generated_token_ids)
        secret_bits = ""
        strings_per_bit: list[str] = []
        sum_v = 0.0
        cnt_v = 0
        current_str = ""

        for bit in output_bits:
            random.seed(seed + str(cnt_v))
            random_number = random.random()
            val = int(bit) * self._sampling_function(1 - random_number, func_type) + (1 - int(bit)) * self._sampling_function(
                random_number, func_type
            )
            sum_v += float(val)
            cnt_v += 1

            delta = sum_v / cnt_v
            crit = math.exp(-2 * (delta ** 2) * cnt_v)
            current_str += bit

            if crit <= threshold and delta >= 0:
                secret_bits += "0"
                sum_v = 0.0
                cnt_v = 0
                seed = seed + "a"
                strings_per_bit.append(current_str)
                current_str = ""
                continue
            if crit <= threshold and delta <= 0:
                secret_bits += "1"
                sum_v = 0.0
                cnt_v = 0
                seed = seed + "a"
                strings_per_bit.append(current_str)
                current_str = ""
                continue

        return secret_bits, strings_per_bit

    def _robust_decode_1_bit(self, output_bits: Sequence[int], r: Sequence[float], func_type: int, threshold: float) -> tuple[str, int]:
        bits_np = np.asarray(output_bits, dtype=int)
        r_np = np.asarray(r, dtype=float)
        terms = (1 - bits_np) * self._sampling_function(r_np, func_type) + bits_np * self._sampling_function(1 - r_np, func_type)
        cumsum = np.cumsum(terms)
        mu = np.arange(len(bits_np)) + 1
        delta = cumsum / mu
        crit = np.exp(-2 * (delta ** 2) * mu)
        valid = crit <= threshold
        if valid.any():
            idx = int(valid.argmax())
            bit = "1" if delta[idx] <= 0 else "0"
            return bit, idx + 1
        return "", len(output_bits)

    def _decode_robust(self, context: StegoDecodeContext) -> tuple[str, list[dict[str, Any]]]:
        secure_parameter = int(self._get_extra(context, "secure_parameter", 32))
        func_type = int(self._get_extra(context, "func_type", 0))
        seed = str(self._get_extra(context, "seed", "12345"))
        search_window = int(self._get_extra(context, "robust_search_window", 1000))
        threshold = 2 ** (-secure_parameter)
        length = self._bit_length(context.tokenizer)

        output_bits = "".join(self._int_to_bin(token_id, length) for token_id in context.generated_token_ids)
        bit_arr = [int(b) for b in output_bits]

        secret_bits = ""
        bit_num = 0
        spans: list[dict[str, Any]] = []

        while bit_num < len(bit_arr):
            r = []
            for k in range(len(bit_arr)):
                random.seed(seed + str(k))
                r.append(random.random())

            found = False
            upper = min(bit_num + search_window, len(bit_arr))
            for cnt in range(bit_num, upper):
                secret_bit, cnt_v = self._robust_decode_1_bit(bit_arr[cnt:], r[: len(r) - cnt], func_type, threshold)
                if secret_bit:
                    start = cnt
                    end = bit_num + cnt_v
                    secret_bits += secret_bit
                    spans.append(
                        {
                            "bit": secret_bit,
                            "start_bit_index": start,
                            "end_bit_index": end,
                            "start_token_index": start // length,
                            "end_token_index": end // length,
                        }
                    )
                    bit_num += cnt_v
                    seed = seed + "a"
                    found = True
                    break

            if not found:
                bit_num = upper

        return secret_bits, spans
