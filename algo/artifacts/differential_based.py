from __future__ import annotations

from typing import Any, Sequence

import torch

from algo.common import _filter_distribution, _prepare_prefix_ids, _stop_on_eos
from .common import ArtifactsCommonMixin
from core.stego_algorithm import StegoDecodeResult, StegoEncodeResult
from core.stego_context import StegoDecodeContext, StegoEncodeContext


class DifferentialBasedStrategy(ArtifactsCommonMixin):
    def encode(self, context: StegoEncodeContext) -> StegoEncodeResult:
        prefix_ids = _prepare_prefix_ids(context.messages, context.model, context.tokenizer)
        x = prefix_ids
        past_key_values = None
        bit_index = 0
        cur_interval = None
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
                cur_interval=cur_interval,
                extra=context.extra,
            )
            sampled_token_id = er.get("sampled_token_id")
            if sampled_token_id is None:
                raise RuntimeError("DifferentialBasedStrategy._encode_token_step returned sampled_token_id=None")
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
        prob, indices = self._to_tensors(prob_table, indices)

        indices_nonzero, bins, prob_new = self._differential_based_recombination(prob, indices)
        prob_new = prob_new / prob_new.sum()

        random_p = prg.generate_random(n=precision)
        cdf = torch.cumsum(prob_new, dim=0)
        bin_idx = torch.searchsorted(cdf, random_p).item()
        bin_vals = indices_nonzero[bins[bin_idx]:]

        idx, bits = self._uni_cyclic_shift_enc(
            bit_stream=bit_stream[bit_index:],
            n=len(bin_vals),
            prg=prg,
            precision=precision,
        )
        sampled_token_id = int(bin_vals[idx].item())
        bits_used = len(bits)
        return {"sampled_token_id": sampled_token_id, "bits_consumed": bits_used, "next_bit_index": bit_index + bits_used}

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
        prob, indices = self._to_tensors(prob_table, indices)

        indices_nonzero, bins, prob_new = self._differential_based_recombination(prob, indices)
        prob_new = prob_new / prob_new.sum()

        random_p = prg.generate_random(n=precision)
        cdf = torch.cumsum(prob_new, dim=0)
        bin_idx = torch.searchsorted(cdf, random_p).item()
        bin_vals = indices_nonzero[bins[bin_idx]:]

        prev = int(prev_token_id)
        pos = (bin_vals == prev).nonzero()
        if len(pos) == 0:
            return {"bits": "", "bits_len": 0}
        idx = int(pos.item())
        bits = self._uni_cyclic_shift_dec(idx=idx, n=len(bin_vals), prg=prg, precision=precision)
        return {"bits": bits, "bits_len": len(bits)}
