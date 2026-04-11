from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch

from stegokit import (
    GenerationConfig,
    HuffmanConfig,
    RuntimeContext,
    StegoAlgorithm,
    StegoDecodeContext,
    StegoDispatcher,
    StegoEncodeContext,
)
from stegokit.algo.huffman.huffman import _build_huffman_codes


class _FakeTokenizer:
    def __init__(self, vocab_size: int = 32, eos_token_id: int = 3) -> None:
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id
        self.pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"):
        del tokenize, return_tensors
        ids = [1]
        for msg in messages:
            text = f"{msg.get('role', '')}:{msg.get('content', '')}"
            ids.extend((b % (self.vocab_size - 1)) + 1 for b in text.encode("utf-8"))
        if add_generation_prompt:
            ids.append(2)
        return torch.tensor([ids], dtype=torch.long)

    def __call__(self, text, return_tensors="pt"):
        del return_tensors
        ids = [1]
        if isinstance(text, str):
            ids.extend((b % (self.vocab_size - 1)) + 1 for b in text.encode("utf-8"))
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}

    def decode(self, token_ids):
        return " ".join(str(int(t)) for t in token_ids)


class _FakeModel:
    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size
        self.device = torch.device("cpu")

    def __call__(self, input_ids, past_key_values=None, use_cache=True):
        del past_key_values, use_cache
        last = int(input_ids[0, -1].item())
        ids = torch.arange(self.vocab_size, device=input_ids.device, dtype=torch.float64)
        center = (last * 5 + 9) % self.vocab_size
        logits = -((ids - center) ** 2) / 14.0
        return SimpleNamespace(logits=logits.to(torch.float32).view(1, 1, -1), past_key_values=None)


class TestHuffmanRoundTrip(unittest.TestCase):
    def test_legacy_huffman_codes_match_expected_order(self) -> None:
        codes = _build_huffman_codes([0.4, 0.3, 0.2, 0.1])
        self.assertEqual(codes, ["0", "10", "111", "110"])

    def test_huffman_encode_decode_roundtrip(self) -> None:
        tokenizer = _FakeTokenizer()
        model = _FakeModel(vocab_size=tokenizer.vocab_size)
        dispatcher = StegoDispatcher(verbose=False)

        messages = [
            {"role": "system", "content": "You are concise."},
            {"role": "user", "content": "Explain steganography briefly."},
        ]
        secret_bits = "01001101" * 24
        runtime = RuntimeContext(
            model=model,
            tokenizer=tokenizer,
            messages=messages,
            generation=GenerationConfig(
                max_new_tokens=64,
                temperature=1.0,
                top_k=16,
                precision=16,
                stop_on_eos=False,
            ),
        )
        decode_runtime = RuntimeContext(
            model=model,
            tokenizer=tokenizer,
            messages=messages,
            generation=GenerationConfig(
                temperature=1.0,
                top_k=16,
                precision=16,
            ),
        )

        enc = dispatcher.dispatch_encode(
            StegoEncodeContext(
                algorithm=StegoAlgorithm.HUFFMAN,
                runtime=runtime,
                secret_bits=secret_bits,
                config=HuffmanConfig(bit_num=3),
            )
        )

        dec = dispatcher.dispatch_decode(
            StegoDecodeContext(
                algorithm=StegoAlgorithm.HUFFMAN,
                runtime=decode_runtime,
                generated_token_ids=enc.generated_token_ids,
                max_bits=enc.consumed_bits,
                config=HuffmanConfig(bit_num=3),
            )
        )

        self.assertGreater(len(enc.generated_token_ids), 0)
        self.assertGreater(enc.consumed_bits, 0)
        self.assertEqual(dec.bits, secret_bits[: enc.consumed_bits])


if __name__ == "__main__":
    unittest.main()
