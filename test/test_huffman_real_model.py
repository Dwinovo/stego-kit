from __future__ import annotations

import argparse
import math
import os
import random
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Allow running as: python test/test_huffman_real_model.py
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from stegokit.core.algorithm_enum import StegoAlgorithm
from stegokit.core.stego_dispatcher import StegoDispatcher


def generate_secret_bits(length: int, seed: int) -> str:
    rng = random.Random(seed)
    return "".join(rng.choice("01") for _ in range(length))


def calc_bit_accuracy(gt: str, pred: str) -> tuple[float, int]:
    usable = min(len(gt), len(pred))
    if usable == 0:
        return 0.0, 0
    correct = sum(1 for i in range(usable) if gt[i] == pred[i])
    return correct / usable, usable


def build_messages() -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "你是信息安全研究助理。回答要求准确、紧凑、结构化，"
                "尽量使用中文，并避免无关寒暄。"
            ),
        },
        {
            "role": "user",
            "content": (
                "请简要解释企业数据保护为什么重要，并给出 5 条落地建议，"
                "覆盖访问控制、数据分级、传输保护、审计告警、员工培训。"
            ),
        },
    ]


def _apply_fallback_chat_template(tokenizer) -> None:
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{ message['role'] + ': ' + message['content'] + '\\n' }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}assistant: {% endif %}"
    )


def load_model(model_path: str, allow_template_fallback: bool):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if not getattr(tokenizer, "chat_template", None):
        if allow_template_fallback and hasattr(tokenizer, "apply_chat_template"):
            _apply_fallback_chat_template(tokenizer)
            print("Tokenizer has no built-in chat_template; applied a simple fallback template for testing.")
        else:
            raise ValueError(f"Tokenizer for {model_path} has no chat_template configured")

    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

    model.eval()
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def run_case(
    *,
    model,
    tokenizer,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    precision: int,
    bit_num: int | None,
    candidate_count: int | None,
    secret_len: int,
    stop_on_eos: bool,
    secret_seed: int,
) -> dict:
    dispatcher = StegoDispatcher(verbose=False)
    messages = build_messages()

    if secret_len <= 0:
        effective_candidates = candidate_count if candidate_count is not None else 2 ** int(bit_num or 3)
        approx_bits_per_token = max(1, int(math.log2(max(2, effective_candidates))))
        secret_len = approx_bits_per_token * max_new_tokens + 32
    secret_bits = generate_secret_bits(secret_len, secret_seed)

    extra: dict[str, int] = {}
    if candidate_count is not None:
        extra["candidate_count"] = int(candidate_count)
    elif bit_num is not None:
        extra["bit_num"] = int(bit_num)

    enc = dispatcher.embed(
        algorithm=StegoAlgorithm.HUFFMAN,
        model=model,
        tokenizer=tokenizer,
        secret_bits=secret_bits,
        messages=messages,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        precision=precision,
        stop_on_eos=stop_on_eos,
        extra=extra,
    )

    dec = dispatcher.extract(
        algorithm=StegoAlgorithm.HUFFMAN,
        model=model,
        tokenizer=tokenizer,
        generated_token_ids=enc.generated_token_ids,
        messages=messages,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        precision=precision,
        max_bits=enc.consumed_bits,
        extra=extra,
    )

    used_secret = secret_bits[: enc.consumed_bits]
    decoded = dec.bits[: enc.consumed_bits]
    strict_match = len(used_secret) > 0 and used_secret == decoded
    bit_acc, compared_bits = calc_bit_accuracy(used_secret, decoded)
    expected_capacity = (enc.consumed_bits / len(enc.generated_token_ids)) if enc.generated_token_ids else 0.0

    return {
        "model_path": getattr(model, "name_or_path", None) or getattr(tokenizer, "name_or_path", "unknown"),
        "secret_len": len(secret_bits),
        "consumed_bits": enc.consumed_bits,
        "decoded_bits": len(dec.bits),
        "compared_bits": compared_bits,
        "bit_accuracy": bit_acc,
        "strict_match": strict_match,
        "tokens": len(enc.generated_token_ids),
        "encode_time_seconds": enc.encode_time_seconds,
        "decode_time_seconds": dec.decode_time_seconds,
        "embedding_capacity": enc.embedding_capacity,
        "capacity_consistent": abs(enc.embedding_capacity - expected_capacity) <= 1e-9,
        "generated_text": enc.text,
        "used_secret": used_secret,
        "decoded_secret": decoded,
        "extra": extra,
    }


def print_result(result: dict) -> None:
    print("\n===== HUFFMAN Real-Model Result =====")
    print(f"model: {result['model_path']}")
    print(f"extra: {result['extra']}")
    print(
        f"tokens={result['tokens']} consumed_bits={result['consumed_bits']} "
        f"decoded_bits={result['decoded_bits']} compared_bits={result['compared_bits']}"
    )
    print(
        f"bit_accuracy={result['bit_accuracy']:.4f} strict_match={result['strict_match']} "
        f"capacity={result['embedding_capacity']:.6f} capacity_consistent={result['capacity_consistent']}"
    )
    print(
        f"encode_t={result['encode_time_seconds']:.4f}s "
        f"decode_t={result['decode_time_seconds']:.4f}s"
    )
    print(f"text:\n{result['generated_text']}\n")
    print(f"secret head:  {result['used_secret'][:128]}")
    print(f"decoded head: {result['decoded_secret'][:128]}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run Huffman stego on a real chat model")
    parser.add_argument(
        "--model-path",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Local path or Hugging Face model id with chat_template",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--precision", type=int, default=16)
    parser.add_argument("--bit-num", type=int, default=3, help="Candidate count will be 2 ** bit_num")
    parser.add_argument("--candidate-count", type=int, default=None, help="Overrides bit-num when provided")
    parser.add_argument("--secret-len", type=int, default=0, help="<= 0 means auto-size")
    parser.add_argument("--secret-seed", type=int, default=2026)
    parser.add_argument("--stop-on-eos", action="store_true", help="Stop generation once eos token is sampled")
    parser.add_argument(
        "--allow-template-fallback",
        action="store_true",
        help="Inject a simple chat_template when the tokenizer has none, useful for models like gpt2",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Loading model from: {args.model_path}")
    model, tokenizer = load_model(args.model_path, allow_template_fallback=args.allow_template_fallback)
    result = run_case(
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        precision=args.precision,
        bit_num=args.bit_num,
        candidate_count=args.candidate_count,
        secret_len=args.secret_len,
        stop_on_eos=args.stop_on_eos,
        secret_seed=args.secret_seed,
    )
    print_result(result)


if __name__ == "__main__":
    main()
