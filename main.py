from __future__ import annotations

import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from core import StegoAlgorithm, StegoDispatcher


MODEL_PATH = "/root/autodl-fs/Meta-Llama-3-8B-Instruct/"


def _pick_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    return torch.float16


def _load_model_and_tokenizer(model_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = _pick_dtype()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype)
    model.to(device)
    model.eval()
    return model, tokenizer


def run_asymmetric_demo(model_path: str = MODEL_PATH):
    dispatcher = StegoDispatcher(verbose=True)
    model, tokenizer = _load_model_and_tokenizer(model_path)

    prompt = "Please write some advice on privacy protection."
    bit_stream = "01011100110101101001011100100111010110100101110011010110"
    seed = "12345"
    secure_parameter = 32
    func_type = 1

    enc = dispatcher.embed(
        algorithm=StegoAlgorithm.ASYMMETRIC,
        model=model,
        tokenizer=tokenizer,
        secret_bits=bit_stream,
        prompt=prompt,
        max_new_tokens=512,
        temperature=1.2,
        top_k=50,
        precision=16,
        extra={
            "seed": seed,
            "secure_parameter": secure_parameter,
            "func_type": func_type,
            "use_chat_template": False,
        },
    )

    dec_regular = dispatcher.extract(
        algorithm=StegoAlgorithm.ASYMMETRIC,
        model=model,
        tokenizer=tokenizer,
        generated_token_ids=enc.generated_token_ids,
        prompt=prompt,
        temperature=1.2,
        top_k=50,
        precision=16,
        max_bits=enc.consumed_bits,
        extra={
            "decode_mode": "regular",
            "seed": seed,
            "secure_parameter": secure_parameter,
            "func_type": func_type,
            "use_chat_template": False,
        },
    )

    dec_robust = dispatcher.extract(
        algorithm=StegoAlgorithm.ASYMMETRIC,
        model=model,
        tokenizer=tokenizer,
        generated_token_ids=enc.generated_token_ids,
        prompt=prompt,
        temperature=1.2,
        top_k=50,
        precision=16,
        max_bits=enc.consumed_bits,
        extra={
            "decode_mode": "robust",
            "seed": seed,
            "secure_parameter": secure_parameter,
            "func_type": func_type,
            "robust_search_window": 1000,
            "use_chat_template": False,
        },
    )

    print("\n=== Stego Demo: asymmetric ===")
    print("model_path:", model_path)
    print("generated_text:", enc.text)
    print("consumed_bits:", enc.consumed_bits)
    print("regular_bits:", dec_regular.bits)
    print("regular_match:", dec_regular.bits == bit_stream[:enc.consumed_bits])
    print("robust_bits:", dec_robust.bits)
    print("robust_match:", dec_robust.bits == bit_stream[:enc.consumed_bits])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    run_asymmetric_demo(MODEL_PATH)
