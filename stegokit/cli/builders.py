from __future__ import annotations

from dataclasses import is_dataclass
from typing import Any, Mapping

from transformers import AutoModelForCausalLM, AutoTokenizer

from stegokit.core.generation_config import GenerationConfig
from stegokit.core.runtime_context import RuntimeContext
from stegokit.core.security_material import BitMaskMaterial, NoMaterial, RandomnessMaterial
from stegokit.utils.prg import PRG


def load_model_and_tokenizer(model_name_or_path: str, *, trust_remote_code: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("tokenizer.chat_template must be configured for StegoKit")
    return model, tokenizer


def build_generation_config(
    *,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    precision: int,
    stop_on_eos: bool | None,
    max_new_tokens: int,
) -> GenerationConfig:
    return GenerationConfig(
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        precision=precision,
        stop_on_eos=stop_on_eos,
        max_new_tokens=max_new_tokens,
    )


def build_runtime_context(
    *,
    model_name_or_path: str,
    messages: list[dict[str, Any]],
    generation: GenerationConfig,
    trust_remote_code: bool = False,
) -> RuntimeContext:
    model, tokenizer = load_model_and_tokenizer(model_name_or_path, trust_remote_code=trust_remote_code)
    return RuntimeContext(
        model=model,
        tokenizer=tokenizer,
        messages=messages,
        generation=generation,
    )


def build_config(config_type: type[Any], payload: Mapping[str, Any]) -> Any:
    if config_type.__name__ == "NoConfig":
        if payload:
            raise ValueError("NoConfig does not accept any fields")
        return config_type()

    if not is_dataclass(config_type):
        if payload:
            raise TypeError(f"unsupported config type for CLI: {config_type.__name__}")
        return config_type()

    return config_type(**dict(payload))


def _extract_seed(payload: Mapping[str, Any]) -> int:
    if "prg_seed" in payload:
        return int(payload["prg_seed"])
    if "seed" in payload:
        return int(payload["seed"])
    raise ValueError("material JSON must provide 'prg_seed'")


def build_material(material_type: type[Any], payload: Mapping[str, Any]) -> Any:
    if material_type is NoMaterial:
        if payload:
            raise ValueError("NoMaterial does not accept any fields")
        return NoMaterial()

    if material_type is RandomnessMaterial:
        seed = _extract_seed(payload)
        return RandomnessMaterial(prg=PRG.from_int_seed(seed))

    if material_type is BitMaskMaterial:
        seed = _extract_seed(payload)
        return BitMaskMaterial(prg=PRG.from_int_seed(seed))

    if not is_dataclass(material_type):
        if payload:
            raise TypeError(f"unsupported material type for CLI: {material_type.__name__}")
        return material_type()

    return material_type(**dict(payload))
