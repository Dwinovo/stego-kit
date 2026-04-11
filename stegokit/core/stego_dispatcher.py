from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from stegokit.core.generation_config import GenerationConfig
from stegokit.core.runtime_context import RuntimeContext
from stegokit.core.security_material import validate_material_instance
from stegokit.core.stego_algorithm import StegoDecodeResult, StegoEncodeResult
from stegokit.core.stego_context import StegoDecodeContext, StegoEncodeContext
from stegokit.core.stego_registry import StegoAlgorithmRegistry
from stegokit.core.algorithm_enum import StegoAlgorithm

logger = logging.getLogger(__name__)


class StegoDispatcher:
    """Dispatcher for model-in-the-loop steganography."""

    def __init__(self, registry: StegoAlgorithmRegistry | None = None, verbose: bool = True) -> None:
        self._registry = registry or StegoAlgorithmRegistry.default()
        self._verbose = verbose

    @property
    def registry(self) -> StegoAlgorithmRegistry:
        return self._registry

    def dispatch_encode(self, context: StegoEncodeContext) -> StegoEncodeResult:
        spec = self._registry.get_spec(context.algorithm)
        self._validate_encode_context(context, spec)
        if self._verbose:
            logger.info(
                "[Stego Encode] algorithm=%s paradigm=%s max_new_tokens=%s secret_bits_len=%s precision=%s",
                spec.name,
                spec.paradigm.value,
                context.max_new_tokens,
                len(context.secret_bits),
                context.precision,
            )
        return spec.strategy.encode(context)

    def dispatch_decode(self, context: StegoDecodeContext) -> StegoDecodeResult:
        spec = self._registry.get_spec(context.algorithm)
        self._validate_decode_context(context, spec)
        if self._verbose:
            logger.info(
                "[Stego Decode] algorithm=%s paradigm=%s generated_steps=%s precision=%s max_bits=%s",
                spec.name,
                spec.paradigm.value,
                len(context.generated_token_ids),
                context.precision,
                context.max_bits,
            )
        return spec.strategy.decode(context)

    def embed(
        self,
        *,
        algorithm: StegoAlgorithm | str,
        secret_bits: str,
        runtime: RuntimeContext | None = None,
        model: PreTrainedModel | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        messages: Sequence[dict[str, Any]] | None = None,
        generation: GenerationConfig | None = None,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        precision: int = 52,
        stop_on_eos: bool | None = None,
        config: Any | None = None,
        material: Any | None = None,
    ) -> StegoEncodeResult:
        spec = self._registry.get_spec(algorithm)
        runtime_obj = self._build_runtime(
            runtime=runtime,
            model=model,
            tokenizer=tokenizer,
            messages=messages,
            generation=generation,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            precision=precision,
            stop_on_eos=stop_on_eos,
        )
        ctx = StegoEncodeContext(
            algorithm=algorithm,
            runtime=runtime_obj,
            secret_bits=secret_bits,
            config=config if config is not None else self._default_instance(spec.encode_config_type),
            material=material if material is not None else self._default_instance(spec.encode_material_type),
        )
        return self.dispatch_encode(ctx)

    def extract(
        self,
        *,
        algorithm: StegoAlgorithm | str,
        generated_token_ids: Sequence[int],
        runtime: RuntimeContext | None = None,
        model: PreTrainedModel | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        messages: Sequence[dict[str, Any]] | None = None,
        generation: GenerationConfig | None = None,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        precision: int = 52,
        max_bits: int | None = None,
        config: Any | None = None,
        material: Any | None = None,
    ) -> StegoDecodeResult:
        spec = self._registry.get_spec(algorithm)
        runtime_obj = self._build_runtime(
            runtime=runtime,
            model=model,
            tokenizer=tokenizer,
            messages=messages,
            generation=generation,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            precision=precision,
        )
        ctx = StegoDecodeContext(
            algorithm=algorithm,
            runtime=runtime_obj,
            generated_token_ids=generated_token_ids,
            max_bits=max_bits,
            config=config if config is not None else self._default_instance(spec.decode_config_type),
            material=material if material is not None else self._default_instance(spec.decode_material_type),
        )
        return self.dispatch_decode(ctx)

    @staticmethod
    def _default_instance(value_type: type[Any]) -> Any:
        try:
            return value_type()
        except TypeError:
            return None

    @staticmethod
    def _build_runtime(
        *,
        runtime: RuntimeContext | None,
        model: PreTrainedModel | None,
        tokenizer: PreTrainedTokenizerBase | None,
        messages: Sequence[dict[str, Any]] | None,
        generation: GenerationConfig | None,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        precision: int = 52,
        stop_on_eos: bool | None = None,
    ) -> RuntimeContext:
        if runtime is not None:
            return runtime
        return RuntimeContext(
            model=model,
            tokenizer=tokenizer,
            messages=list(messages or []),
            generation=generation
            or GenerationConfig(
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                precision=precision,
                stop_on_eos=stop_on_eos,
                max_new_tokens=max_new_tokens,
            ),
        )

    @staticmethod
    def _validate_runtime(runtime: RuntimeContext, *, require_max_new_tokens: bool) -> None:
        if not isinstance(runtime, RuntimeContext):
            raise TypeError("runtime must be a RuntimeContext")
        if runtime.model is None:
            raise ValueError("runtime.model cannot be None")
        if runtime.tokenizer is None:
            raise ValueError("runtime.tokenizer cannot be None")
        if not hasattr(runtime.model, "__call__"):
            raise TypeError("runtime.model must be callable like transformers causal LM")
        if not hasattr(runtime.tokenizer, "__call__") or not hasattr(runtime.tokenizer, "decode"):
            raise TypeError("runtime.tokenizer must implement __call__ and decode")
        if not isinstance(runtime.messages, list):
            raise TypeError("runtime.messages must be a list of message dicts")
        if len(runtime.messages) == 0:
            raise ValueError("runtime.messages cannot be empty")
        for i, msg in enumerate(runtime.messages):
            if not isinstance(msg, dict):
                raise TypeError(f"runtime.messages[{i}] must be a dict")
            if "role" not in msg:
                raise ValueError(f"runtime.messages[{i}] missing required key: role")
            if "content" not in msg and "tool_calls" not in msg:
                raise ValueError(f"runtime.messages[{i}] must contain content or tool_calls")

        generation = runtime.generation
        if not isinstance(generation, GenerationConfig):
            raise TypeError("runtime.generation must be a GenerationConfig")
        if generation.precision <= 0:
            raise ValueError("generation.precision must be positive")
        if generation.temperature <= 0:
            raise ValueError("generation.temperature must be > 0")
        if generation.top_k is not None and generation.top_k <= 0:
            raise ValueError("generation.top_k must be positive")
        if generation.top_p is not None and not (0 < generation.top_p <= 1):
            raise ValueError("generation.top_p must be in (0, 1]")
        if generation.stop_on_eos is not None and not isinstance(generation.stop_on_eos, bool):
            raise TypeError("generation.stop_on_eos must be bool or None")
        if require_max_new_tokens and generation.max_new_tokens <= 0:
            raise ValueError("generation.max_new_tokens must be positive")

    @staticmethod
    def _validate_typed_binding(value: Any, expected_type: type[Any], label: str) -> None:
        if not isinstance(value, expected_type):
            raise TypeError(f"{label} must be an instance of {expected_type.__name__}")

    def _validate_encode_context(self, context: StegoEncodeContext, spec) -> None:
        self._validate_runtime(context.runtime, require_max_new_tokens=True)
        if not isinstance(context.secret_bits, str):
            raise TypeError("secret_bits must be a string of 0/1")
        if set(context.secret_bits) - {"0", "1"}:
            raise ValueError("secret_bits must contain only '0' and '1'")

        self._validate_typed_binding(context.config, spec.encode_config_type, f"{spec.name} encode config")
        self._validate_typed_binding(context.material, spec.encode_material_type, f"{spec.name} encode material")
        validate_material_instance(context.material, spec.encode_material_type, f"{spec.name} encode")

        if spec.encode_config_validator is not None:
            spec.encode_config_validator(context.config)
        if spec.encode_material_validator is not None:
            spec.encode_material_validator(context.material)

    def _validate_decode_context(self, context: StegoDecodeContext, spec) -> None:
        self._validate_runtime(context.runtime, require_max_new_tokens=False)
        if context.max_bits is not None and context.max_bits < 0:
            raise ValueError("max_bits must be >= 0")

        self._validate_typed_binding(context.config, spec.decode_config_type, f"{spec.name} decode config")
        self._validate_typed_binding(context.material, spec.decode_material_type, f"{spec.name} decode material")
        validate_material_instance(context.material, spec.decode_material_type, f"{spec.name} decode")

        if spec.decode_config_validator is not None:
            spec.decode_config_validator(context.config)
        if spec.decode_material_validator is not None:
            spec.decode_material_validator(context.material)
