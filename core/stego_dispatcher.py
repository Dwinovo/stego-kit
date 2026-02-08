from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from core.algorithm_enum import StegoAlgorithm
from core.stego_algorithm import (
    StegoDecodeResult,
    StegoEncodeResult,
)
from core.stego_context import StegoDecodeContext, StegoEncodeContext
from core.stego_registry import StegoAlgorithmRegistry
from transformers import PreTrainedModel, PreTrainedTokenizerBase

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
        self._validate_encode_context(context)
        strategy = self._registry.get(context.algorithm)
        if self._verbose:
            logger.info(
                f"[Stego Encode] algorithm={context.algorithm}, max_new_tokens={context.max_new_tokens}, "
                f"secret_bits_len={len(context.secret_bits)}, precision={context.precision}"
            )
        return strategy.encode(context)

    def dispatch_decode(self, context: StegoDecodeContext) -> StegoDecodeResult:
        self._validate_decode_context(context)
        strategy = self._registry.get(context.algorithm)
        if self._verbose:
            logger.info(
                f"[Stego Decode] algorithm={context.algorithm}, generated_steps={len(context.generated_token_ids)}, "
                f"precision={context.precision}, max_bits={context.max_bits}"
            )
        return strategy.decode(context)

    @staticmethod
    def _validate_encode_context(context: StegoEncodeContext) -> None:
        if context.model is None:
            raise ValueError("model cannot be None")
        if context.tokenizer is None:
            raise ValueError("tokenizer cannot be None")
        if not hasattr(context.model, "__call__"):
            raise TypeError("model must be callable like transformers causal LM")
        if not hasattr(context.tokenizer, "__call__") or not hasattr(context.tokenizer, "decode"):
            raise TypeError("tokenizer must implement __call__ and decode")
        if not isinstance(context.secret_bits, str):
            raise TypeError("secret_bits must be a string of 0/1")
        if set(context.secret_bits) - {"0", "1"}:
            raise ValueError("secret_bits must contain only '0' and '1'")
        if not isinstance(context.messages, list):
            raise TypeError("messages must be a list of message dicts")
        if len(context.messages) == 0:
            raise ValueError("messages cannot be empty")
        for i, msg in enumerate(context.messages):
            if not isinstance(msg, dict):
                raise TypeError(f"messages[{i}] must be a dict")
            if "role" not in msg:
                raise ValueError(f"messages[{i}] missing required key: role")
            if "content" not in msg and "tool_calls" not in msg:
                raise ValueError(f"messages[{i}] must contain content or tool_calls")
        if context.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if context.precision <= 0:
            raise ValueError("precision must be positive")
        if context.temperature <= 0:
            raise ValueError("temperature must be > 0")
        if context.top_k is not None and context.top_k <= 0:
            raise ValueError("top_k must be positive")
        if context.top_p is not None and not (0 < context.top_p <= 1):
            raise ValueError("top_p must be in (0, 1]")

    def embed(
        self,
        *,
        algorithm: StegoAlgorithm | str,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        secret_bits: str,
        messages: Sequence[dict[str, Any]],
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        precision: int = 52,
        prg: Any | None = None,
        extra: dict[str, Any] | None = None,
    ) -> StegoEncodeResult:
        ctx = StegoEncodeContext(
            algorithm=algorithm,
            model=model,
            tokenizer=tokenizer,
            secret_bits=secret_bits,
            messages=list(messages),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            precision=precision,
            prg=prg,
            extra=extra or {},
        )
        return self.dispatch_encode(ctx)

    def extract(
        self,
        *,
        algorithm: StegoAlgorithm | str,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        generated_token_ids: Sequence[int],
        messages: Sequence[dict[str, Any]],
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        precision: int = 52,
        prg: Any | None = None,
        max_bits: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> StegoDecodeResult:
        ctx = StegoDecodeContext(
            algorithm=algorithm,
            model=model,
            tokenizer=tokenizer,
            generated_token_ids=generated_token_ids,
            messages=list(messages),
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            precision=precision,
            prg=prg,
            max_bits=max_bits,
            extra=extra or {},
        )
        return self.dispatch_decode(ctx)

    @staticmethod
    def _validate_decode_context(context: StegoDecodeContext) -> None:
        if context.model is None:
            raise ValueError("model cannot be None")
        if context.tokenizer is None:
            raise ValueError("tokenizer cannot be None")
        if not hasattr(context.model, "__call__"):
            raise TypeError("model must be callable like transformers causal LM")
        if not hasattr(context.tokenizer, "__call__") or not hasattr(context.tokenizer, "decode"):
            raise TypeError("tokenizer must implement __call__ and decode")
        if not isinstance(context.messages, list):
            raise TypeError("messages must be a list of message dicts")
        if len(context.messages) == 0:
            raise ValueError("messages cannot be empty")
        for i, msg in enumerate(context.messages):
            if not isinstance(msg, dict):
                raise TypeError(f"messages[{i}] must be a dict")
            if "role" not in msg:
                raise ValueError(f"messages[{i}] missing required key: role")
            if "content" not in msg and "tool_calls" not in msg:
                raise ValueError(f"messages[{i}] must contain content or tool_calls")
        if context.precision <= 0:
            raise ValueError("precision must be positive")
        if context.max_bits is not None and context.max_bits < 0:
            raise ValueError("max_bits must be >= 0")
        if context.temperature <= 0:
            raise ValueError("temperature must be > 0")
        if context.top_k is not None and context.top_k <= 0:
            raise ValueError("top_k must be positive")
        if context.top_p is not None and not (0 < context.top_p <= 1):
            raise ValueError("top_p must be in (0, 1]")
