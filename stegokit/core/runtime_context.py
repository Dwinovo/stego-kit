from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .generation_config import GenerationConfig


@dataclass(slots=True)
class RuntimeContext:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    messages: list[dict[str, Any]] = field(default_factory=list)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
