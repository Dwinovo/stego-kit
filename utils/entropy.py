from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def normalize_probability_table(prob_table: Sequence[float]) -> list[float]:
    """Return a normalized copy of a probability table."""
    values = [float(p) for p in prob_table if float(p) > 0.0]
    if not values:
        return []
    total = sum(values)
    if total <= 0.0:
        return []
    return [p / total for p in values]


def shannon_entropy(prob_table: Sequence[float], *, base: float = 2.0) -> float:
    """Compute Shannon entropy for a probability table."""
    probs = normalize_probability_table(prob_table)
    if not probs:
        return 0.0
    if base <= 0 or base == 1:
        raise ValueError("base must be > 0 and != 1")

    log_base = math.log(base)
    return -sum(p * (math.log(p) / log_base) for p in probs if p > 0.0)


def average_entropy(prob_tables: Iterable[Sequence[float]], *, base: float = 2.0) -> float:
    """Compute average Shannon entropy over multiple probability tables."""
    entropy_values = [shannon_entropy(table, base=base) for table in prob_tables]
    if not entropy_values:
        return 0.0
    return sum(entropy_values) / len(entropy_values)
