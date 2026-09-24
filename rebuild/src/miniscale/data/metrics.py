"""Small, deterministic statistics shared by dataset audits."""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def integer_percentiles(
    values: Iterable[int],
    percentiles: Sequence[int] = (50, 75, 90, 95, 99),
) -> dict[str, int]:
    """Return nearest-index percentiles without changing historical rounding.

    NumPy and ``statistics.quantiles`` use interpolation conventions that can
    produce different answers for small audit samples. This project needs
    stable integer results for reproducible reports, so the established
    nearest-index rule remains explicit here.
    """

    ordered = sorted(values)
    if not ordered:
        return {}
    result = {
        f"p{percentile}": ordered[round((len(ordered) - 1) * percentile / 100)]
        for percentile in percentiles
    }
    result["max"] = ordered[-1]
    return result
