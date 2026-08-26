"""Small, deterministic performance-instrumentation helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any


class PerformanceContractError(ValueError):
    """Raised when a performance canary cannot honor its static-shape contract."""


def select_length_bucket(required: int, buckets: Sequence[int]) -> int:
    """Return the smallest configured bucket that contains ``required``."""

    if required < 1:
        raise PerformanceContractError("required sequence length must be positive")
    if not buckets:
        return required
    for bucket in buckets:
        if required <= bucket:
            return int(bucket)
    raise PerformanceContractError(
        f"required sequence length {required} exceeds largest bucket {buckets[-1]}"
    )


def numeric_shape_signature(**values: Any) -> int:
    """Return a stable, W&B-safe integer identifier for a shape signature."""

    payload = json.dumps(values, sort_keys=True, separators=(",", ":"))
    # 52 bits stay exactly representable in the float metrics path.
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:7], "big") >> 4


def jit_cache_size(compiled: Any) -> int:
    """Best-effort JAX cache size without making private APIs a hard dependency."""

    candidates = (compiled, getattr(compiled, "inner", None))
    for candidate in candidates:
        if candidate is None:
            continue
        getter = getattr(candidate, "_cache_size", None)
        if callable(getter):
            try:
                return int(getter())
            except (TypeError, ValueError, RuntimeError):
                pass
    return -1


def jax_memory_metrics(jax_module: Any) -> dict[str, int]:
    """Return portable numeric memory evidence when the backend exposes it."""

    result = {
        "memory_bytes_in_use": -1,
        "memory_peak_bytes_in_use": -1,
        "memory_bytes_limit": -1,
    }
    try:
        devices = tuple(jax_module.devices())
    except (AttributeError, RuntimeError):
        return result
    for device in devices:
        getter = getattr(device, "memory_stats", None)
        if not callable(getter):
            continue
        try:
            stats = getter() or {}
        except (RuntimeError, TypeError):
            continue
        for output_name, candidates in {
            "memory_bytes_in_use": ("bytes_in_use",),
            "memory_peak_bytes_in_use": (
                "peak_bytes_in_use",
                "peak_bytes_in_use_per_device",
            ),
            "memory_bytes_limit": ("bytes_limit",),
        }.items():
            values = [int(stats[name]) for name in candidates if name in stats]
            if values:
                result[output_name] = max(result[output_name], max(values))
    return result
