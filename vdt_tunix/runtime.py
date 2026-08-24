"""Lazy TPU runtime gate that can be unit-tested without importing JAX."""

from __future__ import annotations

import platform
from typing import Any


class TPUPreflightError(RuntimeError):
    """Raised unless the active JAX runtime is exactly a TPU v5e-8 slice."""


def require_tpu_v5e8(
    *, expected_device_count: int = 8, jax_module: Any | None = None
) -> tuple[list[Any], dict[str, Any]]:
    if jax_module is None:
        try:
            import jax as jax_module
        except ImportError as exc:
            raise TPUPreflightError("JAX is unavailable for TPU preflight") from exc

    devices = list(jax_module.devices())
    kinds = [str(getattr(device, "device_kind", "")) for device in devices]
    kind_match = bool(kinds) and all(
        "v5e" in kind.lower() or "v5 lite" in kind.lower() for kind in kinds
    )
    evidence = {
        "python": platform.python_version(),
        "jax": str(getattr(jax_module, "__version__", "unknown")),
        "backend": str(jax_module.default_backend()),
        "device_count": len(devices),
        "devices": [str(device) for device in devices],
        "device_kinds": kinds,
        "v5e_kind_match": kind_match,
    }
    if (
        evidence["backend"] != "tpu"
        or len(devices) != expected_device_count
        or not kind_match
    ):
        raise TPUPreflightError(
            "expected exactly a TPU v5e-8 runtime; "
            f"observed backend={evidence['backend']!r}, "
            f"device_count={len(devices)}, device_kinds={kinds!r}"
        )
    return devices, evidence
