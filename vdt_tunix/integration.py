"""Fail-closed extension point for the optional real Tunix backend."""

from __future__ import annotations

import importlib
import importlib.util

from vdt_tunix.config import RunConfig
from vdt_tunix.contracts import BackendBundle


REAL_BACKEND_MODULE = "vdt_tunix.real_backend"


class RealModelIntegrationUnavailable(RuntimeError):
    """Raised when the requested real model integration cannot be loaded."""


def load_real_backend_bundle(config: RunConfig) -> BackendBundle:
    """Load an opt-in real adapter without ever falling back to CPU mocks."""

    try:
        specification = importlib.util.find_spec(REAL_BACKEND_MODULE)
    except (ImportError, ValueError) as exc:
        raise RealModelIntegrationUnavailable(
            f"could not resolve {REAL_BACKEND_MODULE}: {exc}"
        ) from exc
    if specification is None:
        raise RealModelIntegrationUnavailable(
            "real Tunix/MaxText model integration is not implemented; expected "
            f"{REAL_BACKEND_MODULE}.build_backends(config)"
        )
    try:
        module = importlib.import_module(REAL_BACKEND_MODULE)
    except Exception as exc:
        raise RealModelIntegrationUnavailable(
            f"failed to import real backend: {type(exc).__name__}: {exc}"
        ) from exc
    builder = getattr(module, "build_backends", None)
    if not callable(builder):
        raise RealModelIntegrationUnavailable(
            f"{REAL_BACKEND_MODULE} has no callable build_backends"
        )
    try:
        bundle = builder(config)
    except Exception as exc:
        raise RealModelIntegrationUnavailable(
            "real backend unavailable: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(bundle, BackendBundle):
        raise RealModelIntegrationUnavailable(
            "real backend builder did not return BackendBundle"
        )
    if not (
        bundle.student.real_model_integration
        and bundle.teacher.real_model_integration
    ):
        raise RealModelIntegrationUnavailable(
            "real backend bundle declared a mock or incomplete integration"
        )
    return bundle
