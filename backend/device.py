"""
Compute device selection for the headless pipeline (GPU with CPU fallback).

The GPU path needs CuPy (features) and RAPIDS cuML (random forest) plus a
visible CUDA device.  ``configure("auto")`` uses the GPU when all of that is
present and silently falls back to CPU otherwise; ``configure("gpu")`` raises
``GpuUnavailable`` instead, so batch scripts can detect it and rerun on CPU.

When ``configure`` was never called (e.g. the interactive server), the legacy
behaviour is kept: the classifier uses cuML if it imports, features stay on CPU.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Exit code used by the CLI when the GPU was requested but cannot be used.
# Job scripts treat it as "retry this job on CPU".
GPU_UNAVAILABLE_EXIT = 3


class GpuUnavailable(RuntimeError):
    pass


_active: Optional[str] = None  # None = not configured, else "gpu" / "cpu"


def gpu_status() -> Tuple[bool, str]:
    """Return (usable, reason). Never raises."""
    try:
        import cupy  # type: ignore
    except Exception as e:  # ImportError, or CUDA libs missing
        return False, f"cupy not importable: {e}"
    try:
        n = cupy.cuda.runtime.getDeviceCount()
    except Exception as e:
        return False, f"no CUDA device: {e}"
    if n < 1:
        return False, "no CUDA device visible"
    try:
        from cuml.ensemble import RandomForestClassifier  # type: ignore  # noqa: F401
    except Exception as e:
        return False, f"cuml not importable: {e}"
    try:
        # Touch the device so driver/runtime mismatches surface now, not mid-job.
        cupy.arange(4, dtype=cupy.float32).sum().item()
        props = cupy.cuda.runtime.getDeviceProperties(0)
        name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
    except Exception as e:
        return False, f"CUDA device not usable: {e}"
    return True, f"{n} device(s), using {name}"


def configure(mode: str) -> str:
    """Select the device for this process. Returns "gpu" or "cpu"."""
    global _active
    if mode not in ("auto", "gpu", "cpu"):
        raise ValueError(f"device must be auto, gpu or cpu, not {mode!r}")
    if mode == "cpu":
        _active = "cpu"
        logger.info("Device: CPU (requested)")
        return _active
    ok, reason = gpu_status()
    if ok:
        _active = "gpu"
        logger.info("Device: GPU (%s)", reason)
    elif mode == "gpu":
        raise GpuUnavailable(reason)
    else:
        _active = "cpu"
        logger.info("Device: CPU (GPU unavailable: %s)", reason)
    return _active


def active() -> Optional[str]:
    return _active


def use_gpu_features() -> bool:
    return _active == "gpu"


def use_gpu_classifier(cuml_importable: bool) -> bool:
    if _active is None:
        return cuml_importable  # legacy auto-detect for the server
    return _active == "gpu"
