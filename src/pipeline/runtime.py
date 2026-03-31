"""Runtime helpers for selecting CPU vs CUDA execution."""

from __future__ import annotations

import os


def resolve_device() -> str:
    """Resolve the execution device from env, defaulting to CUDA when available."""
    requested = os.environ.get("RTM_DEVICE", "auto").strip().lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("RTM_DEVICE must be one of: auto, cpu, cuda")

    if requested == "cpu":
        return "cpu"

    try:
        import torch
    except ImportError:
        return "cpu"

    cuda_available = bool(torch.cuda.is_available())

    if requested == "cuda":
        if not cuda_available:
            raise RuntimeError(
                "RTM_DEVICE=cuda was requested, but torch.cuda.is_available() is false."
            )
        return "cuda"

    return "cuda" if cuda_available else "cpu"


def resolve_compute_type(device: str) -> str:
    """Return a sensible faster-whisper compute type for the selected device."""
    default = "float16" if device == "cuda" else "int8"
    return os.environ.get("RTM_COMPUTE_TYPE", default).strip().lower()
