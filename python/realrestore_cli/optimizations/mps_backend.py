"""Apple Silicon MPS (Metal Performance Shaders) optimization module.

Provides MPS-specific optimizations for the RealRestorer pipeline:
- Proper dtype handling (float16 on MPS)
- Attention slicing for memory efficiency
- VAE tiling for large images
- torch.compile for MPS backend
- Unified memory-aware offloading
"""
from __future__ import annotations

import os
from typing import Any

import torch


def is_mps_available() -> bool:
    """Check if MPS backend is available and functional."""
    if not torch.backends.mps.is_available():
        return False
    try:
        # Quick smoke test
        x = torch.zeros(1, device="mps")
        _ = x + 1
        return True
    except Exception:
        return False


def configure_mps_environment() -> dict[str, str]:
    """Set optimal environment variables for MPS inference.

    Returns dict of env vars that were set.
    """
    env_settings = {
        # Use high performance GPU mode
        "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.0",
        # Disable memory limit for unified memory (we have 64GB)
        "PYTORCH_MPS_LOW_WATERMARK_RATIO": "0.0",
        # Enable MPS fallback for unsupported ops
        "PYTORCH_ENABLE_MPS_FALLBACK": "1",
    }

    for key, value in env_settings.items():
        os.environ[key] = value

    return env_settings


def get_optimal_dtype() -> torch.dtype:
    """Get optimal dtype for MPS.

    MPS supports float16 well but bfloat16 support is limited.
    Float16 gives ~2x speedup over float32.
    """
    return torch.float16


def optimize_pipeline(pipe: Any) -> Any:
    """Apply MPS-specific optimizations to a diffusers pipeline.

    Args:
        pipe: A diffusers pipeline instance

    Returns:
        The optimized pipeline
    """
    # Configure environment
    configure_mps_environment()

    # Enable attention slicing — reduces peak memory by processing
    # attention in chunks rather than all at once
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing("auto")

    # Enable VAE slicing — processes VAE in slices
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()

    # Enable VAE tiling for large images (>1024px)
    if hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()

    # Try to enable SDPA (Scaled Dot Product Attention) if available
    _enable_sdpa(pipe)

    # Move to MPS device
    pipe.to("mps")

    return pipe


def _enable_sdpa(pipe: Any) -> None:
    """Enable Scaled Dot Product Attention if supported."""
    try:
        # PyTorch 2.0+ has native SDPA support
        if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            # Check if the pipeline supports SDPA
            if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
                try:
                    pipe.enable_xformers_memory_efficient_attention()
                except Exception:
                    pass  # xformers not available, SDPA will be used as fallback
    except Exception:
        pass


def apply_torch_compile(pipe: Any) -> Any:
    """Apply torch.compile to pipeline components for MPS acceleration.

    Note: torch.compile MPS support is experimental in PyTorch 2.5+.
    Falls back gracefully if compilation fails.
    """
    try:
        # Only compile the denoiser (most compute-intensive part)
        if hasattr(pipe, "unet"):
            pipe.unet = torch.compile(
                pipe.unet,
                backend="aot_eager",  # MPS-compatible backend
                mode="reduce-overhead",
            )
        elif hasattr(pipe, "transformer"):
            pipe.transformer = torch.compile(
                pipe.transformer,
                backend="aot_eager",
                mode="reduce-overhead",
            )
    except Exception:
        pass  # Graceful fallback if compilation fails

    return pipe


def optimize_for_unified_memory(pipe: Any, total_memory_gb: float = 64.0) -> Any:
    """Optimize pipeline for Apple Silicon unified memory.

    On unified memory, CPU and GPU share the same physical memory,
    so traditional CPU offloading doesn't free GPU memory. Instead,
    we optimize for keeping everything in memory simultaneously.
    """
    # On unified memory, model_cpu_offload is counterproductive
    # because it adds transfer overhead without saving memory.
    # Instead, just keep everything on MPS.

    # For very large models that exceed memory, use sequential offloading
    # which only keeps one component on device at a time
    if total_memory_gb < 48:
        # Only use sequential offloading if memory is constrained
        if hasattr(pipe, "enable_sequential_cpu_offload"):
            pipe.enable_sequential_cpu_offload()
    else:
        # With 64GB, keep everything on device
        pipe.to("mps")

    return pipe


def get_memory_stats() -> dict[str, float]:
    """Get MPS memory statistics."""
    stats = {}

    if hasattr(torch.mps, "driver_allocated_memory"):
        stats["driver_allocated_mb"] = torch.mps.driver_allocated_memory() / 1024 / 1024

    if hasattr(torch.mps, "current_allocated_memory"):
        stats["current_allocated_mb"] = torch.mps.current_allocated_memory() / 1024 / 1024

    return stats


def synchronize() -> None:
    """Synchronize MPS operations (useful for accurate timing)."""
    if is_mps_available():
        torch.mps.synchronize()


def clear_cache() -> None:
    """Clear MPS memory cache."""
    if is_mps_available() and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
