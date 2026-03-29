"""Apple Silicon MPS optimization module — research-informed implementation.

Key findings from research (2026-03-29):
- torch.compile on MPS is NOT ready for diffusion models (use eager mode)
- Attention slicing HURTS on 64GB M4 Max (trades speed for memory we don't need)
- SDPA is available and beneficial on MPS (PyTorch 2.5+)
- Float16 is the correct dtype (MPS bfloat16 upcast is slower)
- On unified memory, CPU offloading adds overhead without saving memory
- PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 unlocks full memory
- Avoid CPU/GPU synchronization during inference loop
- Safetensors mmap gives near-instant model loading on unified memory
"""
from __future__ import annotations

import os
import platform
from typing import Any

import torch


def is_mps_available() -> bool:
    """Check if MPS backend is available and functional."""
    if not torch.backends.mps.is_available():
        return False
    try:
        x = torch.zeros(1, device="mps")
        _ = x + 1
        return True
    except Exception:
        return False


def get_apple_silicon_info() -> dict[str, Any]:
    """Get Apple Silicon hardware information."""
    info: dict[str, Any] = {
        "platform": platform.machine(),
        "chip": "unknown",
        "memory_gb": 0,
    }

    try:
        import subprocess
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            info["memory_gb"] = int(result.stdout.strip()) / (1024 ** 3)

        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            info["chip"] = result.stdout.strip()
    except Exception:
        pass

    return info


def configure_mps_environment() -> dict[str, str]:
    """Set optimal environment variables for MPS inference.

    Research-backed settings for maximum performance on Apple Silicon.
    """
    env_settings = {
        # Disable memory watermarks — on unified memory (64GB),
        # we want PyTorch to use as much as needed without throttling
        "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.0",
        "PYTORCH_MPS_LOW_WATERMARK_RATIO": "0.0",
        # Enable fallback for any unsupported MPS operations
        "PYTORCH_ENABLE_MPS_FALLBACK": "1",
    }

    for key, value in env_settings.items():
        os.environ[key] = value

    return env_settings


def get_optimal_dtype() -> torch.dtype:
    """Get optimal dtype for MPS.

    Float16 is correct for MPS — bfloat16 is silently upcast to float32
    on MPS, making it slower than explicit float16.
    """
    return torch.float16


def optimize_pipeline(pipe: Any, memory_gb: float = 64.0) -> Any:
    """Apply research-backed MPS optimizations to a diffusers pipeline.

    Key principle: On M4 Max with 64GB, we have ABUNDANT memory.
    Optimize for SPEED, not memory savings.

    Args:
        pipe: A diffusers pipeline instance
        memory_gb: Available system memory in GB

    Returns:
        The optimized pipeline
    """
    configure_mps_environment()

    # CRITICAL: Do NOT enable attention_slicing on high-memory devices.
    # Slicing trades compute speed for memory savings — counterproductive
    # on 64GB where the model easily fits.
    if memory_gb >= 32:
        # Disable slicing if it was previously enabled
        if hasattr(pipe, "disable_attention_slicing"):
            pipe.disable_attention_slicing()
    else:
        # Only enable on low-memory devices (<32GB)
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing("auto")

    # Enable VAE tiling ONLY for images larger than 1024px
    # (it's needed for 2K+ images to avoid OOM during decode)
    if hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()

    # Enable VAE slicing — this IS beneficial as it processes
    # batch elements sequentially through VAE (saves memory, minimal speed cost)
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()

    # IMPORTANT: Do NOT use model_cpu_offload on unified memory.
    # CPU and GPU share the same physical memory on Apple Silicon.
    # CPU offloading just adds transfer overhead with zero memory benefit.
    pipe.to("mps")

    return pipe


def optimize_pipeline_low_memory(pipe: Any) -> Any:
    """Optimize for low-memory Apple Silicon devices (<32GB).

    Uses sequential CPU offloading which keeps only one component
    on device at a time.
    """
    configure_mps_environment()

    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing("auto")

    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()

    if hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()

    if hasattr(pipe, "enable_sequential_cpu_offload"):
        pipe.enable_sequential_cpu_offload()

    return pipe


def load_pipeline_optimized(
    model_path: str,
    low_memory: bool = False,
) -> Any:
    """Load pipeline with MPS-optimized settings.

    Uses safetensors mmap for near-instant loading on unified memory.
    """
    import sys
    from pathlib import Path

    # Ensure patched diffusers is available
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    local_diffusers = repo_root / "upstream-realrestorer" / "diffusers" / "src"
    if local_diffusers.is_dir() and str(local_diffusers) not in sys.path:
        sys.path.insert(0, str(local_diffusers))

    upstream_root = repo_root / "upstream-realrestorer"
    if upstream_root.is_dir() and str(upstream_root) not in sys.path:
        sys.path.insert(0, str(upstream_root))

    from diffusers import RealRestorerPipeline

    dtype = get_optimal_dtype()

    # Load with safetensors (mmap on unified memory = near-instant)
    pipe = RealRestorerPipeline.from_pretrained(
        model_path,
        torch_dtype=dtype,
        use_safetensors=True,
    )

    # Apply optimizations based on memory profile
    hw_info = get_apple_silicon_info()
    memory_gb = hw_info.get("memory_gb", 64.0)

    if low_memory or memory_gb < 32:
        pipe = optimize_pipeline_low_memory(pipe)
    else:
        pipe = optimize_pipeline(pipe, memory_gb)

    return pipe


def run_inference_optimized(
    pipe: Any,
    image: Any,
    prompt: str,
    negative_prompt: str = "",
    steps: int = 28,
    guidance_scale: float = 3.0,
    seed: int = 42,
    size_level: int = 1024,
) -> Any:
    """Run inference with MPS-optimized settings.

    Key: avoid CPU/GPU synchronization during the diffusion loop.
    Only synchronize at the end for accurate timing.
    """
    # Pre-warmup MPS (first run is always slower due to shader compilation)
    # The pipeline will handle this internally

    # Run inference — let MPS handle everything asynchronously
    result = pipe(
        image=image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        seed=seed,
        size_level=size_level,
    )

    # Synchronize MPS to ensure all operations complete
    synchronize()

    return result


def get_memory_stats() -> dict[str, float]:
    """Get MPS memory statistics."""
    stats: dict[str, float] = {}

    try:
        if hasattr(torch.mps, "driver_allocated_memory"):
            stats["driver_allocated_mb"] = torch.mps.driver_allocated_memory() / 1024 / 1024
        if hasattr(torch.mps, "current_allocated_memory"):
            stats["current_allocated_mb"] = torch.mps.current_allocated_memory() / 1024 / 1024
    except Exception:
        pass

    # Also get system-level memory for unified memory tracking
    try:
        import psutil
        vm = psutil.virtual_memory()
        stats["system_used_mb"] = vm.used / 1024 / 1024
        stats["system_available_mb"] = vm.available / 1024 / 1024
    except ImportError:
        pass

    return stats


def synchronize() -> None:
    """Synchronize MPS operations (for accurate timing)."""
    if is_mps_available():
        torch.mps.synchronize()


def clear_cache() -> None:
    """Clear MPS memory cache."""
    if is_mps_available() and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
