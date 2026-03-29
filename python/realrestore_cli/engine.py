"""Core inference engine for RealRestore CLI.

Orchestrates model loading, device selection, and inference with
research-backed optimization layers (MPS, quantization, MLX, ANE).

Design decisions informed by research (2026-03-29):
- MPS eager mode (torch.compile not ready for diffusion on MPS)
- Float16 dtype (bfloat16 silently upcast on MPS = slower)
- No attention slicing on 64GB (trades speed for unneeded memory savings)
- No CPU offloading on unified memory (zero benefit, adds overhead)
- Safetensors mmap for near-instant model loading
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch
from PIL import Image

# Module-level pipeline cache — avoids reloading 39GB model on every call
# Key: (model_path, device, dtype, quantize)
_pipeline_cache: dict[tuple, Any] = {}
_cache_key: tuple | None = None

# Task prompt mapping
TASK_PROMPTS: dict[str, str] = {
    "auto": "Restore the details and keep the original composition.",
    "deblur": "Please deblur the image and make it sharper",
    "denoise": "Please remove noise from the image.",
    "dehaze": "Please dehaze the image",
    "derain": "Please remove the rain from the image and restore its clarity.",
    "low_light": "Please restore this low-quality image, recovering its normal brightness and clarity.",
    "compression": "Please restore the image clarity and artifacts.",
    "moire": "Please remove the moiré patterns from the image",
    "lens_flare": "Please remove the lens flare and glare from the image.",
    "reflection": "Please remove the reflection from the image.",
}

DEFAULT_NEGATIVE_PROMPT = (
    "worst quality, wrong limbs, unreasonable limbs, normal quality, "
    "low quality, low res, blurry, text, watermark, logo, banner, "
    "extra digits, cropped, jpeg artifacts, signature, username, "
    "error, sketch, duplicate, ugly, monochrome, horror, geometry, "
    "mutation, disgusting"
)


def get_device() -> str:
    """Select best available device."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_dtype(device: str) -> torch.dtype:
    """Select appropriate dtype for device.

    Proven pattern from FLUX.1 community (2026-03-29):
    - bfloat16 WORKS on MPS with PyTorch 2.10+ (tested and confirmed)
    - bfloat16 has range up to 3.4e38 (vs float16 max 65504) — no NaN overflow
    - This is the SAME dtype the model was trained in — zero precision loss
    - Float16 causes NaN in attention (max 65504 overflows)
    - Float32 full model = 78.6GB — OOM on 64GB
    - RoPE computed on CPU with float64 precision (FLUX proven pattern)
    """
    if device == "mps":
        return torch.bfloat16
    if device == "cuda":
        return torch.bfloat16
    return torch.float32


def get_model_path() -> str:
    """Resolve model path from env or default."""
    env_path = os.environ.get("REALRESTORE_MODEL_PATH")
    if env_path:
        return env_path
    return "RealRestorer/RealRestorer"


def get_system_memory_gb() -> float:
    """Get system memory in GB."""
    try:
        import subprocess
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            return int(result.stdout.strip()) / (1024 ** 3)
    except Exception:
        pass
    return 64.0  # Default assumption


def get_peak_memory_mb(device: str) -> float:
    """Get peak memory usage in MB."""
    if device == "mps":
        try:
            if hasattr(torch.mps, "driver_allocated_memory"):
                return torch.mps.driver_allocated_memory() / 1024 / 1024
        except Exception:
            pass
        # Fallback to psutil for unified memory tracking
        try:
            import psutil
            return psutil.Process().memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    if device == "cuda":
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0.0


def _setup_sys_path() -> None:
    """Add upstream diffusers and RealRestorer to sys.path."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    local_diffusers = repo_root / "upstream-realrestorer" / "diffusers" / "src"
    if local_diffusers.is_dir() and str(local_diffusers) not in sys.path:
        sys.path.insert(0, str(local_diffusers))

    upstream_root = repo_root / "upstream-realrestorer"
    if upstream_root.is_dir() and str(upstream_root) not in sys.path:
        sys.path.insert(0, str(upstream_root))


def load_pipeline(
    model_path: str,
    device: str,
    dtype: torch.dtype,
    quantize: str = "none",
) -> Any:
    """Load the RealRestorer pipeline with device-appropriate optimizations.

    Uses module-level caching to avoid reloading 39GB of weights on
    repeated calls (critical for benchmark loops and agent workflows).
    """
    global _pipeline_cache, _cache_key

    cache_key = (model_path, device, str(dtype), quantize)
    if cache_key in _pipeline_cache:
        return _pipeline_cache[cache_key]

    _setup_sys_path()
    from diffusers import RealRestorerPipeline

    # CRITICAL: Always load on CPU first. device_map="mps" silently
    # corrupts weights (diffusers #13227). low_cpu_mem_usage prevents
    # peak memory spike during loading.
    pipe = RealRestorerPipeline.from_pretrained(
        model_path,
        torch_dtype=dtype,
        use_safetensors=True,
        low_cpu_mem_usage=True,
    )

    # Apply quantization BEFORE moving to device (saves memory during transfer)
    if quantize == "int8":
        _apply_int8_quantization(pipe, device)
    elif quantize == "int4":
        _apply_int4_quantization(pipe, device)

    if device == "mps":
        from realrestore_cli.optimizations.mps_backend import configure_mps_environment
        configure_mps_environment()

        # Proven FLUX community pattern for large DiT models on MPS:
        # 1. Load as bfloat16 (same as training dtype, no precision loss)
        # 2. Upcast VAE to float32 (GroupNorm produces NaN in bf16/fp16)
        # 3. RoPE computed on CPU with float64 (patched in upstream)
        # 4. Everything on MPS — no CPU offloading needed
        # Memory: ~20GB bfloat16 weights + 0.6GB VAE float32 + activations
        pipe.vae = pipe.vae.to(dtype=torch.float32)
        pipe.to(device)

    elif device == "cuda":
        pipe.enable_model_cpu_offload()

    else:
        pipe.to(device)

    # Cache the loaded pipeline
    _pipeline_cache[cache_key] = pipe
    _cache_key = cache_key

    return pipe


def _apply_int8_quantization(pipe: Any, device: str) -> None:
    """Apply int8 quantization using Quanto (research-recommended for MPS)."""
    from realrestore_cli.optimizations.quantize import quantize_pipeline
    quantize_pipeline(pipe, weights_dtype="int8", device=device)


def _apply_int4_quantization(pipe: Any, device: str) -> None:
    """Apply int4 quantization using Quanto or torchao."""
    from realrestore_cli.optimizations.quantize import quantize_pipeline
    quantize_pipeline(pipe, weights_dtype="int4", device=device)


def restore_image(
    input_path: str,
    output_path: str,
    task: str = "auto",
    backend: str = "auto",
    quantize: str = "none",
    steps: int = 28,
    seed: int = 42,
    prompt: str = "",
    quality: str | None = None,
    tile: bool = False,
    tile_size: int = 512,
    tile_overlap: int = 64,
) -> dict[str, Any]:
    """Run image restoration and return metrics."""
    start_time = time.time()

    # Auto-detect degradation type if task is "auto"
    detected_task = task
    detection_confidence = 0.0
    if task == "auto":
        try:
            from realrestore_cli.optimizations.auto_detect import auto_detect
            detection = auto_detect(input_path)
            detected_task = detection.get("task", "auto")
            detection_confidence = detection.get("confidence", 0.0)
        except Exception:
            detected_task = "auto"

    # Apply quality preset
    if quality:
        from realrestore_cli.optimizations.scheduling import QualityPreset, get_scheduler_config
        try:
            preset = QualityPreset(quality)
            config = get_scheduler_config(preset)
            steps = config.num_steps
        except (ValueError, KeyError):
            pass

    # Resolve device
    if backend == "auto":
        device = os.environ.get("REALRESTORE_BACKEND", get_device())
    else:
        device = backend

    dtype = get_dtype(device)

    # Resolve prompt
    if not prompt:
        prompt = TASK_PROMPTS.get(detected_task, TASK_PROMPTS["auto"])

    # Load model
    model_path = get_model_path()
    pipe = load_pipeline(model_path, device, dtype, quantize)

    # Load input image
    image = Image.open(input_path).convert("RGB")

    # Run inference
    if device == "mps":
        from realrestore_cli.optimizations.mps_backend import synchronize
        result = pipe(
            image=image,
            prompt=prompt,
            negative_prompt=DEFAULT_NEGATIVE_PROMPT,
            num_inference_steps=steps,
            guidance_scale=3.0,
            seed=seed,
            size_level=1024,
        )
        synchronize()  # Sync only at end for accurate timing
    else:
        result = pipe(
            image=image,
            prompt=prompt,
            negative_prompt=DEFAULT_NEGATIVE_PROMPT,
            num_inference_steps=steps,
            guidance_scale=3.0,
            seed=seed,
            size_level=1024,
        )

    # Save output
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.images[0].save(out_path)

    elapsed = time.time() - start_time
    peak_mem = get_peak_memory_mb(device)

    return {
        "input": input_path,
        "output": str(out_path),
        "task": detected_task,
        "task_requested": task,
        "detection_confidence": round(detection_confidence, 4),
        "backend": device,
        "quantize": quantize,
        "steps": steps,
        "seed": seed,
        "elapsed_seconds": round(elapsed, 2),
        "peak_memory_mb": round(peak_mem, 1),
        "image_size": list(image.size),
    }


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task", default="auto")
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--quantize", default="none")
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = restore_image(
            input_path=args.input,
            output_path=args.output,
            task=args.task,
            backend=args.backend,
            quantize=args.quantize,
            steps=args.steps,
            seed=args.seed,
            prompt=args.prompt,
        )
        print(json.dumps(result))
    except Exception as e:
        error = {
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
        print(json.dumps(error), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
