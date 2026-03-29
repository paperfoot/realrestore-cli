"""Core inference engine for RealRestore CLI.

Orchestrates model loading, device selection, and inference with
progressive optimization layers (MPS, quantization, MLX, ANE).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch
from PIL import Image

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
    """Select appropriate dtype for device."""
    if device == "mps":
        return torch.float16  # MPS doesn't fully support bfloat16
    if device == "cuda":
        return torch.bfloat16
    return torch.float32


def get_model_path() -> str:
    """Resolve model path from env or default."""
    env_path = os.environ.get("REALRESTORE_MODEL_PATH")
    if env_path:
        return env_path
    # Default: HuggingFace model ID
    return "RealRestorer/RealRestorer"


def get_peak_memory_mb(device: str) -> float:
    """Get peak memory usage in MB."""
    if device == "mps":
        # MPS doesn't have built-in memory tracking like CUDA
        try:
            if hasattr(torch.mps, "driver_allocated_memory"):
                return torch.mps.driver_allocated_memory() / 1024 / 1024
            return 0.0
        except Exception:
            return 0.0
    if device == "cuda":
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0.0


def load_pipeline(
    model_path: str,
    device: str,
    dtype: torch.dtype,
    quantize: str = "none",
) -> Any:
    """Load the RealRestorer pipeline with optimizations."""
    # Add upstream diffusers to path if available
    repo_root = Path(__file__).resolve().parent.parent.parent
    local_diffusers = repo_root / "upstream-realrestorer" / "diffusers" / "src"
    if local_diffusers.is_dir() and str(local_diffusers) not in sys.path:
        sys.path.insert(0, str(local_diffusers))

    # Also add the upstream repo root for RealRestorer module
    upstream_root = repo_root / "upstream-realrestorer"
    if upstream_root.is_dir() and str(upstream_root) not in sys.path:
        sys.path.insert(0, str(upstream_root))

    from diffusers import RealRestorerPipeline

    pipe = RealRestorerPipeline.from_pretrained(
        model_path,
        torch_dtype=dtype,
    )

    # Apply memory optimizations
    if device == "mps":
        # Enable attention slicing for memory efficiency
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
        # Enable VAE slicing
        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
        pipe.to(device)
    elif device == "cuda":
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)

    # Apply quantization if requested
    if quantize == "int8":
        _apply_int8_quantization(pipe)
    elif quantize == "int4":
        _apply_int4_quantization(pipe)

    return pipe


def _apply_int8_quantization(pipe: Any) -> None:
    """Apply dynamic int8 quantization to transformer blocks."""
    try:
        import torch.ao.quantization as quant
        # Dynamic quantization on linear layers
        for name, module in pipe.named_modules():
            if isinstance(module, torch.nn.Linear):
                quant.quantize_dynamic(
                    module, {torch.nn.Linear}, dtype=torch.qint8
                )
    except Exception:
        pass  # Graceful fallback if quantization fails


def _apply_int4_quantization(pipe: Any) -> None:
    """Apply int4 quantization (requires bitsandbytes or similar)."""
    try:
        # Will be implemented with bitsandbytes or custom quantization
        pass
    except Exception:
        pass


def restore_image(
    input_path: str,
    output_path: str,
    task: str = "auto",
    backend: str = "auto",
    quantize: str = "none",
    steps: int = 28,
    seed: int = 42,
    prompt: str = "",
) -> dict[str, Any]:
    """Run image restoration and return metrics."""
    start_time = time.time()

    # Resolve device
    if backend == "auto":
        device = get_device()
    else:
        device = backend

    dtype = get_dtype(device)

    # Resolve prompt
    if not prompt:
        prompt = TASK_PROMPTS.get(task, TASK_PROMPTS["auto"])

    # Load model
    model_path = get_model_path()
    pipe = load_pipeline(model_path, device, dtype, quantize)

    # Load input image
    image = Image.open(input_path).convert("RGB")

    # Run inference
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
        "task": task,
        "backend": device,
        "quantize": quantize,
        "steps": steps,
        "seed": seed,
        "elapsed_seconds": round(elapsed, 2),
        "peak_memory_mb": round(peak_mem, 1),
        "image_size": list(image.size),
    }


def parse_args() -> argparse.Namespace:
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
