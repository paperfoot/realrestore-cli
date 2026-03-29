"""Automated benchmark runner for comparing backends and optimizations.

Measures wall time, peak memory, and quality metrics (PSNR/SSIM/LPIPS)
across different backends and quantization levels. Outputs JSON for
both the Rust CLI and autoresearch optimization loops.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def create_test_image(size: tuple[int, int] = (512, 512)) -> str:
    """Create a test image with varied content for benchmarking."""
    img = Image.new("RGB", size)
    pixels = img.load()

    # Create a pattern with gradients, edges, and texture
    for y in range(size[1]):
        for x in range(size[0]):
            r = int(128 + 127 * np.sin(x * 0.05))
            g = int(128 + 127 * np.cos(y * 0.05))
            b = int(128 + 127 * np.sin((x + y) * 0.03))
            pixels[x, y] = (r, g, b)

    # Add some noise to make it realistic
    arr = np.array(img)
    noise = np.random.randint(-20, 20, arr.shape, dtype=np.int16)
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    img.save(tmp.name)
    return tmp.name


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute Peak Signal-to-Noise Ratio."""
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return float("inf")
    return float(10 * np.log10(255.0 ** 2 / mse))


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute Structural Similarity Index."""
    try:
        from skimage.metrics import structural_similarity as ssim
        # Convert to grayscale for SSIM
        if len(img1.shape) == 3:
            gray1 = np.mean(img1, axis=2).astype(np.float64)
            gray2 = np.mean(img2, axis=2).astype(np.float64)
        else:
            gray1, gray2 = img1.astype(np.float64), img2.astype(np.float64)
        return float(ssim(gray1, gray2, data_range=255.0))
    except ImportError:
        # Manual SSIM computation
        return _manual_ssim(img1, img2)


def _manual_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Simple SSIM implementation without scikit-image."""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    if len(img1.shape) == 3:
        img1 = np.mean(img1, axis=2)
        img2 = np.mean(img2, axis=2)

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    sigma1_sq = np.var(img1)
    sigma2_sq = np.var(img2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))

    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(ssim)


def compute_lpips(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute LPIPS perceptual similarity (lower = more similar)."""
    try:
        import torch
        import lpips

        loss_fn = lpips.LPIPS(net="alex", verbose=False)

        # Convert to tensors
        t1 = torch.from_numpy(img1).float().permute(2, 0, 1).unsqueeze(0) / 255.0 * 2 - 1
        t2 = torch.from_numpy(img2).float().permute(2, 0, 1).unsqueeze(0) / 255.0 * 2 - 1

        with torch.no_grad():
            score = loss_fn(t1, t2)
        return float(score.item())
    except Exception:
        return -1.0


def run_single_benchmark(
    image_path: str,
    backend: str,
    quantize: str = "none",
    steps: int = 10,
    seed: int = 42,
) -> dict[str, Any]:
    """Run a single benchmark iteration."""
    from realrestore_cli.engine import restore_image

    out_tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)

    try:
        result = restore_image(
            input_path=image_path,
            output_path=out_tmp.name,
            backend=backend,
            quantize=quantize,
            steps=steps,
            seed=seed,
        )

        # Compute quality metrics
        original = np.array(Image.open(image_path).convert("RGB"))
        restored = np.array(Image.open(out_tmp.name).convert("RGB"))

        result["psnr"] = compute_psnr(original, restored)
        result["ssim"] = compute_ssim(original, restored)
        result["lpips"] = compute_lpips(original, restored)

        return result

    except Exception as e:
        return {
            "error": str(e),
            "backend": backend,
            "quantize": quantize,
            "elapsed_seconds": -1,
            "peak_memory_mb": -1,
        }
    finally:
        try:
            os.unlink(out_tmp.name)
        except Exception:
            pass


def run_benchmarks(
    iterations: int = 3,
    backends: str = "auto",
    image_path: str | None = None,
    quantize_levels: list[str] | None = None,
    steps: int = 10,
) -> dict[str, Any]:
    """Run comprehensive benchmarks across backends and quantization levels."""
    from realrestore_cli.engine import get_device

    # Resolve backends
    if backends == "auto":
        available_backends = [get_device()]
    else:
        available_backends = [b.strip() for b in backends.split(",")]

    if quantize_levels is None:
        quantize_levels = ["none"]

    # Create test image if needed
    if image_path is None:
        image_path = create_test_image()

    results = []

    for backend in available_backends:
        for quantize in quantize_levels:
            config_results = []

            for i in range(iterations):
                result = run_single_benchmark(
                    image_path=image_path,
                    backend=backend,
                    quantize=quantize,
                    steps=steps,
                )
                config_results.append(result)

            # Aggregate
            valid = [r for r in config_results if r.get("elapsed_seconds", -1) > 0]

            if valid:
                summary = {
                    "backend": backend,
                    "quantize": quantize,
                    "steps": steps,
                    "iterations": iterations,
                    "successful": len(valid),
                    "avg_time": round(np.mean([r["elapsed_seconds"] for r in valid]), 2),
                    "min_time": round(min(r["elapsed_seconds"] for r in valid), 2),
                    "max_time": round(max(r["elapsed_seconds"] for r in valid), 2),
                    "std_time": round(float(np.std([r["elapsed_seconds"] for r in valid])), 3),
                    "peak_memory_mb": round(max(r.get("peak_memory_mb", 0) for r in valid), 1),
                    "avg_psnr": round(np.mean([r.get("psnr", 0) for r in valid]), 2),
                    "avg_ssim": round(np.mean([r.get("ssim", 0) for r in valid]), 4),
                    "avg_lpips": round(np.mean([r.get("lpips", -1) for r in valid if r.get("lpips", -1) >= 0]), 4) if any(r.get("lpips", -1) >= 0 for r in valid) else -1,
                }
            else:
                errors = [r.get("error", "unknown") for r in config_results]
                summary = {
                    "backend": backend,
                    "quantize": quantize,
                    "steps": steps,
                    "iterations": iterations,
                    "successful": 0,
                    "error": errors[0] if errors else "all iterations failed",
                }

            results.append(summary)

    benchmark_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "image": image_path,
        "iterations": iterations,
        "results": results,
    }

    # Save to history file
    _save_benchmark_history(benchmark_data)

    return benchmark_data


def _save_benchmark_history(data: dict[str, Any]) -> None:
    """Append benchmark results to history file for regression tracking."""
    history_dir = Path("benchmark_results")
    history_dir.mkdir(exist_ok=True)

    history_file = history_dir / "history.jsonl"
    with open(history_file, "a") as f:
        f.write(json.dumps(data) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--backends", default="auto")
    parser.add_argument("--image", default=None)
    parser.add_argument("--quantize", default="none", help="Comma-separated quantize levels")
    parser.add_argument("--steps", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    quantize_levels = [q.strip() for q in args.quantize.split(",")]

    try:
        result = run_benchmarks(
            iterations=args.iterations,
            backends=args.backends,
            image_path=args.image,
            quantize_levels=quantize_levels,
            steps=args.steps,
        )
        print(json.dumps(result))
    except Exception as e:
        import traceback
        print(json.dumps({"error": str(e), "traceback": traceback.format_exc()}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
