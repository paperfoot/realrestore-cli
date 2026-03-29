"""Quanto-based quantization module for diffusion model pipelines.

Implements int8 and int4 weight quantization targeting transformer/UNet
backbone components while preserving VAE at full precision (critical
for pixel-level restoration quality).

Research-backed decisions (2026-03-29):
- Quanto is the recommended MPS-compatible quantization path
- INT8 is the default for restoration (near-lossless quality)
- INT4 reserved for memory-constrained scenarios (quality cliff for pixel tasks)
- VAE must stay at FP16/BF16 — only ~168 MB, directly affects output quality
- MPS emulates int8/int4 via upcast to BF16/FP32 (no native Metal kernels)
- Calibration with domain-specific images improves quantization accuracy
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Components that are safe to quantize (large, moderate sensitivity)
QUANTIZABLE_COMPONENTS = ("transformer", "unet")

# Components that must stay at full precision (small, high sensitivity)
PROTECTED_COMPONENTS = ("vae", "vae_encoder", "vae_decoder")

# Expected model sizes in MB at FP16 (for memory estimation)
COMPONENT_SIZES_FP16_MB: dict[str, float] = {
    "transformer": 24000.0,
    "unet": 24000.0,
    "text_encoder": 5000.0,
    "text_encoder_2": 10000.0,
    "vae": 168.0,
}


def quantize_pipeline(
    pipe: Any,
    weights_dtype: str = "int8",
    device: str = "mps",
    calibration_images: list[str] | None = None,
) -> dict[str, Any]:
    """Quantize pipeline components using Quanto with torch.ao fallback.

    Args:
        pipe: A diffusers pipeline instance.
        weights_dtype: Target weight dtype — "int8" or "int4".
        device: Target device (affects fallback strategy).
        calibration_images: Optional paths to calibration images for
            improved quantization accuracy.

    Returns:
        Dict with quantization results and metadata.
    """
    start = time.time()
    results: dict[str, Any] = {
        "method": "unknown",
        "weights_dtype": weights_dtype,
        "components_quantized": [],
        "components_skipped": [],
    }

    # Try Quanto first (research-recommended for MPS)
    quantized = _try_quanto(pipe, weights_dtype, calibration_images, results)

    if not quantized:
        # Fallback to torch.ao (CPU-only for dynamic quantization)
        _try_torchao(pipe, weights_dtype, device, results)

    results["elapsed_seconds"] = round(time.time() - start, 2)
    return results


def _try_quanto(
    pipe: Any,
    weights_dtype: str,
    calibration_images: list[str] | None,
    results: dict[str, Any],
) -> bool:
    """Attempt quantization via optimum-quanto.

    Returns True if Quanto was available and quantization succeeded.
    """
    try:
        from optimum.quanto import freeze, qint4, qint8, quantize
    except ImportError:
        return False

    qtype = qint8 if weights_dtype == "int8" else qint4
    results["method"] = "quanto"

    # Run calibration if images provided (feeds forward pass data
    # so Quanto can observe activation ranges before freezing)
    if calibration_images:
        _run_calibration(pipe, calibration_images)

    for name in QUANTIZABLE_COMPONENTS:
        if not hasattr(pipe, name):
            continue
        module = getattr(pipe, name)
        if module is None:
            continue

        try:
            quantize(module, weights=qtype)
            freeze(module)
            results["components_quantized"].append(name)
        except Exception as e:
            results["components_skipped"].append({"name": name, "reason": str(e)})

    for name in PROTECTED_COMPONENTS:
        if hasattr(pipe, name) and getattr(pipe, name) is not None:
            results["components_skipped"].append({"name": name, "reason": "protected"})

    return True


def _try_torchao(
    pipe: Any,
    weights_dtype: str,
    device: str,
    results: dict[str, Any],
) -> None:
    """Fallback quantization via torch.ao.

    Dynamic quantization only works on CPU — MPS and CUDA need Quanto.
    """
    if device != "cpu":
        results["method"] = "none"
        results["error"] = (
            f"Quanto not installed and torch.ao dynamic quantization "
            f"only supports CPU (got device={device}). "
            f"Install optimum-quanto: pip install optimum-quanto"
        )
        return

    results["method"] = "torchao"

    ao_dtype = torch.qint8
    # torch.ao dynamic quantization does not support int4 — use int8
    if weights_dtype == "int4":
        results["torchao_note"] = "int4 not supported by torch.ao dynamic; using int8"
        ao_dtype = torch.qint8

    for name in QUANTIZABLE_COMPONENTS:
        if not hasattr(pipe, name):
            continue
        module = getattr(pipe, name)
        if module is None:
            continue

        try:
            quantized_module = torch.ao.quantization.quantize_dynamic(
                module, {torch.nn.Linear}, dtype=ao_dtype
            )
            setattr(pipe, name, quantized_module)
            results["components_quantized"].append(name)
        except Exception as e:
            results["components_skipped"].append({"name": name, "reason": str(e)})


def _run_calibration(pipe: Any, image_paths: list[str]) -> None:
    """Feed calibration images through the pipeline to prime activation stats.

    Quanto observes activation distributions during forward passes
    before freeze() locks in the quantization parameters. Running a
    few domain-representative images improves quantization accuracy.
    """
    from PIL import Image

    for path in image_paths[:8]:  # Cap at 8 to keep calibration fast
        try:
            img = Image.open(path).convert("RGB")
            # Run a minimal forward pass (1 step) for calibration
            pipe(
                image=img,
                prompt="calibration",
                num_inference_steps=1,
                guidance_scale=1.0,
                seed=0,
                size_level=512,
            )
        except Exception:
            continue


def estimate_memory_savings(
    pipe: Any,
    weights_dtype: str = "int8",
) -> dict[str, Any]:
    """Estimate memory savings from quantization.

    Computes expected savings based on model parameter counts and
    target bit widths. Does not modify the pipeline.
    """
    bits_map = {"int8": 8, "int4": 4}
    target_bits = bits_map.get(weights_dtype, 8)
    # FP16 baseline
    baseline_bits = 16

    report: dict[str, Any] = {
        "weights_dtype": weights_dtype,
        "components": {},
        "total_params": 0,
        "total_baseline_mb": 0.0,
        "total_quantized_mb": 0.0,
        "total_savings_mb": 0.0,
        "savings_percent": 0.0,
    }

    for name in list(QUANTIZABLE_COMPONENTS) + list(PROTECTED_COMPONENTS):
        if not hasattr(pipe, name):
            continue
        module = getattr(pipe, name)
        if module is None:
            continue

        params = sum(p.numel() for p in module.parameters())
        baseline_mb = (params * baseline_bits) / 8 / 1024 / 1024
        is_protected = name in PROTECTED_COMPONENTS

        if is_protected:
            quantized_mb = baseline_mb
        else:
            quantized_mb = (params * target_bits) / 8 / 1024 / 1024

        report["components"][name] = {
            "params": params,
            "baseline_mb": round(baseline_mb, 1),
            "quantized_mb": round(quantized_mb, 1),
            "savings_mb": round(baseline_mb - quantized_mb, 1),
            "quantized": not is_protected,
        }
        report["total_params"] += params
        report["total_baseline_mb"] += baseline_mb
        report["total_quantized_mb"] += quantized_mb

    report["total_baseline_mb"] = round(report["total_baseline_mb"], 1)
    report["total_quantized_mb"] = round(report["total_quantized_mb"], 1)
    report["total_savings_mb"] = round(
        report["total_baseline_mb"] - report["total_quantized_mb"], 1
    )
    if report["total_baseline_mb"] > 0:
        report["savings_percent"] = round(
            100 * report["total_savings_mb"] / report["total_baseline_mb"], 1
        )

    return report


def assess_quality_impact(
    pipe: Any,
    reference_image_path: str,
    weights_dtype: str = "int8",
) -> dict[str, Any]:
    """Assess quantization quality impact by comparing FP16 vs quantized output.

    Runs inference at full precision, then at the target quantization level,
    and computes PSNR/SSIM between the two outputs.

    Args:
        pipe: Pipeline (should be at full precision before calling).
        reference_image_path: Path to test image.
        weights_dtype: Quantization level to assess.

    Returns:
        Dict with quality metrics and impact assessment.
    """
    from PIL import Image

    img = Image.open(reference_image_path).convert("RGB")
    inference_kwargs = {
        "image": img,
        "prompt": "Restore the details and keep the original composition.",
        "num_inference_steps": 10,
        "guidance_scale": 3.0,
        "seed": 42,
        "size_level": 512,
    }

    # Run at full precision
    fp_result = pipe(**inference_kwargs)
    fp_output = np.array(fp_result.images[0])

    # Apply quantization
    quant_info = quantize_pipeline(pipe, weights_dtype=weights_dtype)

    # Run at quantized precision
    q_result = pipe(**inference_kwargs)
    q_output = np.array(q_result.images[0])

    # Compute quality metrics
    psnr = _compute_psnr(fp_output, q_output)
    ssim = _compute_ssim(fp_output, q_output)

    # Quality thresholds for restoration
    # PSNR > 40 dB = near-lossless, 35-40 = acceptable, < 35 = degraded
    if psnr == float("inf") or psnr > 40:
        impact = "near-lossless"
    elif psnr > 35:
        impact = "acceptable"
    elif psnr > 30:
        impact = "moderate-degradation"
    else:
        impact = "severe-degradation"

    return {
        "weights_dtype": weights_dtype,
        "psnr_db": round(psnr, 2) if psnr != float("inf") else "inf",
        "ssim": round(ssim, 4),
        "quality_impact": impact,
        "quantization": quant_info,
        "recommendation": _quality_recommendation(impact, weights_dtype),
    }


def _quality_recommendation(impact: str, weights_dtype: str) -> str:
    """Generate a human-readable recommendation based on quality impact."""
    if impact == "near-lossless":
        return f"{weights_dtype} quantization is safe for production use."
    if impact == "acceptable":
        return f"{weights_dtype} quantization shows minor quality loss; acceptable for most use cases."
    if impact == "moderate-degradation":
        if weights_dtype == "int4":
            return "int4 shows visible quality loss. Consider using int8 for restoration tasks."
        return f"{weights_dtype} quantization shows moderate quality loss; evaluate per use case."
    return f"{weights_dtype} quantization causes severe quality loss; not recommended for restoration."


def _compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute Peak Signal-to-Noise Ratio between two images."""
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return float("inf")
    return float(10 * np.log10(255.0 ** 2 / mse))


def _compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute Structural Similarity Index between two images."""
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


def get_quantization_info() -> dict[str, Any]:
    """Report available quantization backends and capabilities."""
    info: dict[str, Any] = {
        "quanto_available": False,
        "quanto_version": None,
        "torchao_available": False,
        "torch_version": torch.__version__,
        "supported_dtypes": [],
        "recommended": "int8",
    }

    try:
        import optimum.quanto
        info["quanto_available"] = True
        info["quanto_version"] = getattr(optimum.quanto, "__version__", "unknown")
        info["supported_dtypes"] = ["int8", "int4"]
    except ImportError:
        pass

    try:
        _ = torch.ao.quantization.quantize_dynamic
        info["torchao_available"] = True
        if not info["quanto_available"]:
            info["supported_dtypes"] = ["int8"]  # torch.ao lacks int4 dynamic
    except AttributeError:
        pass

    if not info["quanto_available"] and not info["torchao_available"]:
        info["error"] = "No quantization backend available. Install optimum-quanto."

    return info


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description="Quantization utilities for RealRestore CLI"
    )
    parser.add_argument(
        "--info", action="store_true",
        help="Show available quantization backends"
    )
    parser.add_argument(
        "--estimate", action="store_true",
        help="Estimate memory savings (requires model to be loadable)"
    )
    parser.add_argument(
        "--dtype", default="int8", choices=["int8", "int4"],
        help="Target weight dtype"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.info:
        print(json.dumps(get_quantization_info(), indent=2))
        return

    if args.estimate:
        # Provide static estimates based on known component sizes
        bits_map = {"int8": 8, "int4": 4}
        target_bits = bits_map[args.dtype]
        baseline_bits = 16
        ratio = target_bits / baseline_bits

        estimates = {}
        total_baseline = 0.0
        total_quantized = 0.0

        for name, size_mb in COMPONENT_SIZES_FP16_MB.items():
            is_protected = name in PROTECTED_COMPONENTS
            q_size = size_mb if is_protected else size_mb * ratio
            estimates[name] = {
                "baseline_mb": size_mb,
                "quantized_mb": round(q_size, 1),
                "savings_mb": round(size_mb - q_size, 1),
                "quantized": not is_protected,
            }
            total_baseline += size_mb
            total_quantized += q_size

        result = {
            "weights_dtype": args.dtype,
            "components": estimates,
            "total_baseline_mb": round(total_baseline, 1),
            "total_quantized_mb": round(total_quantized, 1),
            "total_savings_mb": round(total_baseline - total_quantized, 1),
            "savings_percent": round(
                100 * (total_baseline - total_quantized) / total_baseline, 1
            ),
        }
        print(json.dumps(result, indent=2))
        return

    # Default: show info
    print(json.dumps(get_quantization_info(), indent=2))


if __name__ == "__main__":
    main()
