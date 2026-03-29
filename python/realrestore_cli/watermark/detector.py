"""Invisible watermark detection using spectral analysis.

Detects various AI watermarking schemes by analyzing frequency domain
patterns that are invisible in spatial domain.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def analyze_frequency_domain(image: np.ndarray) -> dict[str, Any]:
    """Analyze image in frequency domain for watermark patterns."""
    # Convert to grayscale for analysis
    if len(image.shape) == 3:
        gray = np.mean(image, axis=2)
    else:
        gray = image.copy()

    # Apply 2D DFT
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)
    magnitude = np.abs(f_shift)
    phase = np.angle(f_shift)

    # Analyze magnitude spectrum for anomalies
    log_magnitude = np.log1p(magnitude)

    # Statistical analysis of frequency components
    h, w = gray.shape
    center_h, center_w = h // 2, w // 2

    # Split into frequency bands
    bands = {
        "low": _extract_band(log_magnitude, center_h, center_w, 0, 0.1),
        "mid_low": _extract_band(log_magnitude, center_h, center_w, 0.1, 0.3),
        "mid_high": _extract_band(log_magnitude, center_h, center_w, 0.3, 0.6),
        "high": _extract_band(log_magnitude, center_h, center_w, 0.6, 1.0),
    }

    # Detect anomalous peaks in mid-high frequencies (typical watermark location)
    anomaly_score = _detect_spectral_anomalies(log_magnitude, center_h, center_w)

    return {
        "frequency_bands": {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in bands.items()},
        "anomaly_score": float(anomaly_score),
        "phase_coherence": float(_phase_coherence(phase)),
    }


def _extract_band(
    magnitude: np.ndarray, ch: int, cw: int, r_min: float, r_max: float
) -> np.ndarray:
    """Extract frequency band between r_min and r_max (normalized radius)."""
    h, w = magnitude.shape
    max_r = min(ch, cw)
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((y - ch) ** 2 + (x - cw) ** 2) / max_r
    mask = (dist >= r_min) & (dist < r_max)
    return magnitude[mask]


def _detect_spectral_anomalies(
    log_magnitude: np.ndarray, ch: int, cw: int
) -> float:
    """Detect anomalous spectral peaks that may indicate watermarks."""
    h, w = log_magnitude.shape
    max_r = min(ch, cw)

    # Focus on mid-to-high frequency region where watermarks typically hide
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((y - ch) ** 2 + (x - cw) ** 2) / max_r
    mid_high_mask = (dist >= 0.3) & (dist < 0.8)

    region = log_magnitude[mid_high_mask]
    if len(region) == 0:
        return 0.0

    # Z-score based anomaly detection
    mean_val = np.mean(region)
    std_val = np.std(region)
    if std_val < 1e-6:
        return 0.0

    # Count significant outliers
    z_scores = np.abs((region - mean_val) / std_val)
    anomaly_ratio = np.sum(z_scores > 3.0) / len(region)

    # Normalize to 0-1 score
    return min(1.0, anomaly_ratio * 100)


def _phase_coherence(phase: np.ndarray) -> float:
    """Measure phase coherence — watermarks often introduce phase patterns."""
    # Check for periodic phase patterns
    phase_diffs = np.diff(phase, axis=0)
    coherence = 1.0 - np.std(phase_diffs) / (np.pi + 1e-6)
    return max(0.0, min(1.0, coherence))


def detect_dwt_watermark(image: np.ndarray) -> dict[str, Any]:
    """Detect watermarks using Discrete Wavelet Transform analysis."""
    try:
        import pywt
    except ImportError:
        return {"method": "dwt", "available": False, "note": "pywt not installed"}

    if len(image.shape) == 3:
        gray = np.mean(image, axis=2).astype(np.float64)
    else:
        gray = image.astype(np.float64)

    # Multi-level DWT decomposition
    coeffs = pywt.wavedec2(gray, "haar", level=3)

    # Analyze detail coefficients for watermark signatures
    detail_stats = []
    for level, (cH, cV, cD) in enumerate(coeffs[1:], 1):
        stats = {
            "level": level,
            "horizontal": {"mean": float(np.mean(np.abs(cH))), "std": float(np.std(cH))},
            "vertical": {"mean": float(np.mean(np.abs(cV))), "std": float(np.std(cV))},
            "diagonal": {"mean": float(np.mean(np.abs(cD))), "std": float(np.std(cD))},
        }
        detail_stats.append(stats)

    # Watermark score based on unusual energy in detail coefficients
    energy_ratios = []
    for stats in detail_stats:
        for direction in ["horizontal", "vertical", "diagonal"]:
            ratio = stats[direction]["mean"] / (stats[direction]["std"] + 1e-6)
            energy_ratios.append(ratio)

    anomaly_score = float(np.std(energy_ratios))

    return {
        "method": "dwt",
        "available": True,
        "detail_stats": detail_stats,
        "anomaly_score": anomaly_score,
    }


def detect_metadata_watermark(image_path: str) -> dict[str, Any]:
    """Check for metadata-based watermarks (C2PA, XMP, EXIF)."""
    results: dict[str, Any] = {"method": "metadata", "found": []}

    try:
        img = Image.open(image_path)
        exif = img.getexif()

        # Check for C2PA/Content Credentials markers
        if exif:
            for tag_id, value in exif.items():
                tag_name = str(tag_id)
                if isinstance(value, str) and any(
                    kw in value.lower()
                    for kw in ["c2pa", "content credentials", "synthid", "dall-e", "midjourney", "stable diffusion"]
                ):
                    results["found"].append({"tag": tag_name, "value": str(value)[:200]})

        # Check XMP data
        if hasattr(img, "info") and "xmp" in img.info:
            xmp_data = img.info["xmp"]
            if isinstance(xmp_data, bytes):
                xmp_str = xmp_data.decode("utf-8", errors="ignore")
                for marker in ["c2pa", "ai:generated", "dc:creator"]:
                    if marker.lower() in xmp_str.lower():
                        results["found"].append({"type": "xmp", "marker": marker})

    except Exception as e:
        results["error"] = str(e)

    results["has_watermark"] = len(results["found"]) > 0
    return results


def detect_watermarks(image_path: str) -> dict[str, Any]:
    """Run all watermark detection methods on an image."""
    img = np.array(Image.open(image_path).convert("RGB")).astype(np.float64)

    results = {
        "image": image_path,
        "spectral": analyze_frequency_domain(img),
        "metadata": detect_metadata_watermark(image_path),
    }

    # Try DWT if available
    dwt_result = detect_dwt_watermark(img)
    results["dwt"] = dwt_result

    # Overall assessment
    scores = [results["spectral"]["anomaly_score"]]
    if dwt_result.get("available"):
        scores.append(min(1.0, dwt_result["anomaly_score"] / 5.0))

    results["overall_score"] = float(np.mean(scores))
    results["likely_watermarked"] = (
        results["overall_score"] > 0.3 or results["metadata"]["has_watermark"]
    )

    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: python detector.py <image_path>"}))
        sys.exit(3)

    result = detect_watermarks(sys.argv[1])
    print(json.dumps(result, indent=2))
