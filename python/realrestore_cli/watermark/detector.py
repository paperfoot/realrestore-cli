"""Invisible watermark detection using spectral analysis.

Detects various AI watermarking schemes by analyzing frequency domain
patterns that are invisible in spatial domain.  Supports StegaStamp
frequency-pattern matching, Tree-Ring spectral ring analysis, and
C2PA / Content Credentials metadata inspection.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def analyze_frequency_domain(image: np.ndarray) -> dict[str, Any]:
    """Analyze image in frequency domain for watermark patterns.

    Computes the 2-D DFT, splits the magnitude spectrum into four radial
    frequency bands and returns per-band statistics together with an
    anomaly score and phase-coherence metric.
    """
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


def detect_stegastamp(image: np.ndarray) -> dict[str, Any]:
    """Detect StegaStamp watermarks via frequency pattern matching.

    StegaStamp embeds bitstrings into low/mid-frequency bands spread across
    the image.  Detection looks for characteristic energy elevation in the
    mid-frequency band relative to natural image statistics and checks for
    spatial uniformity of mid-frequency energy (StegaStamp modifies the
    entire image uniformly, unlike natural textures).
    """
    if len(image.shape) == 3:
        gray = np.mean(image, axis=2)
    else:
        gray = image.copy()

    f_shift = np.fft.fftshift(np.fft.fft2(gray))
    magnitude = np.abs(f_shift)
    log_mag = np.log1p(magnitude)

    h, w = gray.shape
    cy, cx = h // 2, w // 2
    max_r = min(cy, cx)

    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((y - cy) ** 2 + (x - cx) ** 2) / max_r

    # StegaStamp concentrates in low-to-mid bands (0.05 – 0.35 normalised)
    stega_mask = (dist >= 0.05) & (dist < 0.35)
    # Reference band outside the typical StegaStamp range
    ref_mask = (dist >= 0.5) & (dist < 0.8)

    stega_energy = np.mean(log_mag[stega_mask])
    ref_energy = np.mean(log_mag[ref_mask])
    energy_ratio = stega_energy / (ref_energy + 1e-6)

    # Spatial uniformity check – split image into quadrants and compare
    # mid-frequency energy.  StegaStamp produces uniform modification.
    block_h, block_w = h // 2, w // 2
    quadrant_energies = []
    for qy in range(2):
        for qx in range(2):
            block = gray[qy * block_h : (qy + 1) * block_h, qx * block_w : (qx + 1) * block_w]
            bf = np.fft.fftshift(np.fft.fft2(block))
            bh, bw = block.shape
            bcy, bcx = bh // 2, bw // 2
            b_max_r = min(bcy, bcx)
            by, bx = np.ogrid[:bh, :bw]
            bdist = np.sqrt((by - bcy) ** 2 + (bx - bcx) ** 2) / (b_max_r + 1e-6)
            bmask = (bdist >= 0.05) & (bdist < 0.35)
            quadrant_energies.append(float(np.mean(np.log1p(np.abs(bf))[bmask])))

    uniformity = 1.0 - (np.std(quadrant_energies) / (np.mean(quadrant_energies) + 1e-6))
    uniformity = max(0.0, min(1.0, uniformity))

    # Score: high energy ratio + high uniformity → likely StegaStamp
    # Empirically, energy_ratio > 1.8 and uniformity > 0.85 are suspicious
    score = 0.0
    if energy_ratio > 1.5:
        score += min(1.0, (energy_ratio - 1.5) / 1.0) * 0.6
    if uniformity > 0.8:
        score += min(1.0, (uniformity - 0.8) / 0.15) * 0.4
    score = min(1.0, score)

    return {
        "method": "stegastamp",
        "energy_ratio": round(float(energy_ratio), 4),
        "uniformity": round(float(uniformity), 4),
        "score": round(float(score), 4),
        "likely": score > 0.5,
    }


def detect_tree_ring(image: np.ndarray) -> dict[str, Any]:
    """Detect Tree-Ring watermarks via spectral ring analysis.

    Tree-Ring embeds a circular pattern in the Fourier transform of the
    initial noise vector.  After generation the pattern manifests as
    concentric-ring energy concentrations in the magnitude spectrum of
    the output image.  Detection computes the radial power profile and
    looks for anomalous peaks compared to the smooth fall-off expected
    in natural images.
    """
    if len(image.shape) == 3:
        gray = np.mean(image, axis=2)
    else:
        gray = image.copy()

    f_shift = np.fft.fftshift(np.fft.fft2(gray))
    magnitude = np.abs(f_shift)
    log_mag = np.log1p(magnitude)

    h, w = gray.shape
    cy, cx = h // 2, w // 2

    # Build radial distance map
    y_coords, x_coords = np.ogrid[:h, :w]
    r_map = np.sqrt((y_coords - cy) ** 2 + (x_coords - cx) ** 2).astype(int)
    max_r = int(min(cy, cx))

    # Compute radial power profile
    radial_sum = np.bincount(r_map.ravel(), log_mag.ravel(), minlength=max_r + 1)[:max_r + 1]
    radial_count = np.bincount(r_map.ravel(), minlength=max_r + 1)[:max_r + 1]
    radial_count = np.maximum(radial_count, 1)
    radial_profile = radial_sum / radial_count

    # Smooth baseline via moving average
    kernel_size = max(5, max_r // 20)
    kernel = np.ones(kernel_size) / kernel_size
    baseline = np.convolve(radial_profile, kernel, mode="same")

    # Residual peaks indicate ring patterns
    residual = radial_profile - baseline
    residual_std = np.std(residual)

    if residual_std < 1e-6:
        peak_score = 0.0
        peak_radii: list[int] = []
    else:
        z_scores = residual / residual_std
        # Peaks above 3-sigma in the mid-frequency range
        mid_start = max_r // 10
        mid_end = int(max_r * 0.7)
        z_mid = z_scores[mid_start:mid_end]
        peak_mask = z_mid > 3.0
        peak_count = int(np.sum(peak_mask))
        peak_radii = (np.nonzero(peak_mask)[0] + mid_start).tolist()

        # Tree-Ring produces multiple concentric peaks
        peak_score = min(1.0, peak_count / 5.0)

    return {
        "method": "tree_ring",
        "peak_count": len(peak_radii),
        "peak_radii": peak_radii[:10],  # Cap list length
        "score": round(float(peak_score), 4),
        "likely": peak_score > 0.4,
    }


def detect_metadata_watermark(image_path: str) -> dict[str, Any]:
    """Check for metadata-based watermarks (C2PA, XMP, EXIF).

    Inspects EXIF tags, XMP packets, and PNG text chunks for C2PA
    Content Credentials manifests, AI-generation markers, and other
    provenance metadata.
    """
    results: dict[str, Any] = {"method": "metadata", "found": []}

    _AI_KEYWORDS = [
        "c2pa",
        "content credentials",
        "synthid",
        "dall-e",
        "midjourney",
        "stable diffusion",
        "openai",
        "adobe firefly",
        "imagen",
    ]

    try:
        img = Image.open(image_path)
        exif = img.getexif()

        # Check EXIF tags for AI / C2PA markers
        if exif:
            for tag_id, value in exif.items():
                tag_name = str(tag_id)
                if isinstance(value, str) and any(kw in value.lower() for kw in _AI_KEYWORDS):
                    results["found"].append({"tag": tag_name, "value": str(value)[:200]})

        # Check XMP data
        if hasattr(img, "info") and "xmp" in img.info:
            xmp_data = img.info["xmp"]
            if isinstance(xmp_data, bytes):
                xmp_str = xmp_data.decode("utf-8", errors="ignore")
                for marker in ["c2pa", "ai:generated", "dc:creator", "stds:c2pa"]:
                    if marker.lower() in xmp_str.lower():
                        results["found"].append({"type": "xmp", "marker": marker})

        # PNG-specific text chunks
        if img.format == "PNG" and hasattr(img, "info"):
            for key, value in img.info.items():
                key_lower = str(key).lower()
                if key_lower in ("c2pa", "content-credentials"):
                    results["found"].append({"type": "png_text_chunk", "key": key})
                elif isinstance(value, str) and any(kw in value.lower() for kw in _AI_KEYWORDS):
                    results["found"].append({"type": "png_text_chunk", "key": key, "snippet": str(value)[:200]})

        # JFIF / APP markers in JPEG (C2PA JUMBF box indicator)
        if img.format == "JPEG" and hasattr(img, "applist"):
            for marker, data in getattr(img, "applist", []):
                if isinstance(data, bytes) and b"c2pa" in data.lower():
                    results["found"].append({"type": "jpeg_app_marker", "marker": marker})

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
