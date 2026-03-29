"""Automatic degradation type detection using classical image analysis.

Lightweight detectors for common image degradation types using only
numpy, scipy, and PIL — no heavy ML dependencies required.

Each detector returns a confidence score in [0, 1]. The top-level
auto_detect function picks the most likely degradation or returns
"auto" if no detector is confident enough.
"""
from __future__ import annotations

import json
import sys
from typing import Any

import numpy as np
from PIL import Image


# Minimum confidence to claim a specific degradation type
_CONFIDENCE_THRESHOLD = 0.45


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert RGB image array to float64 grayscale."""
    if image.ndim == 3:
        return np.mean(image, axis=2).astype(np.float64)
    return image.astype(np.float64)


# ---------------------------------------------------------------------------
# Individual detectors
# ---------------------------------------------------------------------------

def detect_blur(image: np.ndarray) -> float:
    """Detect blur via Laplacian variance and edge density.

    Low Laplacian variance and sparse edges indicate a blurry image.
    """
    from scipy.ndimage import convolve

    gray = _to_grayscale(image)

    # Laplacian kernel
    laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
    lap = convolve(gray, laplacian)
    lap_var = np.var(lap)

    # Edge density via Sobel magnitude
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    sobel_y = sobel_x.T
    gx = convolve(gray, sobel_x)
    gy = convolve(gray, sobel_y)
    edge_mag = np.sqrt(gx ** 2 + gy ** 2)

    # Fraction of pixels with significant edges
    edge_threshold = np.percentile(edge_mag, 90)
    edge_density = np.mean(edge_mag > edge_threshold * 0.5) if edge_threshold > 1e-6 else 1.0

    # Combine: low variance + low edge density => blurry
    # Typical sharp image: lap_var > 500, edge_density > 0.15
    blur_score_lap = 1.0 - min(1.0, lap_var / 800.0)
    blur_score_edge = 1.0 - min(1.0, edge_density / 0.20)
    score = 0.6 * blur_score_lap + 0.4 * blur_score_edge

    return float(np.clip(score, 0.0, 1.0))


def detect_noise(image: np.ndarray) -> float:
    """Detect noise via local variance estimation.

    High-frequency residual energy after median filtering indicates noise.
    """
    from scipy.ndimage import median_filter

    gray = _to_grayscale(image)

    # Estimate noise as the difference between the image and its median-filtered version
    filtered = median_filter(gray, size=3)
    residual = gray - filtered

    # Noise level: std of the residual
    noise_std = np.std(residual)

    # Local variance in small patches
    patch_size = 7
    h, w = gray.shape
    n_patches = min(500, (h // patch_size) * (w // patch_size))
    if n_patches < 10:
        return 0.0

    rng = np.random.RandomState(42)
    local_vars = []
    for _ in range(n_patches):
        y = rng.randint(0, h - patch_size)
        x = rng.randint(0, w - patch_size)
        patch = residual[y : y + patch_size, x : x + patch_size]
        local_vars.append(np.var(patch))

    mean_local_var = np.mean(local_vars)

    # Typical noisy images: noise_std > 8, mean_local_var > 30
    score_std = min(1.0, noise_std / 15.0)
    score_var = min(1.0, mean_local_var / 60.0)
    score = 0.5 * score_std + 0.5 * score_var

    return float(np.clip(score, 0.0, 1.0))


def detect_haze(image: np.ndarray) -> float:
    """Detect haze via dark channel prior and contrast analysis.

    Hazy images have bright dark channels and low contrast.
    """
    img = image.astype(np.float64)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=2)

    # Dark channel prior: min across channels, then local min
    dark = np.min(img, axis=2)

    # Local minimum over a patch (approximate erosion)
    from scipy.ndimage import minimum_filter
    dark_channel = minimum_filter(dark, size=15)

    # In haze-free images, the dark channel has many near-zero values
    dark_mean = np.mean(dark_channel) / 255.0

    # Contrast analysis
    gray = _to_grayscale(image)
    contrast = np.std(gray) / (np.mean(gray) + 1e-6)

    # Low contrast + bright dark channel => haze
    haze_dark = min(1.0, dark_mean / 0.5)
    haze_contrast = 1.0 - min(1.0, contrast / 0.6)
    score = 0.55 * haze_dark + 0.45 * haze_contrast

    return float(np.clip(score, 0.0, 1.0))


def detect_compression(image: np.ndarray) -> float:
    """Detect JPEG compression artifacts via 8x8 block boundary analysis.

    JPEG encodes images in 8x8 blocks, creating visible boundaries
    at high compression levels.
    """
    from scipy.ndimage import convolve

    gray = _to_grayscale(image)
    h, w = gray.shape
    if h < 32 or w < 32:
        return 0.0

    # Compute horizontal and vertical gradients
    grad_h = np.abs(np.diff(gray, axis=1))
    grad_v = np.abs(np.diff(gray, axis=0))

    # Extract gradients at 8-pixel boundaries vs non-boundaries
    # Horizontal block boundaries
    boundary_cols = np.arange(7, w - 1, 8)
    non_boundary_cols = np.array([c for c in range(w - 1) if c % 8 != 7])
    if len(boundary_cols) == 0 or len(non_boundary_cols) == 0:
        return 0.0

    mean_boundary_h = np.mean(grad_h[:, boundary_cols])
    mean_non_boundary_h = np.mean(grad_h[:, non_boundary_cols])

    # Vertical block boundaries
    boundary_rows = np.arange(7, h - 1, 8)
    non_boundary_rows = np.array([r for r in range(h - 1) if r % 8 != 7])
    if len(boundary_rows) == 0 or len(non_boundary_rows) == 0:
        return 0.0

    mean_boundary_v = np.mean(grad_v[boundary_rows, :])
    mean_non_boundary_v = np.mean(grad_v[non_boundary_rows, :])

    # Block boundary strength: ratio of boundary to non-boundary gradients
    ratio_h = mean_boundary_h / (mean_non_boundary_h + 1e-6)
    ratio_v = mean_boundary_v / (mean_non_boundary_v + 1e-6)
    block_strength = (ratio_h + ratio_v) / 2.0

    # Also check for reduced high-frequency energy (compression removes it)
    laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
    hf_energy = np.std(convolve(gray, laplacian))

    # block_strength > 1.2 suggests blocking artifacts
    score_block = min(1.0, max(0.0, (block_strength - 1.0) / 0.5))

    # Low HF energy also correlates with heavy compression
    score_hf = 1.0 - min(1.0, hf_energy / 30.0)

    score = 0.65 * score_block + 0.35 * score_hf

    return float(np.clip(score, 0.0, 1.0))


def detect_low_light(image: np.ndarray) -> float:
    """Detect low-light conditions via histogram and brightness analysis."""
    gray = _to_grayscale(image)

    mean_brightness = np.mean(gray) / 255.0
    median_brightness = np.median(gray) / 255.0

    # Histogram concentration in dark region
    hist, _ = np.histogram(gray, bins=256, range=(0, 255))
    hist = hist.astype(np.float64) / hist.sum()
    dark_ratio = np.sum(hist[:64])  # fraction of pixels in bottom 25%

    # Low mean brightness + high dark ratio => low light
    score_mean = 1.0 - min(1.0, mean_brightness / 0.35)
    score_median = 1.0 - min(1.0, median_brightness / 0.30)
    score_dark = min(1.0, dark_ratio / 0.70)

    score = 0.35 * score_mean + 0.35 * score_median + 0.30 * score_dark

    return float(np.clip(score, 0.0, 1.0))


def detect_moire(image: np.ndarray) -> float:
    """Detect moire patterns via frequency domain periodicity analysis.

    Moire creates strong periodic peaks in the frequency spectrum
    away from the DC component.
    """
    gray = _to_grayscale(image)
    h, w = gray.shape

    # Use center crop for efficiency on large images
    max_dim = 512
    if h > max_dim or w > max_dim:
        ch, cw = h // 2, w // 2
        half = max_dim // 2
        gray = gray[ch - half : ch + half, cw - half : cw + half]
        h, w = gray.shape

    # 2D FFT
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)
    magnitude = np.log1p(np.abs(f_shift))

    # Suppress DC component
    center_h, center_w = h // 2, w // 2
    dc_radius = max(3, min(h, w) // 30)
    y, x = np.ogrid[:h, :w]
    dc_mask = np.sqrt((y - center_h) ** 2 + (x - center_w) ** 2) > dc_radius
    magnitude_no_dc = magnitude * dc_mask

    # Detect periodic peaks: moire creates isolated strong peaks
    mean_mag = np.mean(magnitude_no_dc[dc_mask])
    std_mag = np.std(magnitude_no_dc[dc_mask])
    if std_mag < 1e-6:
        return 0.0

    # Count significant spectral peaks (z-score > 4)
    z_scores = (magnitude_no_dc - mean_mag) / std_mag
    peak_ratio = np.sum(z_scores > 4.0) / np.sum(dc_mask)

    # Also check for radial symmetry in peaks (moire is often symmetric)
    # by comparing opposing quadrants
    q1 = magnitude_no_dc[:center_h, center_w:]
    q3 = magnitude_no_dc[center_h:, :center_w]
    q3_flipped = q3[::-1, ::-1]
    min_h = min(q1.shape[0], q3_flipped.shape[0])
    min_w = min(q1.shape[1], q3_flipped.shape[1])
    if min_h > 0 and min_w > 0:
        symmetry = np.corrcoef(
            q1[:min_h, :min_w].ravel(),
            q3_flipped[:min_h, :min_w].ravel(),
        )[0, 1]
        symmetry = max(0.0, symmetry)
    else:
        symmetry = 0.0

    # Moire: many sharp spectral peaks + high symmetry
    score_peaks = min(1.0, peak_ratio / 0.005)
    score_sym = symmetry ** 2  # amplify strong symmetry

    score = 0.6 * score_peaks + 0.4 * score_sym

    return float(np.clip(score, 0.0, 1.0))


def detect_rain(image: np.ndarray) -> float:
    """Detect rain streaks via directional filtering.

    Rain appears as near-vertical bright streaks with consistent direction.
    """
    from scipy.ndimage import convolve

    gray = _to_grayscale(image)

    # Vertical streak detection kernel (emphasizes vertical lines)
    vertical_kernel = np.array([
        [-1, 2, -1],
        [-1, 2, -1],
        [-1, 2, -1],
        [-1, 2, -1],
        [-1, 2, -1],
    ], dtype=np.float64) / 5.0

    # Horizontal kernel for comparison
    horizontal_kernel = vertical_kernel.T

    vert_response = np.abs(convolve(gray, vertical_kernel))
    horiz_response = np.abs(convolve(gray, horizontal_kernel))

    # Rain has strong vertical response relative to horizontal
    mean_vert = np.mean(vert_response)
    mean_horiz = np.mean(horiz_response)
    directionality = mean_vert / (mean_horiz + 1e-6)

    # Rain streaks are bright and thin — check for many thin bright vertical features
    vert_threshold = np.percentile(vert_response, 95)
    streak_density = np.mean(vert_response > vert_threshold * 0.7) if vert_threshold > 1e-6 else 0.0

    # Directionality > 1.3 and reasonable streak density suggest rain
    score_dir = min(1.0, max(0.0, (directionality - 1.0) / 0.6))
    score_density = min(1.0, streak_density / 0.08)

    score = 0.55 * score_dir + 0.45 * score_density

    return float(np.clip(score, 0.0, 1.0))


def detect_lens_flare(image: np.ndarray) -> float:
    """Detect lens flare via bright spot analysis and bloom detection.

    Lens flare creates localized bright regions with gradual falloff (bloom)
    and sometimes colored halos.
    """
    gray = _to_grayscale(image)
    h, w = gray.shape

    # Detect very bright regions
    bright_threshold = np.percentile(gray, 98)
    bright_mask = gray > bright_threshold
    bright_ratio = np.mean(bright_mask)

    # Check if bright regions are localized (flare) vs uniform (overexposure)
    if bright_ratio < 0.001 or bright_ratio > 0.3:
        # Too few or too many bright pixels — unlikely flare
        return 0.0

    # Bloom detection: bright regions should have gradual falloff
    from scipy.ndimage import gaussian_filter, label

    # Smooth the image and find connected bright components
    smoothed = gaussian_filter(gray, sigma=5)
    smooth_bright = smoothed > np.percentile(smoothed, 95)
    labeled, n_components = label(smooth_bright)

    if n_components == 0:
        return 0.0

    # Flare typically has 1-5 bright components, not many
    component_score = 1.0 - min(1.0, max(0.0, (n_components - 1)) / 20.0)

    # Check for bloom: ratio of smoothed bright area to sharp bright area
    sharp_bright = np.sum(bright_mask)
    smooth_bright_area = np.sum(smooth_bright)
    if sharp_bright < 1:
        return 0.0
    bloom_ratio = smooth_bright_area / sharp_bright
    bloom_score = min(1.0, max(0.0, (bloom_ratio - 1.0) / 5.0))

    # Color halo detection (if RGB)
    if image.ndim == 3:
        img_f = image.astype(np.float64)
        # In flare regions, color channels diverge (chromatic aberration)
        channel_std = np.std(img_f, axis=2)
        flare_region_color_var = np.mean(channel_std[bright_mask]) if np.any(bright_mask) else 0.0
        halo_score = min(1.0, flare_region_color_var / 40.0)
    else:
        halo_score = 0.0

    score = 0.35 * bloom_score + 0.35 * component_score + 0.30 * halo_score

    return float(np.clip(score, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Main detection interface
# ---------------------------------------------------------------------------

# Maps detector names to (function, task_key) pairs
_DETECTORS: dict[str, tuple[Any, str]] = {
    "blur": (detect_blur, "deblur"),
    "noise": (detect_noise, "denoise"),
    "haze": (detect_haze, "dehaze"),
    "compression": (detect_compression, "compression"),
    "low_light": (detect_low_light, "low_light"),
    "moire": (detect_moire, "moire"),
    "rain": (detect_rain, "derain"),
    "lens_flare": (detect_lens_flare, "lens_flare"),
}


def auto_detect(image_path: str) -> dict[str, Any]:
    """Detect the most likely degradation type in an image.

    Runs all lightweight detectors and returns the top match.

    Args:
        image_path: Path to the input image.

    Returns:
        Dict with keys:
            - degradation: detected type or "auto" if uncertain
            - task: corresponding RealRestore task key
            - confidence: confidence score of the top match
            - scores: dict of all detector scores
    """
    img = np.array(Image.open(image_path).convert("RGB"))

    scores: dict[str, float] = {}
    for name, (detector_fn, _) in _DETECTORS.items():
        try:
            scores[name] = detector_fn(img)
        except Exception:
            scores[name] = 0.0

    # Pick the highest-scoring degradation
    best_name = max(scores, key=scores.get)  # type: ignore[arg-type]
    best_score = scores[best_name]
    best_task = _DETECTORS[best_name][1]

    if best_score < _CONFIDENCE_THRESHOLD:
        return {
            "degradation": "auto",
            "task": "auto",
            "confidence": round(best_score, 4),
            "scores": {k: round(v, 4) for k, v in scores.items()},
        }

    return {
        "degradation": best_name,
        "task": best_task,
        "confidence": round(best_score, 4),
        "scores": {k: round(v, 4) for k, v in scores.items()},
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: python auto_detect.py <image_path>"}))
        sys.exit(3)

    result = auto_detect(sys.argv[1])
    print(json.dumps(result, indent=2))
