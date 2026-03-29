"""Invisible watermark removal using multiple techniques.

Supports spectral filtering, diffusion-based removal, and ensemble methods.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def remove_spectral(
    image: np.ndarray,
    strength: float = 0.5,
) -> np.ndarray:
    """Remove watermarks via frequency domain filtering.

    Applies a notch filter to suppress anomalous frequency components
    while preserving image content.
    """
    result = np.zeros_like(image, dtype=np.float64)

    for c in range(image.shape[2] if len(image.shape) == 3 else 1):
        channel = image[:, :, c] if len(image.shape) == 3 else image
        channel = channel.astype(np.float64)

        # Forward DFT
        f_transform = np.fft.fft2(channel)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        phase = np.angle(f_shift)

        # Identify anomalous peaks in mid-high frequencies
        h, w = channel.shape
        ch, cw = h // 2, w // 2
        max_r = min(ch, cw)

        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((y - ch) ** 2 + (x - cw) ** 2) / max_r

        # Focus on watermark-prone frequency range
        target_mask = (dist >= 0.2) & (dist < 0.85)
        log_mag = np.log1p(magnitude)

        # Z-score based filtering — suppress outlier frequencies
        target_values = log_mag[target_mask]
        mean_val = np.mean(target_values)
        std_val = np.std(target_values)

        if std_val > 1e-6:
            z_scores = np.abs((log_mag - mean_val) / std_val)
            # Soft suppression of anomalous frequencies
            suppression = np.ones_like(magnitude)
            anomaly_mask = target_mask & (z_scores > 2.5)
            suppression[anomaly_mask] *= (1.0 - strength)

            # Apply smooth transition
            from scipy.ndimage import gaussian_filter
            suppression = gaussian_filter(suppression, sigma=2)

            magnitude *= suppression

        # Reconstruct
        f_modified = magnitude * np.exp(1j * phase)
        f_ishift = np.fft.ifftshift(f_modified)
        reconstructed = np.real(np.fft.ifft2(f_ishift))

        if len(image.shape) == 3:
            result[:, :, c] = reconstructed
        else:
            result = reconstructed

    return np.clip(result, 0, 255).astype(np.uint8)


def remove_dwt(
    image: np.ndarray,
    strength: float = 0.5,
    levels: int = 3,
) -> np.ndarray:
    """Remove watermarks using Discrete Wavelet Transform filtering."""
    try:
        import pywt
    except ImportError:
        # Fallback to spectral if pywt not available
        return remove_spectral(image, strength)

    result = np.zeros_like(image, dtype=np.float64)

    for c in range(image.shape[2] if len(image.shape) == 3 else 1):
        channel = image[:, :, c] if len(image.shape) == 3 else image
        channel = channel.astype(np.float64)

        # Multi-level DWT
        coeffs = pywt.wavedec2(channel, "haar", level=levels)

        # Soft threshold detail coefficients
        modified_coeffs = [coeffs[0]]  # Keep approximation unchanged
        for level_coeffs in coeffs[1:]:
            modified = []
            for detail in level_coeffs:
                threshold = strength * np.std(detail)
                # Soft thresholding
                sign = np.sign(detail)
                abs_val = np.abs(detail)
                thresholded = sign * np.maximum(abs_val - threshold, 0)
                modified.append(thresholded)
            modified_coeffs.append(tuple(modified))

        # Reconstruct
        reconstructed = pywt.waverec2(modified_coeffs, "haar")

        # Handle size mismatch from wavelet reconstruction
        h, w = channel.shape
        reconstructed = reconstructed[:h, :w]

        if len(image.shape) == 3:
            result[:, :, c] = reconstructed
        else:
            result = reconstructed

    return np.clip(result, 0, 255).astype(np.uint8)


def remove_ensemble(
    image: np.ndarray,
    strength: float = 0.5,
) -> np.ndarray:
    """Ensemble watermark removal combining multiple techniques.

    Weighted combination of spectral and DWT methods for robust removal.
    """
    spectral_result = remove_spectral(image, strength).astype(np.float64)
    dwt_result = remove_dwt(image, strength).astype(np.float64)

    # Weighted blend — DWT is generally better for structured watermarks
    result = 0.4 * spectral_result + 0.6 * dwt_result

    return np.clip(result, 0, 255).astype(np.uint8)


def strip_metadata(image_path: str, output_path: str) -> None:
    """Strip all metadata (EXIF, XMP, C2PA) from image."""
    img = Image.open(image_path)
    # Create new image without metadata
    clean = Image.new(img.mode, img.size)
    clean.putdata(list(img.getdata()))
    clean.save(output_path)


def remove_watermark(
    input_path: str,
    output_path: str,
    method: str = "ensemble",
    strength: float = 0.5,
) -> dict[str, Any]:
    """Remove invisible watermarks from an image.

    Args:
        input_path: Path to input image
        output_path: Path for cleaned output
        method: Removal method (spectral, dwt, ensemble, diffusion)
        strength: Removal strength 0.0-1.0

    Returns:
        Dictionary with metrics and status
    """
    start_time = time.time()

    img = Image.open(input_path).convert("RGB")
    image_array = np.array(img)

    # Apply removal method
    if method == "spectral":
        cleaned = remove_spectral(image_array, strength)
    elif method == "dwt":
        cleaned = remove_dwt(image_array, strength)
    elif method == "ensemble":
        cleaned = remove_ensemble(image_array, strength)
    elif method == "diffusion":
        # Diffusion-based removal uses the restoration pipeline
        # This naturally disrupts watermark patterns
        cleaned = _remove_via_diffusion(input_path, output_path)
        if cleaned is None:
            # Fallback to ensemble
            cleaned = remove_ensemble(image_array, strength)
    else:
        cleaned = remove_ensemble(image_array, strength)

    # Save result
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(cleaned).save(out_path)

    # Also strip metadata
    strip_metadata(str(out_path), str(out_path))

    # Compute quality metrics
    psnr = _compute_psnr(image_array, cleaned)

    elapsed = time.time() - start_time

    return {
        "input": input_path,
        "output": str(out_path),
        "method": method,
        "strength": strength,
        "psnr_vs_original": round(psnr, 2),
        "elapsed_seconds": round(elapsed, 2),
        "image_size": list(img.size),
    }


def _remove_via_diffusion(input_path: str, output_path: str) -> np.ndarray | None:
    """Use restoration pipeline for watermark removal.

    Running an image through a diffusion restoration pipeline naturally
    disrupts most watermark patterns while preserving content.
    """
    try:
        from realrestore_cli.engine import restore_image

        result = restore_image(
            input_path=input_path,
            output_path=output_path,
            task="auto",
            steps=15,  # Fewer steps — just enough to disrupt watermark
        )
        return np.array(Image.open(output_path).convert("RGB"))
    except Exception:
        return None


def _compute_psnr(original: np.ndarray, cleaned: np.ndarray) -> float:
    """Compute PSNR between original and cleaned image."""
    mse = np.mean((original.astype(np.float64) - cleaned.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return float("inf")
    return 10 * np.log10(255.0 ** 2 / mse)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--method", default="ensemble")
    parser.add_argument("--strength", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = remove_watermark(
            input_path=args.input,
            output_path=args.output,
            method=args.method,
            strength=args.strength,
        )
        print(json.dumps(result))
    except Exception as e:
        import traceback
        print(json.dumps({"error": str(e), "traceback": traceback.format_exc()}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
