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


def remove_dct(
    image: np.ndarray,
    strength: float = 0.5,
    block_size: int = 8,
) -> np.ndarray:
    """Remove watermarks using block DCT coefficient quantization.

    Watermarks embedded in the DCT domain typically modify mid-frequency
    coefficients within 8x8 blocks (the same structure used by JPEG).
    This method applies soft quantization to those coefficients, disrupting
    the watermark while preserving low-frequency image content.

    Args:
        image: Input image array (H, W, C) in [0, 255].
        strength: Removal strength 0.0-1.0.  Higher values quantize more
                  aggressively, removing more watermark energy at the cost
                  of potential block artifacts.
        block_size: DCT block size (default 8, matching JPEG).
    """
    from scipy.fft import dctn, idctn

    result = np.zeros_like(image, dtype=np.float64)

    # Quantization factor — maps strength to an effective divisor.
    # strength=0 → quant=1 (no change), strength=1 → quant=0.3 (heavy)
    quant_factor = max(0.3, 1.0 - strength * 0.7)

    for c in range(image.shape[2] if len(image.shape) == 3 else 1):
        channel = (image[:, :, c] if len(image.shape) == 3 else image).astype(np.float64)
        h, w = channel.shape

        # Pad to a multiple of block_size
        pad_h = (block_size - h % block_size) % block_size
        pad_w = (block_size - w % block_size) % block_size
        padded = np.pad(channel, ((0, pad_h), (0, pad_w)), mode="reflect")

        for i in range(0, padded.shape[0], block_size):
            for j in range(0, padded.shape[1], block_size):
                block = padded[i : i + block_size, j : j + block_size]
                dct_block = dctn(block, type=2, norm="ortho")

                # Quantize mid-frequency coefficients (freq index 3–10)
                for u in range(block_size):
                    for v in range(block_size):
                        freq = u + v
                        if 3 <= freq <= 10:
                            dct_block[u, v] = (
                                np.round(dct_block[u, v] * quant_factor) / quant_factor
                            )

                padded[i : i + block_size, j : j + block_size] = idctn(
                    dct_block, type=2, norm="ortho"
                )

        if len(image.shape) == 3:
            result[:, :, c] = padded[:h, :w]
        else:
            result = padded[:h, :w]

    return np.clip(result, 0, 255).astype(np.uint8)


def remove_adversarial_purification(
    image: np.ndarray,
    noise_strength: float = 0.15,
) -> np.ndarray:
    """Remove watermarks via adversarial purification (forward + reverse diffusion).

    Adds controlled Gaussian noise (simulating a partial forward diffusion
    step) to disrupt the watermark pattern, then denoises to recover a
    clean image.  The amount of noise controls the trade-off between
    watermark destruction and image preservation.

    For heavy watermarks, use ``noise_strength`` around 0.25-0.35.
    For light / unknown watermarks, 0.10-0.15 typically suffices.

    The denoising stage uses a non-local-means style bilateral filter
    rather than a full diffusion model so the method works without GPU
    or model weights.

    Args:
        image: Input image array (H, W, C) in [0, 255].
        noise_strength: Fraction of the pixel range [0, 255] used as the
                        noise standard deviation.  0.0 = no noise, 1.0 =
                        full-range noise (extremely destructive).
    """
    from scipy.ndimage import gaussian_filter

    img_f = image.astype(np.float64)

    # --- Forward step: add controlled noise ---
    sigma = noise_strength * 255.0
    rng = np.random.default_rng(seed=42)
    noise = rng.normal(0.0, sigma, img_f.shape)
    noisy = img_f + noise

    # --- Reverse step: denoise ---
    # Two-pass approach: Gaussian smoothing followed by an edge-aware
    # weighted blend with the noisy input to preserve sharpness.
    smooth_sigma = max(0.8, noise_strength * 4.0)
    smoothed = np.zeros_like(noisy)
    for c in range(noisy.shape[2] if len(noisy.shape) == 3 else 1):
        ch = noisy[:, :, c] if len(noisy.shape) == 3 else noisy
        smoothed_ch = gaussian_filter(ch, sigma=smooth_sigma)

        # Edge-aware blend: keep sharp details where local variance is high
        local_var = gaussian_filter((ch - smoothed_ch) ** 2, sigma=smooth_sigma * 2)
        # Normalize weight so that high-variance (edge) areas keep more
        # of the noisy signal, low-variance (flat) areas take the smooth.
        weight = np.clip(local_var / (np.percentile(local_var, 95) + 1e-6), 0, 1)
        blended = weight * ch + (1.0 - weight) * smoothed_ch

        if len(noisy.shape) == 3:
            smoothed[:, :, c] = blended
        else:
            smoothed = blended

    return np.clip(smoothed, 0, 255).astype(np.uint8)


def remove_ensemble(
    image: np.ndarray,
    strength: float = 0.5,
) -> np.ndarray:
    """Ensemble watermark removal combining multiple techniques.

    Weighted combination of spectral, DWT, and DCT methods for robust
    removal across different watermark embedding domains.
    """
    spectral_result = remove_spectral(image, strength).astype(np.float64)
    dwt_result = remove_dwt(image, strength).astype(np.float64)
    dct_result = remove_dct(image, strength).astype(np.float64)

    # Weighted blend — each method targets different embedding domains
    result = 0.35 * spectral_result + 0.40 * dwt_result + 0.25 * dct_result

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
        input_path: Path to input image.
        output_path: Path for cleaned output.
        method: Removal method — one of ``spectral``, ``dwt``, ``dct``,
                ``adversarial``, ``ensemble``, or ``diffusion``.
        strength: Removal strength 0.0-1.0.

    Returns:
        Dictionary with quality metrics, watermark detection scores
        before/after, and timing information.
    """
    start_time = time.time()

    img = Image.open(input_path).convert("RGB")
    image_array = np.array(img)

    # Apply removal method
    if method == "spectral":
        cleaned = remove_spectral(image_array, strength)
    elif method == "dwt":
        cleaned = remove_dwt(image_array, strength)
    elif method == "dct":
        cleaned = remove_dct(image_array, strength)
    elif method == "adversarial":
        cleaned = remove_adversarial_purification(image_array, noise_strength=strength)
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

    # Compute quality metrics (PSNR + watermark detection before/after)
    from realrestore_cli.watermark.detector import compute_quality_metrics

    quality = compute_quality_metrics(image_array, cleaned)

    elapsed = time.time() - start_time

    return {
        "input": input_path,
        "output": str(out_path),
        "method": method,
        "strength": strength,
        "psnr_vs_original": quality["psnr"],
        "watermark_scores": {
            "spectral_before": quality["original_spectral_anomaly"],
            "spectral_after": quality["processed_spectral_anomaly"],
            "stegastamp_before": quality["original_stegastamp_score"],
            "stegastamp_after": quality["processed_stegastamp_score"],
            "tree_ring_before": quality["original_tree_ring_score"],
            "tree_ring_after": quality["processed_tree_ring_score"],
        },
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
    parser = argparse.ArgumentParser(description="Remove invisible watermarks from images.")
    parser.add_argument("--input", required=True, help="Path to input image")
    parser.add_argument("--output", required=True, help="Path for cleaned output")
    parser.add_argument(
        "--method",
        default="ensemble",
        choices=["spectral", "dwt", "dct", "adversarial", "ensemble", "diffusion"],
        help="Removal method (default: ensemble)",
    )
    parser.add_argument("--strength", type=float, default=0.5, help="Removal strength 0.0-1.0")
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
