"""Image-level tiling for high-resolution restoration.

Splits large images into overlapping tiles, restores each independently,
then blends results using Gaussian weighting for seamless boundaries.

Supports arbitrary input sizes (2K, 4K, 8K+). Tiles are processed
sequentially to keep memory bounded — only one tile is on-device at a time.
Overlap with Gaussian blending maintains global coherence at seams.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from typing import Any, Callable

import numpy as np
from PIL import Image


def get_available_memory_mb() -> float:
    """Get available system memory in MB."""
    try:
        import psutil
        return psutil.virtual_memory().available / (1024 * 1024)
    except ImportError:
        pass

    try:
        import subprocess
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            # Use 70% of total as estimate when psutil unavailable
            total_mb = int(result.stdout.strip()) / (1024 * 1024)
            return total_mb * 0.7
    except Exception:
        pass

    return 16_000.0  # Conservative 16GB fallback


def select_tile_size(
    image_width: int,
    image_height: int,
    available_memory_mb: float | None = None,
) -> int:
    """Select tile size based on available memory.

    Heuristic: a single tile at float16 through a diffusion model uses
    roughly (tile_size^2 * 3 * 2 * 20) bytes for activations and
    intermediate tensors (~20x multiplier for UNet-style models).

    Args:
        image_width: Input image width.
        image_height: Input image height.
        available_memory_mb: Available memory in MB. Auto-detected if None.

    Returns:
        Tile size in pixels (always a multiple of 64 for model compatibility).
    """
    if available_memory_mb is None:
        available_memory_mb = get_available_memory_mb()

    # Memory budget: use at most 60% of available for the tile
    budget_mb = available_memory_mb * 0.6

    # Estimate: tile_size^2 * 3 channels * 2 bytes (fp16) * 20x activation multiplier
    bytes_per_pixel = 3 * 2 * 20
    max_pixels = (budget_mb * 1024 * 1024) / bytes_per_pixel
    max_tile = int(math.sqrt(max_pixels))

    # Clamp to reasonable range and align to 64px
    max_tile = max(512, min(max_tile, 2048))
    tile_size = (max_tile // 64) * 64

    # If the image fits in a single tile, just use the image size
    max_dim = max(image_width, image_height)
    if max_dim <= tile_size:
        aligned = ((max_dim + 63) // 64) * 64
        return max(512, aligned)

    return tile_size


def _build_gaussian_weight(tile_h: int, tile_w: int) -> np.ndarray:
    """Build a 2D Gaussian weight map for blending.

    The Gaussian peaks at the tile center and falls off toward edges,
    giving interior pixels higher weight during blending. This produces
    seamless transitions at tile boundaries.

    Returns:
        Array of shape (tile_h, tile_w) with values in (0, 1].
    """
    sigma_y = tile_h / 4.0
    sigma_x = tile_w / 4.0

    cy, cx = tile_h / 2.0, tile_w / 2.0
    y = np.arange(tile_h, dtype=np.float64)
    x = np.arange(tile_w, dtype=np.float64)
    yy, xx = np.meshgrid(y, x, indexing="ij")

    weight = np.exp(-((yy - cy) ** 2 / (2 * sigma_y ** 2) +
                      (xx - cx) ** 2 / (2 * sigma_x ** 2)))

    # Ensure minimum weight to avoid division issues
    weight = np.clip(weight, 1e-6, 1.0)
    return weight


def compute_tiles(
    image_width: int,
    image_height: int,
    tile_size: int,
    overlap: int = 64,
) -> list[tuple[int, int, int, int]]:
    """Compute tile coordinates with overlap.

    Tiles are laid out in a grid with `overlap` pixels shared between
    adjacent tiles. Edge tiles are clamped to image boundaries.

    Args:
        image_width: Full image width.
        image_height: Full image height.
        tile_size: Size of each tile (square tiles).
        overlap: Overlap in pixels between adjacent tiles.

    Returns:
        List of (x, y, w, h) tuples for each tile.
    """
    stride = max(1, tile_size - overlap)
    tiles: list[tuple[int, int, int, int]] = []

    y = 0
    while y < image_height:
        x = 0
        while x < image_width:
            w = min(tile_size, image_width - x)
            h = min(tile_size, image_height - y)
            tiles.append((x, y, w, h))

            if x + tile_size >= image_width:
                break
            x += stride

        if y + tile_size >= image_height:
            break
        y += stride

    return tiles


def restore_tiled(
    image: Image.Image,
    restore_fn: Callable[[Image.Image], Image.Image],
    tile_size: int | None = None,
    overlap: int = 64,
    progress_fn: Callable[[int, int], None] | None = None,
) -> Image.Image:
    """Restore an image using overlapping tiles with Gaussian blending.

    Args:
        image: Input PIL image (RGB).
        restore_fn: Function that takes a PIL image tile and returns the
            restored PIL image tile (same size).
        tile_size: Tile size in pixels. Auto-selected if None.
        overlap: Overlap between adjacent tiles in pixels.
        progress_fn: Optional callback(current_tile, total_tiles) for progress.

    Returns:
        Restored PIL image with seamlessly blended tiles.
    """
    width, height = image.size

    if tile_size is None:
        tile_size = select_tile_size(width, height)

    # If image fits in one tile, skip tiling overhead
    if width <= tile_size and height <= tile_size:
        if progress_fn:
            progress_fn(1, 1)
        return restore_fn(image)

    tiles = compute_tiles(width, height, tile_size, overlap)
    total = len(tiles)

    # Accumulator buffers (float64 for precision during blending)
    result = np.zeros((height, width, 3), dtype=np.float64)
    weight_sum = np.zeros((height, width), dtype=np.float64)

    for idx, (tx, ty, tw, th) in enumerate(tiles):
        # Extract tile from input
        tile_img = image.crop((tx, ty, tx + tw, ty + th))

        # Restore tile
        restored_tile = restore_fn(tile_img)
        restored_arr = np.array(restored_tile, dtype=np.float64)

        # Build Gaussian weight for this tile's actual dimensions
        weight = _build_gaussian_weight(th, tw)

        # Accumulate weighted result
        for c in range(3):
            result[ty:ty + th, tx:tx + tw, c] += restored_arr[:, :, c] * weight
        weight_sum[ty:ty + th, tx:tx + tw] += weight

        if progress_fn:
            progress_fn(idx + 1, total)

    # Normalize by accumulated weights
    for c in range(3):
        result[:, :, c] /= weight_sum

    result = np.clip(result, 0, 255).astype(np.uint8)
    return Image.fromarray(result)


def restore_tiled_with_pipeline(
    image: Image.Image,
    pipe: Any,
    prompt: str,
    negative_prompt: str = "",
    steps: int = 28,
    guidance_scale: float = 3.0,
    seed: int = 42,
    size_level: int = 1024,
    tile_size: int | None = None,
    overlap: int = 64,
    progress_fn: Callable[[int, int], None] | None = None,
) -> Image.Image:
    """Convenience wrapper: tile-based restoration with a diffusers pipeline.

    Constructs the restore_fn from pipeline parameters and delegates
    to restore_tiled.

    Args:
        image: Input PIL image (RGB).
        pipe: A diffusers pipeline with __call__(image=..., prompt=..., ...).
        prompt: Text prompt for restoration.
        negative_prompt: Negative prompt.
        steps: Number of inference steps per tile.
        guidance_scale: Classifier-free guidance scale.
        seed: Random seed (same seed per tile for consistency).
        size_level: Model size level parameter.
        tile_size: Tile size in pixels. Auto-selected if None.
        overlap: Overlap between tiles in pixels.
        progress_fn: Optional progress callback.

    Returns:
        Restored PIL image.
    """
    def _restore_tile(tile: Image.Image) -> Image.Image:
        result = pipe(
            image=tile,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
            size_level=size_level,
        )
        return result.images[0]

    return restore_tiled(
        image=image,
        restore_fn=_restore_tile,
        tile_size=tile_size,
        overlap=overlap,
        progress_fn=progress_fn,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test tiling configuration for an image",
    )
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--tile-size", type=int, default=None,
                        help="Tile size (auto-selected if omitted)")
    parser.add_argument("--overlap", type=int, default=64,
                        help="Overlap in pixels between tiles")
    return parser.parse_args()


def main() -> None:
    """Print tiling plan as JSON for a given image."""
    args = parse_args()

    img = Image.open(args.image)
    width, height = img.size
    available_mem = get_available_memory_mb()

    tile_size = args.tile_size
    if tile_size is None:
        tile_size = select_tile_size(width, height, available_mem)

    tiles = compute_tiles(width, height, tile_size, args.overlap)

    result = {
        "image": args.image,
        "image_size": [width, height],
        "available_memory_mb": round(available_mem, 0),
        "tile_size": tile_size,
        "overlap": args.overlap,
        "num_tiles": len(tiles),
        "tiles": [
            {"x": x, "y": y, "w": w, "h": h}
            for x, y, w, h in tiles
        ],
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
