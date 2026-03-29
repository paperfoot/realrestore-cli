"""Automated benchmark runner for comparing backends and optimizations."""
from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any


def run_benchmarks(
    iterations: int = 3,
    backends: str = "auto",
    image_path: str | None = None,
) -> dict[str, Any]:
    """Run benchmarks across specified backends."""
    from realrestore_cli.engine import restore_image, get_device

    # Resolve backends
    if backends == "auto":
        available = [get_device()]
    else:
        available = [b.strip() for b in backends.split(",")]

    results = []

    for backend in available:
        times = []
        peak_mems = []

        for i in range(iterations):
            # Use a test image if none provided
            if image_path is None:
                # Create a simple test image
                from PIL import Image
                import tempfile
                test_img = Image.new("RGB", (512, 512), color=(128, 100, 80))
                tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                test_img.save(tmp.name)
                image_path = tmp.name

            import tempfile
            out_tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)

            try:
                result = restore_image(
                    input_path=image_path,
                    output_path=out_tmp.name,
                    backend=backend,
                    steps=10,  # Fewer steps for benchmarks
                )
                times.append(result["elapsed_seconds"])
                peak_mems.append(result["peak_memory_mb"])
            except Exception as e:
                times.append(-1)
                peak_mems.append(-1)

        valid_times = [t for t in times if t > 0]
        valid_mems = [m for m in peak_mems if m > 0]

        results.append({
            "backend": backend,
            "iterations": iterations,
            "avg_time": round(sum(valid_times) / len(valid_times), 2) if valid_times else -1,
            "min_time": round(min(valid_times), 2) if valid_times else -1,
            "max_time": round(max(valid_times), 2) if valid_times else -1,
            "peak_memory_mb": round(max(valid_mems), 1) if valid_mems else -1,
            "psnr": 0.0,  # Will be computed when reference images available
            "ssim": 0.0,
        })

    return {"results": results, "iterations": iterations}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--backends", default="auto")
    parser.add_argument("--image", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = run_benchmarks(
            iterations=args.iterations,
            backends=args.backends,
            image_path=args.image,
        )
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
