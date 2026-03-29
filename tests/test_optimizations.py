"""Tests for optimization modules — no model weights required."""
import json
import os
import sys
import tempfile

import numpy as np
from PIL import Image

# Add python dir to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))


def _create_test_image(size=(256, 256)):
    """Create a simple test image."""
    img = Image.new("RGB", size, color=(128, 100, 80))
    arr = np.array(img)
    # Add gradient + noise for realistic content
    for c in range(3):
        gradient = np.linspace(50, 200, size[0]).reshape(-1, 1) * np.ones((1, size[1]))
        arr[:, :, c] = np.clip(arr[:, :, c] + gradient * 0.3 + np.random.randint(-10, 10, arr.shape[:2]), 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


class TestAutoDetect:
    def test_detect_returns_dict(self):
        from realrestore_cli.optimizations.auto_detect import auto_detect
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            _create_test_image().save(f.name)
            result = auto_detect(f.name)
            assert isinstance(result, dict)
            assert "degradation" in result
            assert "task" in result
            assert "confidence" in result
            assert "scores" in result
            os.unlink(f.name)

    def test_all_detectors(self):
        from realrestore_cli.optimizations.auto_detect import (
            detect_blur, detect_noise, detect_haze,
            detect_compression, detect_low_light, detect_moire,
        )
        img = np.array(_create_test_image()).astype(np.float64)
        gray = np.mean(img, axis=2)

        for detector in [detect_blur, detect_noise, detect_haze,
                         detect_compression, detect_low_light, detect_moire]:
            score = detector(gray)
            assert 0.0 <= score <= 1.0, f"{detector.__name__} returned {score}"


class TestTiling:
    def test_compute_tiles(self):
        from realrestore_cli.optimizations.tiling import compute_tiles
        tiles = compute_tiles(1024, 1024, tile_size=256, overlap=32)
        assert len(tiles) > 0
        for (x, y, w, h) in tiles:
            assert w <= 256
            assert h <= 256

    def test_tiles_cover_image(self):
        from realrestore_cli.optimizations.tiling import compute_tiles
        width, height = 500, 300
        tiles = compute_tiles(width, height, tile_size=128, overlap=16)
        # Check all pixels are covered
        coverage = np.zeros((height, width))
        for (x, y, w, h) in tiles:
            coverage[y:y+h, x:x+w] = 1
        assert np.all(coverage > 0), "Not all pixels covered"


class TestScheduling:
    def test_quality_presets(self):
        from realrestore_cli.optimizations.scheduling import (
            QualityPreset, get_scheduler_config,
        )
        fast = get_scheduler_config(QualityPreset.FAST)
        balanced = get_scheduler_config(QualityPreset.BALANCED)
        high = get_scheduler_config(QualityPreset.HIGH)

        assert fast.num_steps < balanced.num_steps < high.num_steps
        assert fast.num_steps == 8
        assert balanced.num_steps == 14
        assert high.num_steps == 28


class TestWatermark:
    def test_spectral_detection(self):
        from realrestore_cli.watermark.detector import analyze_frequency_domain
        img = np.array(_create_test_image()).astype(np.float64)
        result = analyze_frequency_domain(img)
        assert "anomaly_score" in result
        assert "frequency_bands" in result

    def test_metadata_detection(self):
        from realrestore_cli.watermark.detector import detect_metadata_watermark
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            _create_test_image().save(f.name)
            result = detect_metadata_watermark(f.name)
            assert "has_watermark" in result
            os.unlink(f.name)

    def test_spectral_removal(self):
        from realrestore_cli.watermark.remover import remove_spectral
        img = np.array(_create_test_image())
        cleaned = remove_spectral(img, strength=0.5)
        assert cleaned.shape == img.shape
        assert cleaned.dtype == np.uint8

    def test_ensemble_removal(self):
        from realrestore_cli.watermark.remover import remove_watermark
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            _create_test_image().save(f.name)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as out:
                result = remove_watermark(f.name, out.name, method="ensemble")
                assert result["psnr_vs_original"] > 20  # Reasonable quality
                os.unlink(f.name)
                os.unlink(out.name)


class TestMPSBackend:
    def test_environment_config(self):
        from realrestore_cli.optimizations.mps_backend import configure_mps_environment
        env = configure_mps_environment()
        assert "PYTORCH_MPS_HIGH_WATERMARK_RATIO" in env
        assert env["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] == "0.0"

    def test_optimal_dtype(self):
        import torch
        from realrestore_cli.optimizations.mps_backend import get_optimal_dtype
        dtype = get_optimal_dtype()
        # Float16: float32 full model exceeds 64GB (78.6GB on MPS)
        assert dtype == torch.float16

    def test_hardware_info(self):
        from realrestore_cli.optimizations.mps_backend import get_apple_silicon_info
        info = get_apple_silicon_info()
        assert info["memory_gb"] > 0


class TestQuantize:
    def test_module_imports(self):
        from realrestore_cli.optimizations.quantize import quantize_pipeline
        assert callable(quantize_pipeline)


if __name__ == "__main__":
    import traceback

    classes = [TestAutoDetect, TestTiling, TestScheduling, TestWatermark, TestMPSBackend, TestQuantize]
    total = 0
    passed = 0
    failed = []

    for cls in classes:
        instance = cls()
        for name in dir(instance):
            if name.startswith("test_"):
                total += 1
                try:
                    getattr(instance, name)()
                    passed += 1
                    print(f"  PASS {cls.__name__}.{name}")
                except Exception as e:
                    failed.append(f"{cls.__name__}.{name}")
                    print(f"  FAIL {cls.__name__}.{name}: {e}")

    print(f"\n{passed}/{total} tests passed")
    if failed:
        print(f"Failed: {', '.join(failed)}")
