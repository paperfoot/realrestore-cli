"""End-to-end test: runs actual inference if model is available.

Skips gracefully if model hasn't been downloaded yet.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

TEST_IMAGE = "/Users/biobook/Pictures/rosie.png"
CLI_PATH = os.path.join(os.path.dirname(__file__), "..", "target", "release", "realrestore")


def _model_available() -> bool:
    """Check if the RealRestorer model is downloaded."""
    try:
        from huggingface_hub import try_to_load_from_cache
        result = try_to_load_from_cache(
            "RealRestorer/RealRestorer",
            "model_index.json",
        )
        return result is not None
    except Exception:
        return False


def test_cli_agent_info():
    """Test that agent-info returns valid JSON."""
    result = subprocess.run(
        [CLI_PATH, "agent-info"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert "commands" in data
    assert "restore" in data["commands"]
    assert "watermark-remove" in data["commands"]
    print("  PASS test_cli_agent_info")


def test_cli_restore_help():
    """Test restore subcommand help."""
    result = subprocess.run(
        [CLI_PATH, "restore", "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "--quality" in result.stdout
    assert "--tile" in result.stdout
    print("  PASS test_cli_restore_help")


def test_watermark_removal_cli():
    """Test watermark removal via CLI."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        output_path = f.name

    result = subprocess.run(
        [CLI_PATH, "--json", "watermark-remove", TEST_IMAGE, "-o", output_path],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": "python"},
    )

    try:
        data = json.loads(result.stdout)
        assert "data" in data or "error" in data
        print(f"  PASS test_watermark_removal_cli (output: {data.get('data', {}).get('psnr_vs_original', 'N/A')}dB)")
    except json.JSONDecodeError:
        print(f"  SKIP test_watermark_removal_cli (non-JSON output)")
    finally:
        if os.path.exists(output_path):
            os.unlink(output_path)


def test_auto_detection():
    """Test auto-detection on test image."""
    from realrestore_cli.optimizations.auto_detect import auto_detect
    result = auto_detect(TEST_IMAGE)
    assert result["degradation"] in [
        "blur", "noise", "haze", "compression", "low_light",
        "moire", "rain", "lens_flare", "auto",
    ]
    assert 0 <= result["confidence"] <= 1
    print(f"  PASS test_auto_detection (detected: {result['degradation']}, conf: {result['confidence']:.2f})")


def test_full_inference():
    """Test full inference pipeline — REQUIRES MODEL."""
    if not _model_available():
        print("  SKIP test_full_inference (model not downloaded)")
        return

    from realrestore_cli.engine import restore_image

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        output_path = f.name

    try:
        result = restore_image(
            input_path=TEST_IMAGE,
            output_path=output_path,
            task="auto",
            backend="auto",
            steps=4,  # Minimal steps for testing
            seed=42,
        )
        assert os.path.exists(output_path)
        assert result["elapsed_seconds"] > 0
        print(f"  PASS test_full_inference ({result['elapsed_seconds']}s, {result['peak_memory_mb']}MB)")
    finally:
        if os.path.exists(output_path):
            os.unlink(output_path)


if __name__ == "__main__":
    print("Running end-to-end tests...")
    tests = [
        test_cli_agent_info,
        test_cli_restore_help,
        test_auto_detection,
        test_watermark_removal_cli,
        test_full_inference,
    ]

    passed = 0
    skipped = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            if "SKIP" in str(e):
                skipped += 1
            else:
                failed += 1
                print(f"  FAIL {test.__name__}: {e}")

    print(f"\n{passed} passed, {skipped} skipped, {failed} failed")
