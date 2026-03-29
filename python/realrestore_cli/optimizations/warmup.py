"""Pipeline warm-up for eliminating first-inference latency.

MPS Metal shader JIT compilation creates a significant first-run penalty.
This module provides background warm-up that forces shader compilation
before the user's actual image is processed.
"""
from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np
import torch
from PIL import Image


def ghost_inference(pipe: Any, device: str = "mps") -> float:
    """Run a minimal inference pass to trigger Metal shader compilation.

    Uses a tiny 64x64 image with 1 step — enough to compile all kernels
    without meaningful compute time.

    Returns elapsed time in seconds.
    """
    start = time.time()

    # Create minimal test image
    tiny_img = Image.fromarray(
        np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    )

    try:
        with torch.no_grad():
            _ = pipe(
                image=tiny_img,
                prompt="test",
                negative_prompt="",
                num_inference_steps=1,
                guidance_scale=1.0,
                seed=0,
                size_level=64,
            )
    except Exception:
        # Some pipelines may not support 64px — try 128
        try:
            tiny_img = Image.fromarray(
                np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
            )
            _ = pipe(
                image=tiny_img,
                prompt="test",
                negative_prompt="",
                num_inference_steps=1,
                guidance_scale=1.0,
                seed=0,
                size_level=128,
            )
        except Exception:
            pass

    # Synchronize MPS
    if device == "mps" and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()

    elapsed = time.time() - start
    return elapsed


def warmup_async(pipe: Any, device: str = "mps") -> threading.Thread:
    """Run ghost inference in background thread.

    Returns the thread so caller can optionally join() before
    the real inference starts.
    """
    thread = threading.Thread(
        target=ghost_inference,
        args=(pipe, device),
        daemon=True,
    )
    thread.start()
    return thread


def precompute_prompt_embeddings(
    pipe: Any,
    prompts: list[str],
    device: str = "mps",
) -> dict[str, Any]:
    """Pre-encode task prompts to avoid text encoder latency during inference.

    Stores encoded prompts in a dict for reuse across multiple
    restoration calls.
    """
    embeddings = {}

    for prompt in prompts:
        try:
            # Access the text encoder directly if available
            if hasattr(pipe, "encode_prompt"):
                with torch.no_grad():
                    emb = pipe.encode_prompt(prompt, device=device)
                    embeddings[prompt] = emb
        except Exception:
            pass

    return embeddings
