# RealRestorer on Apple Silicon (M4 Max 64GB): Optimization Deep Dive

This document addresses the critical challenges of running the 39GB RealRestorer diffusion model (Qwen2.5-VL + Transformer + VAE) on a 64GB Apple Silicon unified memory system, avoiding Out-Of-Memory (OOM) errors and maximizing throughput.

## 1. The EXACT Fix for MPSNDArrayMatrixMultiplication Dtype Assertion
The crash you are experiencing (`MPSNDArrayMatrixMultiplication.mm:644: failed assertion 'LORADOWN GEMV Kernel... will overflow'`) is a known bug in the Apple MPS backend on M4 chips when performing half-precision (`float16`) matrix multiplications where the matrix has specific column strides or paddings.

**The Solution:**
You do not need to fall back to `float32` (which causes OOM). Instead, force PyTorch to use the alternative, stable Metal-based matrix multiplication kernel by setting this environment variable before execution:
```bash
export PYTORCH_MPS_PREFER_METAL=1
```
Alternatively, in Python before importing torch:
```python
import os
os.environ["PYTORCH_MPS_PREFER_METAL"] = "1"
```
*Note: Using `bfloat16` (natively supported on M4 chips) combined with this flag provides the best stability and memory efficiency.*

## 2. Mixed Dtype Loading in Diffusers
Yes, `diffusers` pipelines support mixed precision across different components. You can initialize the pipeline with `low_cpu_mem_usage=True` to prevent memory spikes, load it in a base dtype, and selectively cast specific components.

```python
import torch
from diffusers import DiffusionPipeline

# Load pipeline with base fp16/bf16 to save memory
pipe = DiffusionPipeline.from_pretrained(
    "path/to/realrestorer",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    safetensors=True
)

# Cast text encoder to fp32 if strictly needed
pipe.text_encoder = pipe.text_encoder.to(dtype=torch.float32)

# Move to MPS unified memory
pipe.to("mps")
```
*Warning:* Upcasting the 15.4GB Qwen2.5-VL to `float32` requires 30.8GB, severely restricting the memory available for the 23.2GB transformer and KV cache. Mixed `bfloat16` + MLX quantization is a better approach.

## 3. Google TurboQuant
**Google TurboQuant** (ICLR 2026) is a breakthrough data-oblivious compression algorithm that compresses Key-Value (KV) cache memory by at least 6x (down to 3 bits) with zero measurable loss in accuracy and up to 8x faster attention computation. 

**Relevance here:** TurboQuant has already been ported to Apple Silicon via the `turboquant_mlx` and `turboquant_plus` libraries. For RealRestorer's massive 23.2GB transformer, integrating TurboQuant will drastically shrink the KV cache footprint during generation (which scales quadratically with image resolution), freeing up unified memory and significantly mitigating bandwidth bottlenecks on the M4 Max.

## 4. Latest torchao MPS-compatible Quantization
`torchao` now features highly optimized Metal kernels for Apple Silicon via the MPS backend, enabling weight-only quantization to solve the memory crisis.

For the Qwen2.5-VL text encoder, use **Int4 Weight-Only Quantization**:
```python
import torch
from torchao.quantization import quantize_, int4_weight_only

# Reduces the 15.4GB text encoder to ~3.8GB
quantize_(pipe.text_encoder, int4_weight_only(group_size=32))
```
This reduces the text encoder size by ~75% while maintaining text interpretation capabilities. `int8_weight_only()` is also available if you observe any unacceptable degradation in the generated prompt embeddings.

## 5. Fastest Inference with Maintained Quality
To achieve maximum throughput on the M4 Max (546 GB/s memory bandwidth) without degrading the restoration:

1. **Avoid CPU Offloading:** Do not use `enable_sequential_cpu_offload()`. Unified memory means the CPU and GPU share the same pool; explicit offloading just burns PCIe bandwidth and causes massive latency spikes.
2. **Hybrid MLX Backend:** Use the MLX community's 4-bit quantized Qwen2.5-VL (`mlx-community/Qwen2.5-VL-7B-Instruct-4bit`) via the `mlx-vlm` package for the text encoder (drops memory to ~3.5GB). Keep PyTorch+MPS for the DiT transformer and VAE.
3. **Use `bfloat16` natively:** Standardize on `torch.bfloat16`, which the M4 natively accelerates, rather than `float16`.
4. **Enable SDPA:** Ensure PyTorch's `scaled_dot_product_attention` is utilized (it is automatically enabled in PyTorch 2.x but verify it isn't overridden).
5. **VAE Tiling:** Enable `pipe.enable_vae_tiling()` only if rendering images > 1024px, to keep the VAE's memory profile strictly bounded.

## 6. Innovative Approaches
*   **DeepCache:** Highly recommended. DeepCache skips computing the deep layers of the U-Net/Transformer in certain diffusion steps by caching and reusing feature maps from previous steps. It provides a ~1.5x speedup with almost zero visual degradation and integrates easily into standard diffusers pipelines.
*   **ToMe (Token Merging):** Ideal for the 23.2GB transformer. ToMe merges redundant tokens in the attention layers, drastically cutting down the computational overhead and memory usage per step. It is highly effective for DiT architectures like RealRestorer.
*   **Step Reduction:** Instead of standard DDIM 50-steps, use advanced solvers like `DPMSolverMultistepScheduler` (DPM++ 2M Karras), which can reach convergence in 20-25 steps. Alternatively, investigate if LCM (Latent Consistency Model) or Hyper-SD LoRAs exist or can be trained for RealRestorer, which would drop the required steps to 4-8.
