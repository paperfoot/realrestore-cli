# Running 30-40GB Diffusion Models on 64GB Apple Silicon

Research conducted: 2026-03-29
Target: RealRestorer (39GB) on 64GB Apple Silicon (M4 Pro/Max)

---

## Table of Contents

1. [The Core Problem](#1-the-core-problem)
2. [Apple Silicon Unified Memory Architecture](#2-apple-silicon-unified-memory-architecture)
3. [Offloading Strategies Compared](#3-offloading-strategies-compared)
4. [Group Offloading (Newest, Most Promising)](#4-group-offloading-newest-most-promising)
5. [Quantization Backends for Diffusers](#5-quantization-backends-for-diffusers)
6. [Text Encoder Deletion Pattern](#6-text-encoder-deletion-pattern)
7. [Sharded Checkpoint Loading](#7-sharded-checkpoint-loading)
8. [Mixed Precision on MPS](#8-mixed-precision-on-mps)
9. [ComfyUI Memory Management Approach](#9-comfyui-memory-management-approach)
10. [MLX as Alternative Runtime](#10-mlx-as-alternative-runtime)
11. [GGUF Quantization via stable-diffusion.cpp](#11-gguf-quantization-via-stable-diffusioncpp)
12. [DFloat11 Lossless Compression](#12-dfloat11-lossless-compression)
13. [Layerwise Casting](#13-layerwise-casting)
14. [Meta Tensor Error Fix](#14-meta-tensor-error-fix)
15. [MPS-Specific Gotchas and Limitations](#15-mps-specific-gotchas-and-limitations)
16. [Memory Budget Analysis for RealRestorer (39GB)](#16-memory-budget-analysis-for-realrestorer-39gb)
17. [Recommended Strategy for RealRestorer](#17-recommended-strategy-for-realrestorer)

---

## 1. The Core Problem

RealRestorer is 39GB in full precision. A 64GB Apple Silicon Mac has ~48GB usable for GPU
(the remaining ~16GB is reserved by macOS and system processes). Even the full 64GB of unified
memory is not enough to hold the model weights (39GB) plus activations, intermediate tensors,
and the OS simultaneously without memory pressure.

Key constraint: On Apple Silicon, CPU and GPU share the same physical RAM. Unlike discrete
GPU systems where you can offload to CPU RAM as a separate pool, on Apple Silicon offloading
to CPU does NOT free GPU-accessible memory -- it is the same pool. This fundamentally changes
the offloading calculus.

Sources:
- [Flux.1 + ComfyUI Mac Memory Bottleneck](https://macgpu.com/en/blog/flux1-comfyui-mac-memory-bottleneck-64gb-unified-memory.html)
- [Reduce memory usage - Diffusers](https://huggingface.co/docs/diffusers/en/optimization/memory)

---

## 2. Apple Silicon Unified Memory Architecture

**Key facts:**
- 64GB Mac = ~48GB usable for GPU workloads (75% rule of thumb)
- CPU and GPU share the same physical RAM -- no separate VRAM pool
- Memory bandwidth: 273 GB/s on M4 Max, lower on M4 Pro
- No data copy needed between CPU and GPU (zero-copy in theory)
- macOS will compress and swap to SSD when memory pressure rises
- Swap to SSD causes 8-10x slowdown (760s vs 75s for Flux generation)

**Implication for offloading:**
Traditional CPU offloading (moving weights from GPU to CPU RAM) has reduced benefit on
Apple Silicon because both share the same physical memory. The benefit is only from the
framework releasing GPU-side buffer allocations, not from physically moving data to a
separate memory pool. Group offloading and quantization are therefore MORE important on
Apple Silicon than on discrete GPU systems.

Sources:
- [Flux.1 + ComfyUI Mac Memory Bottleneck](https://macgpu.com/en/blog/flux1-comfyui-mac-memory-bottleneck-64gb-unified-memory.html)
- [Apple Silicon Limitations](https://stencel.io/posts/apple-silicon-limitations-with-usage-on-local-llm%20.html)

---

## 3. Offloading Strategies Compared

### 3a. `pipe.to("mps")` -- Full Device Placement

Places ALL model components on MPS at once. Requires enough memory for the entire model
plus activations.

```python
pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("mps")
```

**Memory:** Highest (entire model resident)
**Speed:** Fastest (no transfer overhead)
**RealRestorer feasibility:** NOT feasible -- 39GB model + activations > 48GB usable

### 3b. `enable_model_cpu_offload()` -- Whole-Model Offloading

Moves entire pipeline components (text encoder, denoiser, VAE) between CPU and GPU.
Only one component on GPU at a time.

```python
pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()
```

**Memory:** Moderate savings -- peak = largest single component + activations
**Speed:** Faster than sequential (fewer transfers)
**SDXL benchmark:** 20.21 GB peak, 16s inference
**Limitation on Apple Silicon:** Limited benefit because CPU and GPU share memory.
The model is still resident in unified memory even when "on CPU."

### 3c. `enable_sequential_cpu_offload()` -- Layer-by-Layer Offloading

Offloads individual submodules (leaf parameters) to CPU, loads only the currently
executing leaf to GPU.

```python
pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
pipe.enable_sequential_cpu_offload()
```

**Memory:** Lowest GPU allocation (only leaf module at a time)
**Speed:** EXTREMELY SLOW (3-6x slower). SDXL: 67s vs 11s baseline.
**WARNING:** Do NOT call `.to("mps")` before calling this -- it negates savings.
**Meta tensor bug:** Can trigger "Cannot copy out of meta tensor; no data!" error
(see Section 14).

### 3d. Comparison Table (SDXL Benchmarks from HuggingFace Blog)

| Strategy | Memory (GB) | Latency (ms) | Notes |
|---|---|---|---|
| FP16 + SDPA baseline | 21.72 | 11,413 | Full GPU placement |
| + model CPU offload | 20.21 | 16,082 | Moderate savings |
| + sequential CPU offload | 19.91 | 67,034 | 6x slower |
| + VAE slicing | 15.40 | 11,232 | Big win for batches |
| + VAE slicing + seq. offload | 11.47 | 66,869 | Maximum savings |

Sources:
- [Simple SDXL Optimizations](https://github.com/huggingface/blog/blob/main/simple_sdxl_optimizations.md)
- [Reduce memory usage - Diffusers](https://huggingface.co/docs/diffusers/en/optimization/memory)
- [Sequential CPU offload 3x slowdown](https://github.com/huggingface/diffusers/issues/2266)

---

## 4. Group Offloading (Newest, Most Promising)

Group offloading is a NEW feature in diffusers (added via PR #10503 and #10516) that sits
between model offloading and sequential offloading in both memory and speed.

### How It Works

Moves groups of internal layers (torch.nn.ModuleList or torch.nn.Sequential) between
CPU and GPU as a unit, rather than individual parameters (sequential) or entire models
(model offload).

### Two Modes

**block_level**: Offloads groups of N layers at a time.
```python
pipeline.transformer.enable_group_offload(
    onload_device=torch.device("mps"),
    offload_device=torch.device("cpu"),
    offload_type="block_level",
    num_blocks_per_group=2  # Load 2 transformer blocks at a time
)
```
For a model with 40 layers and num_blocks_per_group=2: only 2 layers resident at a time
(20 total onload/offload cycles). Drastically reduces memory.

**leaf_level**: Offloads individual layers but with stream prefetching.
```python
pipeline.transformer.enable_group_offload(
    onload_device=torch.device("mps"),
    offload_device=torch.device("cpu"),
    offload_type="leaf_level",
    use_stream=True  # CUDA only -- does NOT work on MPS
)
```

### CUDA Stream Prefetching (NOT available on MPS)

The `use_stream=True` option overlaps data transfer with computation by prefetching
the next layer while the current one executes. This is CUDA-only and does NOT work on MPS.
On MPS, group offloading still works but without stream overlap, making it slower than
on CUDA.

### Disk Offloading

For systems with insufficient RAM, group offloading can spill to disk:
```python
pipeline.transformer.enable_group_offload(
    onload_device=torch.device("mps"),
    offload_device=torch.device("cpu"),
    offload_type="leaf_level",
    offload_to_disk_path="/path/to/ssd"
)
```

### Mixed Offloading Per Component

Different offloading for different components:
```python
# Heavy transformer: aggressive block-level offloading
pipeline.transformer.enable_group_offload(
    onload_device=onload_device,
    offload_device=offload_device,
    offload_type="block_level",
    num_blocks_per_group=1
)

# Lighter VAE: leaf-level
pipeline.vae.enable_group_offload(
    onload_device=onload_device,
    offload_type="leaf_level"
)

# Text encoder: block-level with larger groups
apply_group_offloading(
    pipeline.text_encoder,
    onload_device=onload_device,
    offload_type="block_level",
    num_blocks_per_group=4
)
```

### Apple Silicon Consideration

On Apple Silicon, since CPU and GPU share unified memory, the physical data is NOT
actually moving -- group offloading's benefit comes from the framework releasing
GPU buffer allocations for the offloaded blocks, allowing macOS to reuse that memory.
The speed benefit of fewer synchronization points compared to sequential offloading
still applies.

Sources:
- [Module Group Offloading PR #10503](https://github.com/huggingface/diffusers/pull/10503)
- [Group offloading with CUDA stream prefetching PR #10516](https://github.com/huggingface/diffusers/pull/10516)
- [Reduce memory usage - Diffusers](https://huggingface.co/docs/diffusers/en/optimization/memory)

---

## 5. Quantization Backends for Diffusers

Quantization is the SINGLE MOST IMPACTFUL technique for running large models on memory-
constrained hardware. For a 39GB model, even 8-bit quantization cuts it to ~19.5GB.

### Memory Savings Benchmarks (Flux-dev, 31.4GB BF16 baseline, H100)

| Backend | Precision | Loaded (GB) | Peak (GB) | Time | Quality |
|---|---|---|---|---|---|
| Baseline BF16 | 16-bit | 31.4 | 36.2 | 12s | Perfect |
| bitsandbytes | NF4 | 12.6 | 17.3 | 12s | Very Good |
| bitsandbytes | INT8 | 19.3 | 24.4 | 27s | Excellent |
| torchao | int4_weight_only | 10.6 | 14.7 | 109s | Noticeable |
| torchao | int8_weight_only | 17.0 | 21.5 | 15s | Very Good |
| torchao | float8_weight_only | 17.0 | 21.5 | 15s | Very Good |
| Quanto | INT4 | 12.3 | 16.1 | 109s | Noticeable |
| Quanto | INT8 | 17.3 | 21.8 | 15s | Very Good |
| Quanto | FP8 | 16.4 | 20.9 | 16s | Excellent |
| GGUF | Q2_k | 13.3 | 17.8 | 26s | Noticeable |
| GGUF | Q4_1 | 16.8 | 21.3 | 23s | Very Good |
| GGUF | Q8_0 | 21.5 | 26.0 | 15s | Excellent |

### Apple Silicon / MPS Compatibility

| Backend | MPS Support | Notes |
|---|---|---|
| bitsandbytes | In Progress | MPS backend via mps-bitsandbytes package (PR #1853). Adds INT8 and NF4/FP4 support for Metal. Expected Q4/Q1 2025/2026. |
| torchao | Partial | Only int4_weight_only works on Metal. Other modes not supported. |
| **Quanto** | **YES** | **Recommended for Apple Silicon.** Supports INT4, INT8, FP8 on MPS and CPU. |
| GGUF | YES | Works via stable-diffusion.cpp with Metal acceleration. |
| FP8 Layerwise | NO | Requires NVIDIA Hopper/Ada GPU hardware. |

### Quanto Code Example (Apple Silicon compatible)

```python
import torch
from diffusers import FluxPipeline
from diffusers import QuantoConfig as DiffusersQuantoConfig
from diffusers.quantizers import PipelineQuantizationConfig
from transformers import QuantoConfig as TransformersQuantoConfig

pipeline_quant_config = PipelineQuantizationConfig(
    quant_mapping={
        "transformer": DiffusersQuantoConfig(weights_dtype="int8"),
        "text_encoder_2": TransformersQuantoConfig(weights_dtype="int8"),
    }
)

pipe = FluxPipeline.from_pretrained(
    model_id,
    quantization_config=pipeline_quant_config,
    torch_dtype=torch.float16
)
pipe.to("mps")
```

### Quanto Direct Quantization (More Control)

```python
from optimum.quanto import freeze, qfloat8, qint8, quantize

# Quantize transformer to INT8
transformer = AutoModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=torch.float16)
quantize(transformer, weights=qint8)
freeze(transformer)

# Quantize text encoder to INT8
text_encoder = T5EncoderModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=torch.float16)
quantize(text_encoder, weights=qint8)
freeze(text_encoder)
```

### Combined: Quantization + Offloading (Maximum Savings)

```python
# BnB 4-bit + model CPU offload = 12.4GB peak
pipe = FluxPipeline.from_pretrained(model_id, quantization_config=config, torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()

# FP8 layerwise + group offloading = 14.2GB peak
transformer.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)
transformer.enable_group_offload(onload_device=device, offload_device=cpu, offload_type="leaf_level", use_stream=True)
```

Sources:
- [Exploring Quantization Backends in Diffusers](https://huggingface.co/blog/diffusers-quantization)
- [Diffusers Quantization API](https://huggingface.co/docs/diffusers/en/api/quantization)
- [bitsandbytes MPS backend PR #1853](https://github.com/bitsandbytes-foundation/bitsandbytes/pull/1853)

---

## 6. Text Encoder Deletion Pattern

For multi-component pipelines, you can precompute text embeddings then delete the text
encoder(s) from memory before running the denoiser. This frees significant memory.

### Pattern

```python
import gc
import torch

# Step 1: Load pipeline
pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

# Step 2: Compute embeddings (text encoder runs here)
prompt_embeds, negative_prompt_embeds, pooled_embeds, neg_pooled = pipe.encode_prompt(
    prompt="your prompt",
    device="mps"
)

# Step 3: Delete text encoders to free memory
del pipe.text_encoder, pipe.text_encoder_2, pipe.tokenizer, pipe.tokenizer_2
gc.collect()
torch.mps.empty_cache()  # MPS equivalent of torch.cuda.empty_cache()

# Step 4: Run inference with precomputed embeddings
image = pipe(
    prompt_embeds=prompt_embeds,
    negative_prompt_embeds=negative_prompt_embeds,
    pooled_prompt_embeds=pooled_embeds,
    negative_pooled_prompt_embeds=neg_pooled,
).images[0]
```

### Alternative: Load Without Text Encoders

```python
pipe = StableDiffusionXLPipeline.from_pretrained(
    model_id,
    text_encoder=None,
    text_encoder_2=None,
    tokenizer=None,
    tokenizer_2=None,
    torch_dtype=torch.float16
).to("mps")

# Pass precomputed embeddings
image = pipe(prompt_embeds=precomputed_embeds, ...).images[0]
```

### Memory Impact

For SDXL, the two text encoders (CLIP-L + CLIP-G) use ~1.5GB total.
For SD3/Flux, T5-XXL alone uses ~9.9GB in BF16 (or ~2.9GB quantized Q4).
For RealRestorer, the savings depend on its text encoder architecture.

Sources:
- [Simple SDXL Optimizations](https://github.com/huggingface/blog/blob/main/simple_sdxl_optimizations.md)
- [SDXL Training Guide](https://huggingface.co/docs/diffusers/en/training/sdxl)

---

## 7. Sharded Checkpoint Loading

For models stored as a single large file, loading requires 2x the model size in RAM
(file read buffer + model instantiation). Sharding reduces peak load-time memory.

### Creating Sharded Checkpoints

```python
from diffusers import AutoModel

model = AutoModel.from_pretrained(model_id, subfolder="unet")
model.save_pretrained("model-sharded", max_shard_size="5GB")
# Creates: model-sharded/model-00001.safetensors, model-00002.safetensors, etc.
```

### Loading Sharded Checkpoints

```python
model = AutoModel.from_pretrained("model-sharded", torch_dtype=torch.float16)
```

Shards are loaded one at a time, so peak memory during loading is:
`model_size + largest_shard_size` instead of `2 * model_size`.

### Recommendation

Shard when the fp32 checkpoint exceeds 5GB. Default shard size is 5GB.
For RealRestorer at 39GB, sharding into ~5GB pieces would create ~8 shards and
reduce peak load-time memory from ~78GB to ~44GB.

### low_cpu_mem_usage Loading

```python
model = AutoModel.from_pretrained(
    model_id,
    low_cpu_mem_usage=True,  # Default True since PyTorch >= 1.9
    torch_dtype=torch.float16
)
```

This avoids initializing weights before loading pretrained values, keeping peak
memory at ~1x model size instead of ~2x during loading. Enabled by default in
modern diffusers.

Sources:
- [Inference with Big Models](https://huggingface.co/docs/diffusers/tutorials/inference_with_big_models)
- [Reduce memory usage - Diffusers](https://huggingface.co/docs/diffusers/en/optimization/memory)

---

## 8. Mixed Precision on MPS

Apple Silicon supports mixed precision inference through `torch.autocast` with the MPS
backend, but with important limitations.

### Using Autocast on MPS

```python
import torch

with torch.autocast(device_type="mps", dtype=torch.float16):
    output = model(input_tensor)
```

Both float16 and bfloat16 are supported as autocast dtypes on MPS (added in PyTorch 2.1+).

### Loading Models in Float16

```python
pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("mps")
```

This halves memory from fp32 and is generally the baseline for all MPS inference.

### Known Issues

1. **No native FP8 support:** Apple Silicon has no hardware FP8 units. FP8 operations
   are emulated via upcasting to BF16, providing no actual memory benefit during compute.
2. **NDArray size limit:** MPS backend does not support NDArray sizes > 2^32.
3. **Some ops fallback to CPU:** Set `PYTORCH_ENABLE_MPS_FALLBACK=1` to handle unsupported ops.
4. **MPS dtype assertion errors:** Some operations fail when tensors have mismatched dtypes.
   Use float32 for those specific operations (we already fixed this in commit 2641227).

### Practical Recommendation

Use `torch.float16` as the standard dtype for MPS inference. Do NOT use bfloat16 unless
specifically needed -- float16 has better hardware support on Apple Silicon.

Sources:
- [PyTorch AMP Documentation](https://docs.pytorch.org/docs/stable/amp.html)
- [Enable AMP for MPS devices - PyTorch Issue #88415](https://github.com/pytorch/pytorch/issues/88415)
- [WWDC23: Optimize ML for Metal](https://developer.apple.com/videos/play/wwdc2023/10050/)
- [State of PyTorch Hardware Acceleration 2025](https://tunguz.github.io/PyTorch_Hardware_2025/)

---

## 9. ComfyUI Memory Management Approach

ComfyUI has the most mature memory management for diffusion models on macOS. Its strategies
are instructive for our implementation.

### Key Techniques

1. **Model caching in memory pool:** Loaded models remain resident in unified memory across
   multiple generations. Eviction only happens when a new model is loaded or memory pressure
   forces it.

2. **Graph-based execution:** Each workflow node maps to Metal command buffers with automatic
   synchronization. The DAG structure allows sequential execution of heavy components.

3. **GGUF quantization:** On 64GB, GGUF Q8_0 is recommended as the best quality-to-speed
   ratio. BF16 full precision also fits comfortably.

4. **Critical launch flags:**
   - `PYTORCH_ENABLE_MPS_FALLBACK=1` (required)
   - `--fp32vae` flag to prevent black output from MPS VAE quantization bugs

### Flux Component Memory Breakdown (Relevant to RealRestorer)

| Component | Size (BF16) | Size (Q8_0) | Size (Q4_K_M) |
|---|---|---|---|
| Diffusion model | 23.8 GB | 11.9 GB | ~7 GB |
| T5-XXL text encoder | 9.9 GB | ~5 GB | 2.9 GB |
| CLIP-L encoder | 246 MB | 246 MB | 246 MB |
| VAE | 335 MB | 335 MB | 335 MB |
| **Peak inference** | **>34 GB** | **~20 GB** | **~12 GB** |

### Performance on 64GB Mac

| Config | Memory Pressure | Swap | Time (1024x1024) |
|---|---|---|---|
| GGUF Q4_K_S, 16GB Mac | RED | 4.6 GB | 760s (12.7 min) |
| GGUF Q4_K_S, 64GB Mac | GREEN | 0 GB | ~95s |
| BF16 full, 64GB Mac | GREEN | 0 GB | ~75s |

The 8x speed improvement from 16GB to 64GB comes entirely from eliminating SSD swap.

Sources:
- [Flux.1 + ComfyUI Mac Memory Bottleneck](https://macgpu.com/en/blog/flux1-comfyui-mac-memory-bottleneck-64gb-unified-memory.html)
- [ComfyUI MLX Extension](https://apatero.com/blog/comfyui-mlx-extension-70-faster-apple-silicon-guide-2025)
- [ComfyUI GitHub](https://github.com/Comfy-Org/ComfyUI)

---

## 10. MLX as Alternative Runtime

Apple's MLX framework is purpose-built for Apple Silicon and provides memory advantages
over PyTorch's MPS backend.

### Advantages Over PyTorch MPS

1. **True zero-copy:** MLX arrays live in shared memory natively. No buffer management
   overhead.
2. **Lazy evaluation:** Operations are fused and memory allocation is minimized.
3. **Native quantization:** Efficient dequantization kernels designed for Apple Silicon.
4. **50-70% faster than PyTorch MPS** for image generation (ComfyUI MLX extension benchmarks).

### Diffusion Model Support

- **MFLUX (MacFLUX):** Line-by-line port of HuggingFace Diffusers FLUX to MLX.
- **DiffusionKit:** Library for running diffusion models on Apple Silicon with Core ML and MLX.
- **MLX examples include Stable Diffusion.**

### Limitation for RealRestorer

MLX requires models to be converted from PyTorch format. If RealRestorer uses custom
architectures or non-standard layers, conversion may require significant engineering effort.
MLX does not support the full diffusers pipeline API, so integration would need custom code.

### Performance

FLUX-dev-4bit on M5 with MLX is 3.8x faster than on M4. MLX is the best-performing
runtime for Apple Silicon when a model can be converted.

Sources:
- [MLX GitHub](https://github.com/ml-explore/mlx)
- [WWDC25: Get started with MLX](https://developer.apple.com/videos/play/wwdc2025/315/)
- [Benchmarking On-Device ML on Apple Silicon with MLX](https://arxiv.org/html/2510.18921v1)
- [PyTorch and MLX for Apple Silicon](https://towardsdatascience.com/pytorch-and-mlx-for-apple-silicon-4f35b9f60e39/)

---

## 11. GGUF Quantization via stable-diffusion.cpp

stable-diffusion.cpp provides inference for diffusion models in pure C/C++ with GGUF
quantization support.

### Key Features

- Supports SD1.x, SD2.x, SDXL, SD3, SD3.5, FLUX, Wan, Qwen Image
- Quantization from 2-bit to 8-bit
- Metal acceleration on Apple Silicon
- VAE tiling for large images
- Parameter offloading between CPU and GPU
- Flash attention support
- Text encoders on CPU while diffusion model on GPU

### Memory Requirements

| Quantization | VRAM (512x512) | Quality |
|---|---|---|
| Q4_0 | 2 GB minimum | Acceptable |
| Q8_0 | 4 GB comfortable | Near-lossless |
| Full precision | Model size + activations | Perfect |

### Conversion Command

```bash
# Convert safetensors to GGUF with quantization
./bin/sd-convert --type q8_0 model.safetensors model-q8_0.gguf
```

### Relevance to RealRestorer

If RealRestorer can be decomposed into standard diffusion model components (text encoder,
denoiser, VAE), it could be converted to GGUF format. At Q8_0, a 39GB model becomes ~19.5GB.
At Q4_0, it becomes ~9.75GB. However, this requires the model architecture to be supported
by stable-diffusion.cpp.

Sources:
- [stable-diffusion.cpp GitHub](https://github.com/leejet/stable-diffusion.cpp)
- [stable-diffusion.cpp Quantization Docs](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/quantization_and_gguf.md)
- [Practical GGUF Quantization Guide for iPhone and Mac](https://enclaveai.app/blog/2025/11/12/practical-quantization-guide-iphone-mac-gguf/)

---

## 12. DFloat11 Lossless Compression

DFloat11 (NeurIPS 2025) achieves 30% lossless compression of diffusion models and LLMs.

### How It Works

- Exploits low entropy in BFloat16 weight representations
- Applies entropy coding (dynamic-length encodings based on frequency)
- Decompression via compact hierarchical lookup tables (LUTs) that fit in GPU SRAM
- Transformer-block-level decompression to minimize latency

### Results

- 30% size reduction with bit-for-bit identical outputs
- 2.3-46.2x higher throughput than CPU offloading
- Enables Llama 3.1 405B (810GB) on 8x80GB GPUs
- FLUX.1-Krea-dev: reduces from 17.5GB to 9.8GB peak GPU memory

### Apple Silicon Compatibility

DFloat11 currently targets CUDA GPUs. MPS/Metal support is NOT available.
However, the entropy coding concept could theoretically be implemented for Metal.

Sources:
- [DFloat11 GitHub](https://github.com/LeanModels/DFloat11)
- [DFloat11 Paper](https://arxiv.org/abs/2504.11651)
- [DFloat11 FLUX example](https://github.com/LeanModels/DFloat11/tree/master/examples/flux.1)

---

## 13. Layerwise Casting

Stores weights in FP8 (float8_e4m3fn or float8_e5m2) and upcasts to float16/bfloat16
only during computation. This reduces memory for weight storage without losing compute
precision.

```python
from diffusers import AutoModel

transformer = AutoModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=torch.bfloat16)
transformer.enable_layerwise_casting(
    storage_dtype=torch.float8_e4m3fn,
    compute_dtype=torch.bfloat16
)
```

### Memory Impact

Flux-dev transformer: 23.7GB in BF16 -> ~11.85GB in FP8 storage (compute still in BF16).

### Apple Silicon Limitation

FP8 storage types (float8_e4m3fn, float8_e5m2) may not have native support on Apple Silicon.
The upcasting to float16/bfloat16 happens during compute, so the compute path works, but
storage of FP8 tensors may fall back to CPU emulation. This needs testing.

### Combining with Group Offloading

```python
transformer.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)
transformer.enable_group_offload(
    onload_device=torch.device("mps"),
    offload_device=torch.device("cpu"),
    offload_type="leaf_level"
)
# H100 result: 9.3GB loaded, 14.2GB peak
```

Sources:
- [Reduce memory usage - Diffusers](https://huggingface.co/docs/diffusers/en/optimization/memory)
- [Exploring Quantization Backends in Diffusers](https://huggingface.co/blog/diffusers-quantization)

---

## 14. Meta Tensor Error Fix

### The Problem

When loading pipelines with separately instantiated components (e.g., loading a T5 encoder
separately and passing it to the pipeline), calling `enable_sequential_cpu_offload()` or
`enable_model_cpu_offload()` triggers:

```
NotImplementedError: Cannot copy out of meta tensor; no data!
```

### Root Cause

The pipeline's `from_single_file()` loader uses an empty weights context manager that creates
"meta tensors" (placeholder tensors without data) for uninitialized components. When a
pre-loaded component coexists with meta tensors, the offloading hooks try to copy meta
tensors to CPU, which fails.

### Workarounds

**Workaround 1: Disable auto-offload context**
```python
import contextlib
import diffusers
diffusers.loaders.single_file_utils.init_empty_weights = contextlib.nullcontext
```

**Workaround 2: Load all components explicitly**
Instead of using `from_single_file()`, load each component separately with `from_pretrained()`:
```python
from diffusers import AutoModel
transformer = AutoModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=torch.float16)
vae = AutoModel.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float16)
# Then construct pipeline with all explicit components
```

**Workaround 3: Use group offloading instead**
Group offloading applies to individual model components AFTER they are loaded, avoiding
the meta tensor issue in pipeline-level offloading:
```python
# Instead of pipeline-level offloading:
# pipe.enable_sequential_cpu_offload()  # This fails with meta tensors

# Use component-level group offloading:
pipe.transformer.enable_group_offload(
    onload_device=torch.device("mps"),
    offload_device=torch.device("cpu"),
    offload_type="block_level",
    num_blocks_per_group=2
)
```

### Status

Fixed in diffusers 0.29.1+. If using older versions, apply workaround 1.

Sources:
- [Meta tensor error issue #8644](https://github.com/huggingface/diffusers/issues/8644)
- [Meta tensor error issue #2531](https://github.com/huggingface/diffusers/issues/2531)

---

## 15. MPS-Specific Gotchas and Limitations

### Attention Slicing

- Recommended for machines with < 64GB RAM
- Improves performance by ~20% on machines without universal memory
- On 64GB+ machines: may actually HURT performance
- Do NOT combine with SDPA or xFormers (causes serious slowdowns)

```python
# Only use if < 64GB RAM:
pipe.enable_attention_slicing()

# NEVER use if PyTorch >= 2.0 (SDPA is already active):
# pipe.enable_attention_slicing()  # DON'T
```

### PYTORCH_ENABLE_MPS_FALLBACK

Required environment variable for MPS:
```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```
Tells PyTorch to fall back to CPU for operations not yet implemented in MPS.

### VAE FP32 Requirement

The VAE on MPS often needs to run in FP32 to avoid black/corrupted output:
```python
pipe.vae = pipe.vae.to(dtype=torch.float32)
# Or use the --fp32vae flag in ComfyUI
```

### Batch Inference

Generating multiple prompts in a batch can crash on MPS. Use iteration instead:
```python
# BAD on MPS:
images = pipe(["prompt1", "prompt2", "prompt3"]).images

# GOOD on MPS:
images = [pipe(prompt).images[0] for prompt in ["prompt1", "prompt2", "prompt3"]]
```

### Warmup Pass

For PyTorch 1.13 (legacy), a dummy warmup pass is required:
```python
_ = pipe("warmup", num_inference_steps=1)
# Then run actual inference
```

### NDArray Size Limit

MPS does not support NDArray sizes > 2^32. This limits resolution and batch sizes.

### torch.compile on MPS

torch.compile has limited benefit on MPS. Focus on correctness in eager mode.
Performance tuning with compile is better suited to CUDA/cloud.

Sources:
- [MPS Optimization - Diffusers](https://huggingface.co/docs/diffusers/en/optimization/mps)
- [torch.compile and Diffusers](https://pytorch.org/blog/torch-compile-and-diffusers-a-hands-on-guide-to-peak-performance/)

---

## 16. Memory Budget Analysis for RealRestorer (39GB)

### The Math

```
Available:             64 GB total unified memory
macOS + apps:         -16 GB (conservative estimate)
Usable for inference:  48 GB

RealRestorer (FP16):   19.5 GB (39GB / 2)
RealRestorer (INT8):    9.75 GB (39GB / 4)
RealRestorer (INT4):    4.875 GB (39GB / 8)
```

### Scenario Analysis

| Strategy | Model Size | Activations | Total | Fits 48GB? |
|---|---|---|---|---|
| FP32 full | 39 GB | ~10 GB | ~49 GB | NO (barely) |
| FP16 full | 19.5 GB | ~5 GB | ~24.5 GB | YES |
| FP16 + model offload | ~19.5 GB total but peak = largest component | ~5 GB | ~15-20 GB | YES |
| INT8 (Quanto) | 9.75 GB | ~5 GB | ~14.75 GB | YES, comfortable |
| INT8 + group offload | Peak: 2-3 GB | ~5 GB | ~7-8 GB | YES, very comfortable |
| INT4 (Quanto) | 4.875 GB | ~5 GB | ~9.875 GB | YES, very comfortable |
| INT4 + group offload | Peak: 1-2 GB | ~5 GB | ~6-7 GB | YES, minimal footprint |

### Recommended Memory Targets

- **Green zone (no pressure):** < 40 GB total usage
- **Yellow zone (some pressure):** 40-50 GB
- **Red zone (swapping):** > 50 GB -- performance degrades 8-10x

---

## 17. Recommended Strategy for RealRestorer

Based on all research, the following is the recommended approach, ordered by
implementation priority:

### Priority 1: FP16 Loading (Immediate, No Quality Loss)

```python
pipe = RealRestorerPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
```
Cuts model from 39GB to 19.5GB. This alone may be sufficient for 64GB Mac.

### Priority 2: Quanto INT8 Quantization (Best Quality/Size Ratio for MPS)

```python
from optimum.quanto import freeze, qint8, quantize

# Quantize the heavy components
quantize(pipe.denoiser, weights=qint8)
freeze(pipe.denoiser)
```
Cuts model to ~9.75GB. Quanto is the ONLY mature quantization backend with MPS support.
INT8 provides near-imperceptible quality loss.

### Priority 3: Text Encoder Deletion After Embedding

```python
embeds = pipe.encode_prompt(prompt, device="mps")
del pipe.text_encoder
gc.collect()
torch.mps.empty_cache()
image = pipe(prompt_embeds=embeds, ...).images[0]
```
Frees whatever the text encoder uses (could be 1-10GB depending on architecture).

### Priority 4: Group Offloading (If Still Memory-Pressured)

```python
pipe.denoiser.enable_group_offload(
    onload_device=torch.device("mps"),
    offload_device=torch.device("cpu"),
    offload_type="block_level",
    num_blocks_per_group=2
)
```
Only loads 2 transformer blocks at a time. Works on MPS (without stream prefetching).

### Priority 5: Sharded Checkpoints (For Loading Phase)

```python
pipe.denoiser.save_pretrained("model-sharded", max_shard_size="4GB")
```
Reduces peak memory during model loading from 2x to 1.x model size.

### Priority 6: VAE Tiling (For Large Output Images)

```python
pipe.enable_vae_tiling()
pipe.vae = pipe.vae.to(dtype=torch.float32)  # Required for MPS VAE stability
```

### Anti-Patterns to Avoid

1. **Do NOT use `enable_attention_slicing()` on 64GB machines** -- it hurts performance
   when SDPA is available (PyTorch >= 2.0).
2. **Do NOT use `enable_sequential_cpu_offload()`** -- extremely slow and limited benefit
   on unified memory.
3. **Do NOT use bitsandbytes** -- MPS support is not yet stable.
4. **Do NOT use `torch.compile` on MPS** -- limited benefit, focus on eager mode.
5. **Do NOT use FP8 layerwise casting on MPS** -- no native hardware support.
6. **Do NOT call `.to("mps")` before calling offloading methods** -- negates savings.
7. **Do NOT batch generate on MPS** -- use iteration instead.

### Expected Memory Profile (INT8 + Text Encoder Deletion + Group Offload)

```
Model weights (INT8):       ~9.75 GB
Active group (2 blocks):    ~0.5 GB on MPS
Activations/intermediates:  ~3-5 GB
VAE (FP32):                 ~0.67 GB
OS + apps:                  ~16 GB
------------------------------------------
Total:                      ~30-32 GB of 64 GB
Memory pressure:            GREEN
Swap usage:                 ZERO
```

This leaves ~32 GB headroom for safe operation without any swap.

---

## Additional Resources

### Official Documentation
- [Diffusers Memory Optimization](https://huggingface.co/docs/diffusers/en/optimization/memory)
- [Diffusers MPS Guide](https://huggingface.co/docs/diffusers/en/optimization/mps)
- [Diffusers Quantization](https://huggingface.co/docs/diffusers/en/api/quantization)
- [Inference with Big Models](https://huggingface.co/docs/diffusers/main/en/tutorials/inference_with_big_models)
- [HuggingFace Accelerate Big Modeling](https://huggingface.co/docs/accelerate/main/en/usage_guides/big_modeling)

### Key GitHub Issues and PRs
- [Group Offloading PR #10503](https://github.com/huggingface/diffusers/pull/10503)
- [Stream Prefetching PR #10516](https://github.com/huggingface/diffusers/pull/10516)
- [Disk Offloading PR #11682](https://github.com/huggingface/diffusers/pull/11682)
- [Meta Tensor Error #8644](https://github.com/huggingface/diffusers/issues/8644)
- [bitsandbytes MPS Backend PR #1853](https://github.com/bitsandbytes-foundation/bitsandbytes/pull/1853)

### Blog Posts and Guides
- [Simple SDXL Optimizations](https://github.com/huggingface/blog/blob/main/simple_sdxl_optimizations.md)
- [Exploring Quantization Backends](https://huggingface.co/blog/diffusers-quantization)
- [Flux.1 ComfyUI Mac Memory Analysis](https://macgpu.com/en/blog/flux1-comfyui-mac-memory-bottleneck-64gb-unified-memory.html)
- [ComfyUI MLX Extension](https://apatero.com/blog/comfyui-mlx-extension-70-faster-apple-silicon-guide-2025)

### Tools and Libraries
- [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)
- [DFloat11](https://github.com/LeanModels/DFloat11)
- [MLX Framework](https://github.com/ml-explore/mlx)
- [Apple ML Stable Diffusion](https://github.com/apple/ml-stable-diffusion)
- [Quanto (optimum-quanto)](https://huggingface.co/docs/optimum/quanto/overview)
