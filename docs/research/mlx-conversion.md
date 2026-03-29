# MLX Conversion Research for RealRestorer

Research date: 2026-03-29

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [RealRestorer Architecture Analysis](#realrestorer-architecture-analysis)
3. [MLX Framework Capabilities](#mlx-framework-capabilities)
4. [Existing MLX Diffusion Implementations](#existing-mlx-diffusion-implementations)
5. [Qwen2.5-VL MLX Support](#qwen25-vl-mlx-support)
6. [Conversion Strategy](#conversion-strategy)
7. [Quantization Options](#quantization-options)
8. [MLX vs MPS Performance](#mlx-vs-mps-performance)
9. [M4 Max Optimization Opportunities](#m4-max-optimization-opportunities)
10. [Performance Optimization Patterns](#performance-optimization-patterns)
11. [Risk Assessment](#risk-assessment)
12. [Recommended Approach](#recommended-approach)

---

## Executive Summary

Converting RealRestorer from PyTorch/diffusers to MLX is **feasible but non-trivial**. The pipeline has three major components that each require different conversion strategies:

| Component | MLX Support | Conversion Difficulty | Existing Ports |
|-----------|------------|----------------------|----------------|
| Qwen2.5-VL (text encoder) | Strong | Low | mlx-community has 4-bit/8-bit weights |
| Transformer denoiser (DiT) | Moderate | Medium-High | MFLUX/DiffusionKit patterns exist |
| VAE (encoder/decoder) | Moderate | Medium | MLX stable_diffusion has VAE |
| Flow matching scheduler | Easy | Low | Pure math, trivial to port |

**Key finding**: The biggest win is likely a **hybrid approach** -- use MLX for the Qwen2.5-VL text encoder (already ported) and keep PyTorch+MPS for the denoiser/VAE initially, then incrementally port components based on profiling data.

---

## RealRestorer Architecture Analysis

From the upstream pipeline (`pipeline_realrestorer.py`), RealRestorer consists of:

### Components

1. **Text Encoder**: `Qwen2_5_VLForConditionalGeneration` from HuggingFace Transformers
   - Largest component (~7B parameters for the base model)
   - Used for prompt enhancement and conditioning
   - Runs once per image (not in denoising loop)

2. **Transformer Denoiser**: `RealRestorerTransformer2DModel`
   - Custom DiT (Diffusion Transformer) architecture
   - Runs 28 times per image (num_inference_steps=28)
   - Contains attention mechanisms, feed-forward layers
   - Supports guidance embeddings and mask tokens
   - Versions: v1.0 (with scale_factor), v1.1 (without)

3. **VAE**: `RealRestorerAutoencoderKL`
   - Encodes input image to latent space
   - Decodes restored latents back to pixel space
   - 16 latent channels, scale factor of 8
   - Runs in float32 (important for quality)

4. **Scheduler**: `RealRestorerFlowMatchScheduler`
   - Flow matching based (not traditional DDPM/DDIM)
   - Pure math operations, no learned parameters

### Memory Profile (PyTorch bfloat16 baseline)

- Peak GPU memory: ~34 GB at 1024x1024 (per upstream docs)
- Text encoder: ~14-15 GB (Qwen2.5-VL 7B in bf16)
- Transformer denoiser: ~10-12 GB
- VAE: ~2-3 GB (float32)
- Inference tensors: ~4-5 GB

---

## MLX Framework Capabilities

### Core Strengths for This Use Case

1. **Unified Memory**: MLX operates directly on Apple Silicon's unified memory, eliminating CPU-GPU data transfers. This is critical for RealRestorer's 34GB peak usage on a 64GB M4 Max.

2. **Lazy Evaluation**: MLX builds computation graphs and evaluates them lazily. For the 28-step denoising loop, this enables graph-level optimizations across steps.

3. **JIT Compilation (`mx.compile`)**: Fuses operations, eliminates redundant computations, and generates optimized Metal kernels.
   - Compilation pipeline: Graph tracing -> Simplification -> Operation fusion -> Kernel generation
   - Shapeless compilation mode allows reuse across varying dimensions
   - Max fusion depth: 11 operations, max fusion arrays: 24

4. **Custom Metal Kernels**: MLX supports writing custom Metal kernels via `mx.fast.metal_kernel()` for performance-critical operations. Example speedups from the docs:
   - Grid sample: 55.7ms -> 6.7ms (8x speedup on M1 Max)
   - Grid sample VJP: 676.4ms -> 16.7ms (40x speedup)

5. **Built-in Fast Operations**:
   - `mx.fast.scaled_dot_product_attention` -- optimized attention kernel
   - `mx.fast.rms_norm`, `mx.fast.layer_norm` -- fused normalization
   - `mx.fast.rope` -- rotary position embeddings

### Current Limitations

1. **No Conv3d** support (not needed for RealRestorer)
2. **Immutable arrays** -- in-place operations common in PyTorch need rewriting
3. **No native diffusers integration** -- pipeline must be manually ported
4. **Training support is limited** -- inference-only focus (fine for our case)
5. **No equivalent to `torch.compile`'s dynamic shapes** -- shapeless mode is close but not identical

---

## Existing MLX Diffusion Implementations

### 1. MLX Stable Diffusion (ml-explore/mlx-examples)

Apple's reference implementation for Stable Diffusion in MLX.

**Architecture**: UNet-based diffusion (SD 2.1, SDXL-turbo)
**Key patterns**:
- Weights loaded directly from HuggingFace Hub safetensors
- Float16 default, with 4-bit text encoder / 8-bit UNet quantization option
- Runs on 8GB devices with quantization enabled

**Relevant code pattern** (adapted from MLX examples):
```python
import mlx.core as mx
import mlx.nn as nn

class MLXDiffusionPipeline:
    def __init__(self, transformer, vae, text_encoder, scheduler):
        self.transformer = transformer
        self.vae = vae
        self.text_encoder = text_encoder
        self.scheduler = scheduler

    def generate(self, prompt, num_steps=28, guidance_scale=3.0):
        # Encode prompt (runs once)
        text_embeds = self.text_encoder.encode(prompt)
        mx.eval(text_embeds)  # Evaluate before loop

        # Initialize latents
        latents = mx.random.normal(shape)

        # Denoising loop
        for t in self.scheduler.timesteps(num_steps):
            noise_pred = self.transformer(latents, t, text_embeds)
            latents = self.scheduler.step(noise_pred, t, latents)
            mx.eval(latents)  # Evaluate at each step boundary

        # Decode
        image = self.vae.decode(latents)
        mx.eval(image)
        return image
```

### 2. MFLUX (filipstrand/mflux)

Line-by-line MLX port of FLUX from HuggingFace Diffusers. Most relevant reference for RealRestorer conversion.

**Supported models** (as of March 2026):
- FLUX.1 (12B, legacy)
- FLUX.2 (4B-9B, fastest)
- Z-Image (6B, high quality)
- Qwen Image (20B)
- SeedVR2 (3B-7B, upscaler)

**Key conversion patterns used by MFLUX**:
- Ground-up MLX implementation (not wrappers)
- Only tokenizers from HuggingFace Transformers
- Single-stream transformer optimized for unified memory
- Fused Q/K/V projections to minimize memory reads
- Uses `mx.fast.scaled_dot_product_attention` for GPU-optimized attention

**Quantization**: 4-bit and 8-bit via CLI flag `-q 4` or `-q 8`

**Python API example**:
```python
from mflux.models.z_image import ZImageTurbo

model = ZImageTurbo(quantize=8)
image = model.generate_image(
    prompt="Restore details and clarity",
    seed=42,
    num_inference_steps=9,
    width=1024,
    height=1024,
)
image.save("output.png")
```

### 3. DiffusionKit (argmaxinc/DiffusionKit)

Dual-framework toolkit supporting both Core ML and MLX for diffusion models.

**Supported models**: SD3, FLUX.1-schnell, FLUX.1-dev

**MLX integration pattern**:
```python
from diffusionkit.mlx import DiffusionPipeline

pipeline = DiffusionPipeline(
    shift=3.0,
    use_t5=False,
    model_version="argmaxinc/mlx-stable-diffusion-3-medium",
    low_memory_mode=True,
    a16=True,  # 16-bit activations
    w16=True,  # 16-bit weights
)

image = pipeline.generate_image(
    prompt="...",
    seed=42,
    num_inference_steps=50,
)
```

### 4. MLX Z-Image (uqer1244/MLX_z-image)

Recent MLX port of a diffusion model showing the pattern of porting transformer, text encoder, VAE, tokenizer, and scheduler together. Working toward removing torch and diffusers dependencies entirely.

---

## Qwen2.5-VL MLX Support

### Current State: Well Supported

The Qwen2.5-VL vision-language model has robust MLX support through two channels:

#### mlx-community Weights on HuggingFace

Pre-converted MLX weights are available for all Qwen2.5-VL sizes:

| Model | Quantization | HuggingFace ID | Converter |
|-------|-------------|----------------|-----------|
| Qwen2.5-VL-3B-Instruct | 4-bit | `mlx-community/Qwen2.5-VL-3B-Instruct-4bit` | mlx-vlm 0.1.11 |
| Qwen2.5-VL-7B-Instruct | 4-bit | `mlx-community/Qwen2.5-VL-7B-Instruct-4bit` | mlx-vlm 0.1.11 |
| Qwen2.5-VL-7B-Instruct | 8-bit | `mlx-community/Qwen2.5-VL-7B-Instruct-8bit` | mlx-vlm |
| Qwen2.5-VL-32B-Instruct | 4-bit | `mlx-community/Qwen2.5-VL-32B-Instruct-4bit` | mlx-vlm 0.1.21 |
| Qwen2.5-VL-72B-Instruct | 4-bit | `mlx-community/Qwen2.5-VL-72B-Instruct-4bit` | mlx-vlm 0.1.11 |

#### mlx-vlm (Blaizzy/mlx-vlm)

Full-featured VLM inference package for MLX:
```python
from mlx_vlm import load, generate
from mlx_vlm.utils import apply_chat_template

model, processor = load("mlx-community/Qwen2.5-VL-7B-Instruct-4bit")

messages = [{"role": "user", "content": [
    {"type": "image", "image": "input.jpg"},
    {"type": "text", "text": "Describe the image quality issues"}
]}]

formatted = apply_chat_template(processor, messages)
output = generate(model, processor, formatted, max_tokens=256)
```

### RealRestorer Integration Consideration

RealRestorer uses Qwen2.5-VL specifically as a **text encoder** (for prompt enhancement), not as a chat model. The conversion would need to:

1. Load the MLX-converted Qwen2.5-VL weights
2. Extract hidden states (not generate tokens) for conditioning
3. Handle the custom prompt template (`QWEN25VL_PREFIX`)
4. Ensure numerical compatibility with the denoiser's expected input format

The mlx-vlm package handles loading and running Qwen2.5-VL natively, but the hidden-state extraction pattern would need custom code since mlx-vlm is designed for generation, not embedding extraction.

---

## Conversion Strategy

### Option A: Full MLX Port (Highest Performance, Highest Effort)

Port all components to MLX, eliminating PyTorch dependency entirely.

**Steps**:
1. Port `RealRestorerTransformer2DModel` to `mlx.nn` modules
   - Rewrite attention layers using `mx.fast.scaled_dot_product_attention`
   - Convert all Conv2d layers (MLX uses `[O,H,W,I]` layout vs PyTorch `[O,I,H,W]`)
   - Handle guidance embeddings and mask tokens
2. Port `RealRestorerAutoencoderKL` to MLX
   - Careful with float32 requirement for VAE quality
3. Adapt Qwen2.5-VL from mlx-vlm for embedding extraction
4. Port `RealRestorerFlowMatchScheduler` (trivial -- pure math)
5. Convert safetensors weights with proper tensor transpositions

**Weight conversion pattern** (from torch2mlx):
```python
import mlx.core as mx
import numpy as np
from safetensors import safe_open

def convert_pytorch_to_mlx(pytorch_path, output_path):
    weights = {}
    with safe_open(pytorch_path, framework="numpy") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)

            # Transpose Conv2d weights: [O,I,H,W] -> [O,H,W,I]
            if "conv" in key and "weight" in key and tensor.ndim == 4:
                tensor = np.transpose(tensor, (0, 2, 3, 1))

            weights[key] = tensor

    mx.savez(output_path, **weights)
```

**Estimated effort**: 3-4 weeks for a working prototype, 6-8 weeks for production quality.

### Option B: Hybrid MLX + PyTorch/MPS (Moderate Performance, Lower Effort)

Use MLX for the text encoder (Qwen2.5-VL), keep PyTorch+MPS for denoiser and VAE.

**Architecture**:
```
[Input Image] -> [PyTorch/MPS: VAE Encode]
                          |
[Prompt] -> [MLX: Qwen2.5-VL Encode] -> text_embeds (convert to torch tensor)
                          |
              [PyTorch/MPS: 28-step Denoising Loop]
                          |
              [PyTorch/MPS: VAE Decode] -> [Output Image]
```

**Key advantage**: The Qwen2.5-VL text encoder is the single largest component (~14-15GB in bf16). Using the 4-bit MLX version drops it to ~3.5GB, immediately solving the memory problem.

**Data transfer overhead**: Minimal -- text embeddings are small tensors (~[1, 640, 4096]) transferred once before the denoising loop.

**Estimated effort**: 1-2 weeks.

### Option C: MLX Denoiser + MPS VAE (Best Balance)

Port the denoiser (performance-critical, runs 28x) to MLX. Keep VAE in PyTorch (needs float32, runs twice).

**Steps**:
1. Use mlx-community Qwen2.5-VL weights (already done)
2. Port `RealRestorerTransformer2DModel` to MLX (following MFLUX patterns)
3. Keep VAE in PyTorch+MPS with float32
4. Convert between MLX/PyTorch arrays at component boundaries

**Estimated effort**: 2-3 weeks.

### Recommendation: Start with Option B, evolve to C

Option B gives immediate memory savings (from 34GB to ~20-22GB) with minimal effort. Profile the result to determine if the denoiser port (Option C) is worth the investment.

---

## Quantization Options

### MLX Native Quantization

MLX supports quantization through `mx.quantize()` and `nn.QuantizedLinear`:

```python
import mlx.core as mx
import mlx.nn as nn

# Quantize a model
model = load_model()
nn.quantize(model, group_size=64, bits=4)  # 4-bit quantization

# Or quantize specific layers
nn.quantize(model.transformer, bits=8)  # 8-bit for transformer
nn.quantize(model.text_encoder, bits=4)  # 4-bit for text encoder
```

### Available Bit Widths

| Bits | Memory Reduction | Quality Impact | Best For |
|------|-----------------|----------------|----------|
| 4-bit | ~75% | Moderate (acceptable for text encoder) | Qwen2.5-VL text encoder |
| 6-bit | ~62% | Low | Balanced option |
| 8-bit | ~50% | Very low | Transformer denoiser |
| bf16 | Baseline | None | VAE (quality critical) |

### RealRestorer Quantization Strategy

```
Text Encoder (Qwen2.5-VL):  4-bit  (~3.5 GB, down from ~14 GB)
Transformer Denoiser:        8-bit  (~5-6 GB, down from ~10-12 GB)
VAE:                         fp32   (~2-3 GB, keep full precision)
---------------------------------------------------------------
Total estimated:             ~12-15 GB (down from 34 GB)
```

This fits comfortably within the M4 Max's 64GB unified memory with substantial headroom for batch processing.

### Memory Projection on M4 Max (64GB)

| Configuration | Memory Usage | Headroom | Feasible |
|---------------|-------------|----------|----------|
| All bf16 (baseline) | ~34 GB | 30 GB | Yes, but tight for batches |
| Text enc 4-bit, rest bf16 | ~23 GB | 41 GB | Comfortable |
| Text enc 4-bit, denoiser 8-bit, VAE fp32 | ~12-15 GB | 49-52 GB | Excellent |
| All 4-bit (experimental) | ~8-10 GB | 54-56 GB | Maximum headroom, quality risk |

---

## MLX vs MPS Performance

### Production-Grade Benchmark Data

From the paper "Production-Grade Local LLM Inference on Apple Silicon" (arXiv:2511.05502):

| Framework | Throughput | Latency | Notes |
|-----------|-----------|---------|-------|
| MLX | ~230 tok/s | 5-7 ms | Fastest on Apple Silicon |
| MLC-LLM | ~190 tok/s | ~7 ms | Close second |
| PyTorch MPS | ~7-9 tok/s | ~120 ms | Order of magnitude slower |
| llama.cpp | ~180 tok/s | ~8 ms | Competitive |

**Key insight**: MLX is consistently 20-30x faster than PyTorch+MPS for inference workloads.

### Diffusion-Specific Comparisons

| Implementation | Framework | Performance | Notes |
|---------------|-----------|-------------|-------|
| Draw Things + Metal FlashAttention 2.0 | Metal/MLX | Fastest | 43-120% faster than without FlashAttention |
| MFLUX | MLX | Fast | ~25% slower than Draw Things |
| DiffusionKit | MLX + Core ML | Fast | Optimized for specific models |
| PyTorch + MPS | MPS | Baseline | Good but not optimized |
| ggml-based | CPU/GPU | Slowest | 94% slower than Metal FA 2.0 |

### Why MLX Wins for Diffusion

1. **Zero-copy memory**: No CPU-GPU transfers on unified memory
2. **Lazy evaluation**: Graph optimizations across denoising steps
3. **Kernel fusion**: JIT-compiled fused operations reduce memory bandwidth
4. **Native attention**: `mx.fast.scaled_dot_product_attention` uses optimized Metal kernels
5. **Compilation**: `mx.compile` optimizes the entire denoising function

### When MPS is Still Viable

- Existing PyTorch code that works correctly (lower risk)
- Complex operations not yet supported in MLX
- When using `torch.compile` with the MPS backend (improving rapidly)
- Float32 VAE operations (MLX float32 performance is comparable to MPS)

---

## M4 Max Optimization Opportunities

### Hardware Specifications

- **Memory bandwidth**: 546 GB/s (unified memory)
- **Max unified memory**: 128 GB (our target: 64 GB)
- **GPU cores**: 40 cores
- **Neural Engine**: 16-core, 38 TOPS
- **Media Engine**: Hardware ProRes, H.264/HEVC

### macOS 26 MLX Improvements

macOS 26.2 introduced significant MLX performance improvements:
- Neural accelerator support for M5 (M4 benefits from general MLX improvements)
- Up to 4x time-to-first-token improvement on M5 vs M4 baseline
- M4 Max benefits from MLX framework improvements, not neural accelerator support (M5-only feature)

### Memory Bandwidth Utilization

The M4 Max's 546 GB/s bandwidth is critical for diffusion models:

```
Per denoising step memory access (estimated for RealRestorer):
  - Transformer weights read: ~10 GB (bf16)
  - Activation memory: ~2 GB
  - Attention intermediate: ~1 GB
  Total per step: ~13 GB

28 steps total memory traffic: ~364 GB
Time at peak bandwidth: 364 / 546 = 0.67 seconds (theoretical minimum)

With 4-bit quantized transformer:
  - Weights read: ~2.5 GB per step
  28 steps total: ~98 GB
  Time at peak bandwidth: 98 / 546 = 0.18 seconds (theoretical minimum)
```

Real-world performance will be 3-5x slower than theoretical due to compute-bound operations, but this shows quantization directly reduces the bandwidth bottleneck.

### Optimization Techniques for M4 Max

1. **Fused Q/K/V projections**: Single matrix multiply instead of three
   ```python
   # Instead of separate projections:
   # q = self.q_proj(x); k = self.k_proj(x); v = self.v_proj(x)

   # Fused projection:
   qkv = self.qkv_proj(x)  # Single matmul
   q, k, v = mx.split(qkv, 3, axis=-1)
   ```

2. **Compiled denoising function**:
   ```python
   @mx.compile
   def denoise_step(transformer, latents, timestep, text_embeds, guidance_scale):
       noise_pred = transformer(latents, timestep, text_embeds)
       # CFG guidance
       noise_pred_uncond, noise_pred_cond = mx.split(noise_pred, 2)
       noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
       return noise_pred
   ```

3. **Async evaluation pipeline**:
   ```python
   # Pipeline graph construction with computation
   for i, t in enumerate(timesteps):
       noise_pred = denoise_step(transformer, latents, t, text_embeds, cfg)
       latents = scheduler.step(noise_pred, t, latents)
       mx.async_eval(latents)  # Non-blocking evaluation
   ```

4. **Precision-aware memory management**:
   ```python
   # Cast before evaluating to avoid double-precision memory
   weights = mx.load("model.safetensors")
   weights = {k: v.astype(mx.bfloat16) for k, v in weights.items()}
   mx.eval(weights)  # Only bf16 copy in memory
   ```

---

## Performance Optimization Patterns

### Writing Fast MLX Code (from Apple's engineering guide)

1. **Evaluate at iteration boundaries**:
   ```python
   for t in timesteps:
       latents = denoise_step(latents, t)
       mx.eval(latents)  # Natural boundary
   ```

2. **Use `mx.fast` operations**:
   ```python
   # Attention
   attn_output = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)

   # Normalization
   normed = mx.fast.rms_norm(x, weight, eps=1e-6)

   # Positional encoding
   q, k = mx.fast.rope(q, k, dims=head_dim, offset=offset)
   ```

3. **Avoid Python scalars triggering promotion**:
   ```python
   # Bad: promotes half to float32
   x = x * mx.array(2.0)

   # Good: Python scalar has relaxed promotion
   x = x * 2.0
   ```

4. **Use `x @ W.T` for vector-matrix multiply** (faster than `x @ W`)

5. **Custom Metal kernels for bottlenecks**:
   ```python
   # Example: fused bias + activation
   kernel = mx.fast.metal_kernel(
       name="fused_bias_gelu",
       input_names=["x", "bias"],
       output_names=["out"],
       source="""
           uint idx = thread_position_in_grid.x;
           float val = x[idx] + bias[idx % bias_size];
           // GELU approximation
           float cdf = 0.5f * (1.0f + metal::fast::tanh(
               0.7978845608f * (val + 0.044715f * val * val * val)));
           out[idx] = val * cdf;
       """,
   )
   ```

### Metal FlashAttention Integration

For maximum attention performance, consider integrating Metal FlashAttention 2.0:

- 43-120% faster than standard SDPA
- 20% faster than Metal FlashAttention 1.0 on M3/M4
- Includes fused multi-head output projection
- Available as open-source Metal shaders

---

## Risk Assessment

### High Risk

1. **Numerical divergence**: RealRestorer uses custom attention patterns and flow matching. Subtle floating-point differences between PyTorch and MLX could produce visible artifacts in restored images. **Mitigation**: Side-by-side quality comparison at every stage.

2. **Custom operations**: `RealRestorerTransformer2DModel` likely has custom operations not in standard MLX. **Mitigation**: Identify all custom ops during analysis phase, implement as custom Metal kernels if needed.

3. **VAE float32 precision**: The VAE must run in float32 for quality. MLX's float32 performance may not match MPS for conv-heavy operations. **Mitigation**: Keep VAE in PyTorch/MPS as fallback.

### Medium Risk

4. **Weight conversion correctness**: Conv2d weight transposition (`[O,I,H,W]` -> `[O,H,W,I]`) must be exact. **Mitigation**: Automated validation comparing output tensors.

5. **Memory layout**: MLX uses row-major layout; PyTorch is row-major but operations may produce non-contiguous tensors. **Mitigation**: Use `ensure_row_contiguous` in custom kernels.

6. **Einops dependency**: RealRestorer uses `einops.rearrange` and `einops.repeat` extensively. These need MLX equivalents. **Mitigation**: Replace with native `mx.reshape`, `mx.transpose`, `mx.broadcast_to`.

### Low Risk

7. **Scheduler conversion**: Flow matching scheduler is pure math, trivial to port.
8. **Image I/O**: PIL/numpy interop works identically.
9. **mlx-community weights**: Qwen2.5-VL weights are well-tested by the community.

---

## Recommended Approach

### Phase 1: Hybrid Pipeline (Weeks 1-2)

**Goal**: Reduce memory from 34GB to ~20GB with minimal code changes.

1. Load Qwen2.5-VL text encoder via mlx-vlm in 4-bit
2. Extract text embeddings in MLX, convert to PyTorch tensors
3. Keep transformer denoiser and VAE in PyTorch+MPS
4. Benchmark end-to-end latency and memory

```python
# Phase 1 architecture
import mlx.core as mx
import torch
from mlx_vlm import load as mlx_load

# MLX text encoder (4-bit, ~3.5 GB)
qwen_mlx, processor = mlx_load("mlx-community/Qwen2.5-VL-7B-Instruct-4bit")

# Extract embeddings in MLX
text_embeds_mlx = encode_prompt_mlx(qwen_mlx, processor, prompt)
mx.eval(text_embeds_mlx)

# Convert to PyTorch for denoiser
text_embeds_pt = torch.from_numpy(np.array(text_embeds_mlx)).to("mps")

# Rest of pipeline in PyTorch+MPS (unchanged)
result = denoise_and_decode(text_embeds_pt, image, pipe)
```

### Phase 2: MLX Denoiser (Weeks 3-5)

**Goal**: Port the hot loop to MLX for 5-10x speedup over MPS.

1. Port `RealRestorerTransformer2DModel` to `mlx.nn`
2. Use MFLUX's DiT patterns as reference
3. Apply `mx.compile` to denoising step function
4. Convert weights with automated validation
5. Benchmark per-step latency

### Phase 3: Full MLX Pipeline (Weeks 6-8)

**Goal**: Eliminate PyTorch dependency, pure MLX inference.

1. Port VAE to MLX (with float32 precision)
2. Port flow matching scheduler (trivial)
3. Remove all PyTorch/diffusers dependencies
4. End-to-end benchmark and quality validation
5. Package as standalone MLX pipeline

### Phase 4: Production Optimization (Weeks 9-10)

**Goal**: Squeeze maximum performance from M4 Max.

1. Custom Metal kernels for bottleneck operations
2. Metal FlashAttention 2.0 integration
3. Quantization tuning (mixed precision per layer)
4. Memory-mapped weight loading for instant startup
5. Batch processing optimization

### Expected Performance Targets

| Metric | PyTorch+MPS (baseline) | Phase 1 (hybrid) | Phase 2 (MLX denoiser) | Phase 3 (full MLX) |
|--------|----------------------|-------------------|----------------------|-------------------|
| Peak memory | ~34 GB | ~20 GB | ~15 GB | ~12-15 GB |
| Latency (1024x1024) | TBD | TBD - ~10% faster | TBD - 3-5x faster | TBD - 5-10x faster |
| First run startup | Slow (model load) | Moderate | Fast (cached kernels) | Fastest |
| Quality | Reference | Identical | Near-identical | Near-identical |

---

## Key References

### GitHub Repositories
- [ml-explore/mlx](https://github.com/ml-explore/mlx) -- Core MLX framework
- [ml-explore/mlx-examples](https://github.com/ml-explore/mlx-examples) -- Stable Diffusion in MLX
- [filipstrand/mflux](https://github.com/filipstrand/mflux) -- FLUX port to MLX (best reference for DiT conversion)
- [argmaxinc/DiffusionKit](https://github.com/argmaxinc/DiffusionKit) -- MLX + Core ML diffusion
- [Blaizzy/mlx-vlm](https://github.com/Blaizzy/mlx-vlm) -- Qwen2.5-VL on MLX
- [SynapticSage/torch2mlx](https://github.com/SynapticSage/torch2mlx) -- Automated PyTorch to MLX conversion
- [SattamAltwaim/Xforge](https://github.com/SattamAltwaim/Xforge) -- PyTorch to MLX/CoreML converter
- [elementalcollision/safetensorstomlx](https://github.com/elementalcollision/safetensorstomlx) -- Safetensors to MLX conversion
- [riccardomusmeci/mlx-image](https://github.com/riccardomusmeci/mlx-image) -- Image models in MLX
- [TristanBilot/mlx-benchmark](https://github.com/TristanBilot/mlx-benchmark) -- MLX benchmarks across Apple Silicon

### HuggingFace Models
- [mlx-community/Qwen2.5-VL-7B-Instruct-4bit](https://huggingface.co/mlx-community/Qwen2.5-VL-7B-Instruct-4bit)
- [mlx-community/Qwen2.5-VL-7B-Instruct-8bit](https://huggingface.co/mlx-community/Qwen2.5-VL-7B-Instruct-8bit)
- [mlx-community/Qwen2.5-VL-32B-Instruct-4bit](https://huggingface.co/mlx-community/Qwen2.5-VL-32B-Instruct-4bit)

### Documentation
- [MLX Documentation](https://ml-explore.github.io/mlx/build/html/index.html)
- [Custom Metal Kernels](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html)
- [Writing Fast MLX Code](https://gist.github.com/awni/4beb1f7dfefc6f9426f3a7deee74af50) (Awni Hannun, Apple)
- [WWDC 2025: Get started with MLX](https://developer.apple.com/videos/play/wwdc2025/315/)
- [WWDC 2025: Explore LLMs on Apple Silicon with MLX](https://developer.apple.com/videos/play/wwdc2025/298/)

### Research Papers
- [Benchmarking On-Device ML on Apple Silicon with MLX](https://arxiv.org/abs/2510.18921)
- [Production-Grade Local LLM Inference on Apple Silicon](https://arxiv.org/abs/2511.05502)
- [Metal FlashAttention 2.0](https://engineering.drawthings.ai/p/metal-flashattention-2-0-pushing-forward-on-device-inference-training-on-apple-silicon-fe8aac1ab23c)
- [Exploring LLMs with MLX on M5](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)
