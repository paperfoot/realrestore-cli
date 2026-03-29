# MPS + torch.compile Optimizations for Diffusion Models on Apple Silicon

**Research Date:** 2026-03-29
**Status:** Comprehensive research complete

---

## Table of Contents

1. [torch.compile with MPS Backend](#1-torchcompile-with-mps-backend)
2. [Attention Optimizations](#2-attention-optimizations)
3. [VAE Tiling and Slicing](#3-vae-tiling-and-slicing)
4. [Model CPU Offloading on Unified Memory](#4-model-cpu-offloading-on-unified-memory)
5. [Float16 vs BFloat16 Handling](#5-float16-vs-bfloat16-handling)
6. [MPS Environment Variables and Flags](#6-mps-environment-variables-and-flags)
7. [Production Pipeline Template](#7-production-pipeline-template)
8. [Benchmarks and Comparisons](#8-benchmarks-and-comparisons)
9. [Recommendations for RealRestore](#9-recommendations-for-realrestore)

---

## 1. torch.compile with MPS Backend

### Current Status (PyTorch 2.8-2.11, as of March 2026)

**torch.compile on MPS is an early prototype.** The PyTorch team tracks progress in [Issue #150121](https://github.com/pytorch/pytorch/issues/150121), targeting tentative beta status. As of March 2026, approximately 8 of 13 tracked items are completed with 3/33 sub-issues resolved.

**Key limitation:** "Attempt to use it to accelerate end-to-end network is likely to fail." Complex fusions often fall back to CPU or run as unfused generic Metal kernels.

### What Works

- Basic operations and simple models compile successfully
- Multi-stage Welford reductions (fixed)
- RMS norm tracing (fixed)
- Dynamic shape support (implemented)
- T5Small and M2M100 shader generation (fixed)

### What Doesn't Work Yet

- **Argument buffer support** not enabled
- **Matrix multiplication decomposition** optimization pending
- **FlexAttention for MPS** (stretch goal)
- **Scaled dot product attention decomposition** decision still pending
- End-to-end diffusion model compilation will fail
- Reduction performance is worse than eager mode for LLMs

### Practical Guidance

```python
import torch

# DO NOT attempt torch.compile on full diffusion pipelines for MPS
# model = torch.compile(unet)  # Will likely fail or regress

# Instead, use eager mode — this is the recommended path for MPS
model = model.to("mps")

# If you must experiment with torch.compile on MPS:
# Enable CPU fallback for unsupported ops
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Try compiling individual small submodules, not the full model
# small_module = torch.compile(model.some_small_layer, backend="inductor")
```

### MPS Inductor Backend Architecture

The MPS backend supports three execution modes:
1. **Eager Mode** (recommended) — direct ATen operations via Metal shaders and MPSGraph
2. **Inductor Compiled** — code generation via MetalKernel class (experimental)
3. **AOTI** (Ahead-of-time compilation) — pre-compiled Metal kernels (experimental)

The Inductor backend generates Metal Shading Language (MSL) kernels, not Triton PTX. This is a fundamentally different compilation path from CUDA.

---

## 2. Attention Optimizations

### Scaled Dot-Product Attention (SDPA) on MPS

At WWDC 2024, Apple introduced fused SDPA in MPSGraph, collapsing multiple matrix operations into a single efficient kernel. However, the MPS SDPA implementation has significant limitations compared to CUDA:

- **No native FlashAttention support** — MPS relies on Apple's SDPA implementation
- **Memory issues with long sequences** — crashes with sequences >12,000 tokens
- **No variable sequence length support** in a single batch without padding (unlike FA2/FA3 on CUDA)

### Attention Slicing (Recommended for < 64GB RAM)

```python
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32,  # float32 preferred on MPS
)
pipe = pipe.to("mps")

# Enable attention slicing — ~20% improvement, better on Apple Silicon
pipe.enable_attention_slicing(slice_size=1)  # ~40% memory reduction

# For 64GB+ RAM systems, attention slicing may actually hurt performance
# Test with and without to determine best setting for your hardware
```

### Metal FlashAttention (External, Not PyTorch Native)

[Metal FlashAttention](https://github.com/philipturner/metal-flash-attention) by Philip Turner provides the best attention performance on Apple Silicon, used by Draw Things:

**Performance gains:**
- Image generation latencies roughly halved (43-120% faster)
- FLUX.1: up to 25% faster than mflux on M2 Ultra per iteration
- FLUX.1: up to 94% faster than ggml implementations
- SD Large 3.5: up to 163% faster than DiffusionKit per iteration
- Up to 20% improvement for FLUX.1/SD3/AuraFlow on M3/M4
- Training: FLUX.1 LoRA at 9s/step/image at 1024x1024 on M2 Ultra

**Availability:**
- Swift reference: [github.com/philipturner/metal-flash-attention](https://github.com/philipturner/metal-flash-attention)
- C++ version: [github.com/liuliu/ccv](https://github.com/liuliu/ccv/tree/unstable/lib/nnc/mfa)
- Integrated in Draw Things app, not directly usable from PyTorch

**Metal FlashAttention v2.5 (M5 support):**
- Up to 4.6x performance improvement on M5 over M4
- Neural Accelerator integration on non-Pro M-series chips
- FLUX.1 [schnell] (12B), Qwen Image (20B), HiDream (17B) run on M5 iPad in under a minute

### Chunked Attention for MPS (Community Workaround)

For large sequence lengths on MPS, dynamic chunking prevents OOM:

```python
import torch
import torch.nn.functional as F

def chunked_sdpa_mps(query, key, value, chunk_size=4096):
    """Memory-efficient SDPA for MPS with dynamic chunking."""
    seq_len = query.shape[-2]
    if seq_len <= chunk_size:
        return F.scaled_dot_product_attention(query, key, value)

    outputs = []
    for i in range(0, seq_len, chunk_size):
        end = min(i + chunk_size, seq_len)
        q_chunk = query[..., i:end, :]
        out_chunk = F.scaled_dot_product_attention(q_chunk, key, value)
        outputs.append(out_chunk)

    return torch.cat(outputs, dim=-2)
```

---

## 3. VAE Tiling and Slicing

### VAE Slicing

Splits batch processing into single images — useful when generating multiple images:

```python
from diffusers import StableDiffusionXLPipeline
import torch

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float32,
).to("mps")

# Process each image in a batch sequentially to reduce peak memory
pipe.enable_vae_slicing()

# Generate batch — each image decoded separately
images = pipe(["prompt"] * 4, num_inference_steps=20).images
```

### VAE Tiling

Divides large images into overlapping tiles for decoding — enables high-resolution generation:

```python
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32,
).to("mps")

# Enable tiled VAE — essential for generating images > 512x512
pipe.enable_vae_tiling()

# Can now generate 4K images without OOM
image = pipe(
    "A detailed landscape",
    width=3840,
    height=2160,
    num_inference_steps=25,
).images[0]
```

**Trade-offs:**
- May introduce slight tone variation between tiles (no visible seams typically)
- Small performance overhead for single images
- Essential for high-resolution output on memory-constrained systems

### Combined Memory Optimization

```python
# Maximum memory savings — combine all techniques
pipe.enable_attention_slicing(slice_size=1)
pipe.enable_vae_slicing()
pipe.enable_vae_tiling()
```

---

## 4. Model CPU Offloading on Unified Memory

### Apple Silicon Unified Memory Advantage

Apple Silicon shares a single memory pool across CPU, GPU, and Neural Engine. This eliminates PCIe bus copies that plague discrete GPU systems. On unified memory, data movement between CPU and MPS is essentially a pointer adjustment, not a physical copy.

**Key insight:** On unified memory systems, CPU offloading has lower overhead than on discrete GPU systems. The conventional wisdom that "GPU is always faster" doesn't hold for memory-bandwidth-bound workloads on Apple Silicon.

### Sequential CPU Offloading (Best for 8-16GB Macs)

```python
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32,
)

# Moves each component to MPS only when needed, then back to CPU
pipe.enable_sequential_cpu_offload()
```

### Model CPU Offloading (Better Performance, More Memory)

```python
# Keeps entire submodels on MPS during their turn
# Better performance than sequential, but uses more memory
pipe.enable_model_cpu_offload()
```

### Manual Component Management (Most Control)

```python
# For image restoration pipelines with multiple models,
# manually manage which model is on MPS
import gc

def move_to_mps(model):
    """Move model to MPS and clear cache."""
    model.to("mps")
    torch.mps.empty_cache()
    gc.collect()
    return model

def move_to_cpu(model):
    """Move model to CPU and free MPS memory."""
    model.to("cpu")
    torch.mps.empty_cache()
    gc.collect()
    return model

# Sequential processing for multi-model pipelines
model_a = move_to_mps(model_a)
output_a = model_a(input_data)
model_a = move_to_cpu(model_a)

model_b = move_to_mps(model_b)
output_b = model_b(output_a)
model_b = move_to_cpu(model_b)
```

### Unified Memory Considerations for 64GB Systems

On a 64GB unified memory Mac:
- PyTorch MPS default high watermark (1.7x recommended max) allows overallocation
- Multiple models can coexist if total stays under ~40-50GB allocated
- System will swap to disk if exceeded, causing severe performance degradation
- Monitor with `torch.mps.current_allocated_memory()`

---

## 5. Float16 vs BFloat16 Handling

### BFloat16: NOT Supported on MPS

The MPS backend **does not support bfloat16** operations. Models requiring bfloat16 will error out:

```python
# This WILL FAIL on MPS:
model = model.to(dtype=torch.bfloat16, device="mps")
# RuntimeError: MPS does not support bfloat16

# Workaround: convert bfloat16 models to float16 or float32
model = model.to(dtype=torch.float32, device="mps")
```

### Float16: Supported but Not Always Faster

Unlike NVIDIA GPUs with Tensor Cores, Apple Silicon does not have dedicated float16 acceleration paths. Float16 saves memory but may not improve speed:

```python
# Float16 — saves memory but may be SLOWER than float32 on MPS
pipe = StableDiffusionPipeline.from_pretrained(
    "model_id",
    torch_dtype=torch.float16,  # Half the memory, potentially slower
).to("mps")

# Float32 — recommended for best speed on MPS
pipe = StableDiffusionPipeline.from_pretrained(
    "model_id",
    torch_dtype=torch.float32,  # More memory, but faster on MPS
).to("mps")
```

### Benchmark Reference (512x512, 25 steps)

| Precision | Time | Memory |
|-----------|------|--------|
| Float32 | 18-20s | ~4GB |
| Float16 | 22-25s | ~2GB |

**Recommendation:** Use float32 for speed unless memory-constrained. On 64GB systems, float32 is strongly preferred. On 8-16GB systems, float16 may be necessary to avoid swapping.

### Handling Models with Mixed Precision

```python
# For models trained with bfloat16, convert at load time
import torch

def load_for_mps(model_path, prefer_half=False):
    """Load a model for MPS, handling precision conversion."""
    state_dict = torch.load(model_path, map_location="cpu")

    # Convert any bfloat16 tensors
    for key, tensor in state_dict.items():
        if tensor.dtype == torch.bfloat16:
            target = torch.float16 if prefer_half else torch.float32
            state_dict[key] = tensor.to(target)

    return state_dict
```

### VAE Precision Considerations

The VAE is particularly sensitive to precision on MPS. Always use float32 for VAE decoding:

```python
# Force VAE to float32 even when rest of pipeline is float16
pipe.vae = pipe.vae.to(dtype=torch.float32)

# Or use the upcast_vae flag in some pipelines
# --no-half-vae flag in AUTOMATIC1111
```

---

## 6. MPS Environment Variables and Flags

### Memory Management

```bash
# High watermark — hard limit for total allocations
# Default: 1.7 (170% of recommended max)
# Set to 0.0 to disable limit (may cause system failure)
# Set to 0.95 for conservative memory usage
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0  # disable limit (use with caution)

# Low watermark — soft limit, triggers GC and adaptive commit
# Default: 1.4 (unified memory) or 1.0 (discrete)
# Set to 0.0 to disable adaptive commit and GC
export PYTORCH_MPS_LOW_WATERMARK_RATIO=1.0

# Enable CPU fallback for unsupported MPS operations
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

### Performance Tuning

```bash
# Enable fast math for MPS kernels (may reduce accuracy)
export PYTORCH_MPS_FAST_MATH=1

# Prefer raw Metal kernels over MPSGraph APIs for matmul
# Can be faster for some workloads
export PYTORCH_MPS_PREFER_METAL=1
```

### Debugging and Profiling

```bash
# Verbose allocator logging
export PYTORCH_DEBUG_MPS_ALLOCATOR=1

# Profile logging options (bitmask via LogOptions enum)
export PYTORCH_MPS_LOG_PROFILE_INFO=<bitmask>

# Trace signposts for Instruments.app profiling
export PYTORCH_MPS_TRACE_SIGNPOSTS=<bitmask>
```

### Recommended Configuration for Image Restoration

```bash
# Production settings for 64GB Mac
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0   # No hard limit
export PYTORCH_MPS_LOW_WATERMARK_RATIO=1.4     # Default adaptive GC
export PYTORCH_MPS_FAST_MATH=1                 # Speed boost, acceptable for image gen
export PYTORCH_ENABLE_MPS_FALLBACK=1           # Prevent crashes on unsupported ops

# Conservative settings for 16GB Mac
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7    # Limit to 70% of recommended max
export PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5     # Aggressive GC
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

### Programmatic Memory Management

```python
import torch
import gc

# Check current memory usage
allocated = torch.mps.current_allocated_memory() / (1024**3)
print(f"MPS allocated: {allocated:.2f} GB")

# Free cached memory
torch.mps.empty_cache()
gc.collect()

# Synchronize before timing operations
torch.mps.synchronize()

# Profiling pattern
import time

torch.mps.synchronize()
start = time.perf_counter()

# ... your operation ...

torch.mps.synchronize()
elapsed = time.perf_counter() - start
print(f"Operation took {elapsed:.3f}s")
```

---

## 7. Production Pipeline Template

### Optimized MPS Diffusion Pipeline

```python
import os
import gc
import time
import torch
from diffusers import StableDiffusionPipeline

# Set environment before importing torch
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["PYTORCH_MPS_FAST_MATH"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


class OptimizedMPSPipeline:
    """Production-ready diffusion pipeline for Apple Silicon MPS."""

    def __init__(self, model_id, use_float16=False):
        self.device = torch.device("mps")
        dtype = torch.float16 if use_float16 else torch.float32

        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            safety_checker=None,
        )

        # Always keep VAE in float32 for quality
        self.pipe.vae = self.pipe.vae.to(dtype=torch.float32)

        # Memory optimizations
        self.pipe.enable_attention_slicing(slice_size=1)
        self.pipe.enable_vae_slicing()
        self.pipe.enable_vae_tiling()

        self.pipe = self.pipe.to(self.device)

        # Warmup pass (required for stable first-inference on some PyTorch versions)
        self._warmup()

    def _warmup(self):
        """Prime the pipeline to avoid first-inference anomalies."""
        _ = self.pipe(
            "warmup",
            num_inference_steps=1,
            generator=torch.Generator("cpu").manual_seed(0),
        )
        torch.mps.empty_cache()
        gc.collect()

    def generate(self, prompt, **kwargs):
        """Generate an image with proper MPS memory management."""
        kwargs.setdefault("num_inference_steps", 20)
        kwargs.setdefault("generator", torch.Generator("cpu").manual_seed(42))

        torch.mps.empty_cache()
        gc.collect()

        torch.mps.synchronize()
        start = time.perf_counter()

        result = self.pipe(prompt, **kwargs).images[0]

        torch.mps.synchronize()
        elapsed = time.perf_counter() - start

        print(f"Generated in {elapsed:.2f}s")
        print(f"MPS memory: {torch.mps.current_allocated_memory() / 1024**3:.2f} GB")

        return result

    def cleanup(self):
        """Release all MPS memory."""
        self.pipe.to("cpu")
        del self.pipe
        torch.mps.empty_cache()
        gc.collect()
```

### Multi-Model Pipeline for Image Restoration

```python
class MPSModelScheduler:
    """Schedules multiple models on MPS with unified memory awareness."""

    def __init__(self, max_mps_gb=40):
        self.max_mps_bytes = max_mps_gb * 1024**3
        self.loaded_models = {}

    def _get_available_memory(self):
        allocated = torch.mps.current_allocated_memory()
        return self.max_mps_bytes - allocated

    def load_model(self, name, model):
        """Load a model to MPS, evicting others if needed."""
        model_size = sum(p.numel() * p.element_size() for p in model.parameters())

        # Evict models if needed
        while self._get_available_memory() < model_size and self.loaded_models:
            oldest = next(iter(self.loaded_models))
            self.unload_model(oldest)

        model.to("mps")
        self.loaded_models[name] = model
        return model

    def unload_model(self, name):
        """Move model to CPU and free MPS memory."""
        if name in self.loaded_models:
            self.loaded_models[name].to("cpu")
            del self.loaded_models[name]
            torch.mps.empty_cache()
            gc.collect()

    def run_model(self, name, model, input_data):
        """Run a model, loading to MPS if needed."""
        self.load_model(name, model)
        with torch.no_grad():
            result = model(input_data)
        return result
```

---

## 8. Benchmarks and Comparisons

### MPS vs CUDA vs MLX for Diffusion

| Framework | Device | SD 1.5 512x512 (25 steps) | SDXL 1024x1024 (25 steps) |
|-----------|--------|---------------------------|---------------------------|
| PyTorch MPS (fp32) | M2 Max 32GB | ~18-20s | ~45-60s |
| PyTorch MPS (fp16) | M2 Max 32GB | ~22-25s | ~50-65s |
| PyTorch CUDA | RTX 4090 | ~2-3s | ~5-8s |
| Draw Things (Metal FA) | M2 Ultra | — | ~15-20s |
| MLX (mflux) | M2 Ultra | — | ~20-25s |

### Pipeline Breakdown (512x512, MPS fp32, 25 steps)

| Component | Time | % Total |
|-----------|------|---------|
| Text Encoding | ~0.5s | ~3% |
| U-Net Denoising | 15-25s | ~85% |
| VAE Decoding | 1-2s | ~8% |
| Overhead | ~0.5s | ~3% |

### Metal FlashAttention vs Standard (Draw Things)

| Model | Standard | With Metal FA | Speedup |
|-------|----------|--------------|---------|
| SD 1.5 | baseline | 43-120% faster | 1.4-2.2x |
| SDXL | baseline | ~50% faster | ~1.5x |
| FLUX.1 | baseline | 20% faster (M3/M4) | 1.2x |
| SD Large 3.5 | baseline | 163% faster (M2U) | 2.6x |

### Apple Silicon Generation Comparison

| Chip | Memory BW | GPU Cores | Relative Perf |
|------|-----------|-----------|---------------|
| M1 | 68 GB/s | 8 | 1.0x |
| M1 Max | 400 GB/s | 32 | ~3.5x |
| M2 | 100 GB/s | 10 | ~1.4x |
| M2 Ultra | 800 GB/s | 76 | ~8x |
| M3 | 100 GB/s | 10 | ~1.5x |
| M4 | 120 GB/s | 10 | ~1.7x |
| M4 Max | 546 GB/s | 40 | ~5x |
| M5 | — | — | ~4.6x over M4 (with Metal FA v2.5) |

---

## 9. Recommendations for RealRestore

### Immediate (Use Now)

1. **Use eager mode, not torch.compile** — torch.compile on MPS is unreliable for diffusion models
2. **Use float32** on 64GB systems for best speed; float16 only if memory-constrained
3. **Enable attention slicing** for systems < 64GB RAM
4. **Enable VAE tiling** for any output > 512x512
5. **Force VAE to float32** regardless of model precision (--no-half-vae)
6. **Use CPU generator** for random seeds (`torch.Generator("cpu").manual_seed(seed)`)
7. **Set `PYTORCH_ENABLE_MPS_FALLBACK=1`** to prevent crashes on unsupported ops
8. **Call `torch.mps.empty_cache()` + `gc.collect()`** between model switches
9. **Use `torch.mps.synchronize()`** before timing operations

### Medium-Term (Monitor)

1. **Watch PyTorch issue #150121** for torch.compile MPS progress — once beta, re-evaluate
2. **Consider MLX conversion** for the most performance-critical models (mflux shows 25% gains)
3. **Evaluate Metal FlashAttention** if building custom attention layers
4. **Test `PYTORCH_MPS_FAST_MATH=1`** — may improve speed with acceptable quality impact
5. **Test `PYTORCH_MPS_PREFER_METAL=1`** — raw Metal kernels can be faster than MPSGraph for matmul

### Architecture Decisions

1. **Multi-model scheduling** is essential — don't load all models simultaneously
2. **Unified memory advantage** — CPU offloading is cheaper than on discrete GPU systems
3. **No batch processing** — iterate instead of batching on MPS (batching crashes or is unreliable)
4. **Convert bfloat16 models to float32** at load time
5. **Profile with `torch.mps.synchronize()`** — MPS operations are asynchronous by default

### Alternative Frameworks to Consider

| Framework | Best For | MPS Advantage |
|-----------|----------|---------------|
| PyTorch MPS | Flexibility, ecosystem | Broad model support |
| MLX | Maximum native perf | Built for Apple Silicon |
| Core ML | Production deployment | ANE + GPU + CPU combined |
| Draw Things | End-user image gen | Metal FlashAttention |

---

## Sources

- [PyTorch MPS Backend Docs](https://docs.pytorch.org/docs/stable/notes/mps.html)
- [PyTorch MPS Environment Variables](https://docs.pytorch.org/docs/stable/mps_environment_variables.html)
- [torch.compile on MPS Progress Tracker (Issue #150121)](https://github.com/pytorch/pytorch/issues/150121)
- [HuggingFace Diffusers MPS Optimization](https://huggingface.co/docs/diffusers/en/optimization/mps)
- [HuggingFace Diffusers Memory Reduction](https://huggingface.co/docs/diffusers/en/optimization/memory)
- [Metal FlashAttention 2.0](https://engineering.drawthings.ai/p/metal-flashattention-2-0-pushing-forward-on-device-inference-training-on-apple-silicon-fe8aac1ab23c)
- [Metal FlashAttention GitHub](https://github.com/philipturner/metal-flash-attention)
- [Metal FlashAttention v2.5 (M5 Support)](https://releases.drawthings.ai/p/metal-flashattention-v25-w-neural)
- [Apple Accelerated PyTorch on Mac](https://developer.apple.com/metal/pytorch/)
- [State of PyTorch Hardware Acceleration 2025](https://tunguz.github.io/PyTorch_Hardware_2025/)
- [MPS-Accelerated Image Generation Deep Dive](https://medium.com/@michael.hannecke/unleashing-apple-silicons-hidden-ai-superpower-a-technical-deep-dive-into-mps-accelerated-image-9573ba90570a)
- [Optimizing PyTorch MPS Attention](https://medium.com/@rakshekaraj/optimizing-pytorch-mps-attention-memory-efficient-large-sequence-processing-without-accuracy-5239f565f07b)
- [Apple ml-stable-diffusion](https://github.com/apple/ml-stable-diffusion)
- [mflux (MLX FLUX)](https://github.com/filipstrand/mflux)
- [FLUX on Apple Silicon Guide](https://www.apatero.com/blog/flux-apple-silicon-m1-m2-m3-m4-complete-performance-guide-2025)
- [PyTorch MPS BFloat16 Issue](https://github.com/pytorch/pytorch/issues/141864)
- [PyTorch 2.9 Release](https://pytorch.org/blog/pytorch-2-9/)
- [PyTorch MPS matmul vs MLX Benchmark](https://kevinmartinjose.com/2025/04/21/matmul-using-pytorchs-mps-backend-is-faster-than-apples-mlx/)
