# Memory Optimization for 64GB Unified Memory Apple Silicon

## Research Date: 2026-03-29

## Executive Summary

Running RealRestorer (a diffusion pipeline with Qwen2.5-VL text encoder + custom transformer + VAE) on 64GB Apple Silicon requires fundamentally different optimization strategies than discrete GPU setups. The unified memory architecture eliminates PCIe transfer bottlenecks but introduces a ~48GB effective GPU memory ceiling (75% of 64GB) and shared memory pressure from the OS and other applications. This document covers optimal strategies for maximizing throughput and minimizing memory footprint on these systems.

---

## 1. CPU Offloading on Unified Memory: A Different Game

### Why Traditional Offloading is Counter-Productive

On discrete GPU systems (NVIDIA), CPU offloading moves tensors across PCIe (16-64 GB/s bandwidth). The CPU and GPU have separate memory pools, so offloading genuinely frees GPU VRAM.

On Apple Silicon with unified memory, **CPU and GPU share the same physical RAM**. Moving a tensor from "GPU" to "CPU" does not free physical memory -- it only changes the memory's residency hint. The tensor remains in the same DRAM. This means:

- `enable_sequential_cpu_offload()` adds transfer overhead (~3-5x slowdown) with **zero actual memory savings** on unified memory
- `enable_model_cpu_offload()` adds ~10-20% latency overhead, also with minimal memory benefit
- The only benefit of offloading on unified memory is changing Metal allocation tracking, which can help avoid hitting the `recommendedMaxWorkingSetSize` limit

### Recommended Strategy for 64GB

```
Priority: Keep everything on MPS device. Avoid offloading entirely.

RealRestorer pipeline components:
- text_encoder (Qwen2.5-VL): ~3-7GB depending on quantization
- transformer: ~5-12GB depending on model variant
- vae: ~200MB-1GB
- Total: ~8-20GB — fits comfortably in 64GB unified memory
```

**When offloading IS useful on unified memory:** Only when the combined model weights + activations + intermediate tensors exceed the Metal `recommendedMaxWorkingSetSize` (~48GB on 64GB systems). For RealRestorer at standard resolutions, this should not be necessary.

### The 75% GPU Memory Ceiling

Metal's GPU allocator has a hard limit at approximately 75% of total system RAM:
- 64GB system → ~48GB usable for GPU allocations
- This is enforced by `recommendedMaxWorkingSetSize` in the Metal driver

**Workarounds:**
```bash
# Disable PyTorch's MPS high watermark (use with caution)
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# System-level override (requires sudo, may cause instability)
sudo sysctl iogpu.wired_limit_mb=57344  # ~56GB
```

**Recommendation:** Do not override the 75% limit. Instead, optimize model memory usage through quantization and tiling. The 48GB ceiling is more than sufficient for RealRestorer inference.

---

## 2. Memory-Mapped Model Loading

### PyTorch mmap Loading

PyTorch supports memory-mapped checkpoint loading since v2.1:

```python
state_dict = torch.load("model.safetensors", mmap=True, weights_only=True)
```

**Benefits on unified memory:**
- Pages are loaded on-demand from disk, reducing peak memory during model initialization
- Particularly useful for the text encoder (Qwen2.5-VL), which may have large embedding tables
- OS can page out unused weight pages back to SSD without explicit management

**Caveats:**
- Running forward passes directly on mmap'd weights is very slow — always materialize weights to contiguous tensors before inference
- On Apple Silicon, SSD bandwidth (up to 7.4 GB/s on M4 Pro/Max) is high enough for fast initial loading but not for repeated access during inference

### Recommended Loading Strategy

```python
# Phase 1: mmap load to reduce peak memory during initialization
pipe = RealRestorerPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,  # Uses meta tensors + sequential loading
)

# Phase 2: Move to MPS — on unified memory this is just a residency hint change
pipe.to("mps")
```

### Safetensors Advantage

Safetensors format enables true zero-copy loading through memory mapping. Unlike pickle-based PyTorch checkpoints, safetensors files can be mmap'd without deserialization overhead. The RealRestorer pipeline should prefer safetensors format for all model weights.

---

## 3. Attention Computation Memory Optimization

### The Attention Memory Problem

For diffusion models, attention is the primary memory bottleneck. Standard attention requires O(n^2) memory for the attention matrix where n is the sequence length (spatial tokens for image models). For a 1024x1024 image with 8x downscaling, this is 128x128 = 16,384 tokens.

### Strategy A: Attention Slicing (Baseline)

Diffusers' built-in attention slicing processes attention heads sequentially:

```python
pipe.enable_attention_slicing(slice_size="auto")
# or specify exact slice size
pipe.enable_attention_slicing(slice_size=1)  # minimum memory, maximum latency
```

**Memory reduction:** Divides attention memory by number of heads
**Speed cost:** ~10-20% slower on MPS (less than on CUDA due to unified memory)
**Recommendation:** Use `slice_size="auto"` as a baseline. Good for 64GB but not optimal.

### Strategy B: Chunked Attention (Better)

Process attention in spatial chunks rather than by head:

```python
# Custom chunked attention for MPS
def chunked_attention(query, key, value, chunk_size=4096):
    """Process attention in chunks to bound peak memory."""
    seq_len = query.shape[1]
    output = torch.empty_like(query)
    for i in range(0, seq_len, chunk_size):
        end = min(i + chunk_size, seq_len)
        q_chunk = query[:, i:end]
        attn_weights = torch.matmul(q_chunk, key.transpose(-2, -1)) / math.sqrt(key.shape[-1])
        attn_weights = torch.softmax(attn_weights, dim=-1)
        output[:, i:end] = torch.matmul(attn_weights, value)
    return output
```

**Memory reduction:** Linear in chunk_size instead of quadratic in sequence length
**Recommendation:** Use chunk_size=4096 as default, reduce to 2048 for very high-res images

### Strategy C: Metal FlashAttention (Optimal)

The Draw Things project has released Metal FlashAttention 2.0, a native Metal implementation:

- **Repository:** [philipturner/metal-flash-attention](https://github.com/philipturner/metal-flash-attention)
- **C++ backend:** Available in [liuliu/ccv](https://github.com/liuliu/ccv/tree/unstable/lib/nnc/mfa)
- **Performance:** 43-120% faster than standard attention across SD architectures
- **Memory:** O(1) in sequence length (constant memory regardless of resolution)

**Key results on Apple Silicon:**
- Up to 20% inference improvement for FLUX.1 on M3/M4
- Up to 163% faster than DiffusionKit for SD 3.5 Large on M2 Ultra
- Supports both forward pass (inference) and backward pass (training/LoRA)

**Metal FlashAttention 2.5 + Neural Accelerators:** On M5, delivers breakthrough performance by also leveraging the GPU's neural accelerators for attention computation.

**Integration path for RealRestorer:**
1. Use `torch.nn.functional.scaled_dot_product_attention()` which has MPS support in recent PyTorch
2. For maximum performance, integrate Metal FlashAttention via custom Metal kernels
3. MLX path provides FlashAttention-equivalent memory efficiency natively

### Strategy D: PyTorch SDPA on MPS

PyTorch 2.x's `scaled_dot_product_attention` has MPS support but with caveats:

```python
# Enable SDPA globally in diffusers
pipe.transformer.set_attn_processor(AttnProcessor2_0())
```

**Caveat:** MPS SDPA has known issues with sequences >12,000 tokens. For high-resolution RealRestorer inference (e.g., 2048x2048 → 65,536 tokens in latent space), chunked or Metal FlashAttention is preferred.

---

## 4. VAE Tiling for Large Images

### The VAE Memory Problem

The VAE encoder/decoder processes images at full pixel resolution. For a 2048x2048 image, the VAE must process a 2048x2048x3 tensor through multiple convolution layers, creating massive intermediate activations.

### Tiled VAE Implementation

Diffusers has built-in tiled VAE support:

```python
pipe.enable_vae_tiling()
# or with custom tile size
pipe.vae.enable_tiling(tile_sample_min_size=512)
```

**How it works:**
1. Split image into overlapping tiles (default overlap: 32-64 pixels)
2. Encode/decode each tile independently
3. Blend overlapping regions using linear interpolation
4. Handle GroupNorm by computing global statistics across all tiles first

**Memory profile for different tile sizes:**

| Image Size | No Tiling | 512px Tiles | 256px Tiles |
|-----------|-----------|-------------|-------------|
| 1024x1024 | ~4GB      | ~1.5GB      | ~0.8GB      |
| 2048x2048 | ~16GB     | ~1.5GB      | ~0.8GB      |
| 4096x4096 | ~64GB     | ~1.5GB      | ~0.8GB      |

**Key insight:** VAE tiling makes memory usage approximately constant regardless of output resolution.

### Advanced: Tiled VAE with GroupNorm Correction

The naive tiling approach breaks GroupNorm statistics (mean/variance computed per tile differs from global). The pkuliyi2015 algorithm handles this:

1. Forward pass: compute GroupNorm statistics for all tiles
2. Aggregate global mean/variance
3. Re-apply GroupNorm with corrected statistics
4. Continue forward pass with corrected activations

**Recommendation for RealRestorer:**
- Enable VAE tiling by default for images >1024x1024
- Use 512px tiles for 64GB systems (good speed/memory balance)
- Use 256px tiles only when processing extremely large images (>4096px)

### Tiled Diffusion (Latent Space Tiling)

For even larger images, tile at the latent/diffusion level:

```python
pipe.enable_vae_tiling()
# Additionally tile the diffusion process itself
# Using MultiDiffusion or Mixture of Diffusers approach
```

This processes the denoising loop in spatial tiles, dramatically reducing transformer/attention memory. However, this requires careful implementation to avoid seam artifacts.

---

## 5. Activation Checkpointing During Inference

### Why This Matters for Inference (Not Just Training)

While gradient checkpointing is typically associated with training, the same principle applies to inference when activation memory is the bottleneck. During the forward pass, intermediate activations from each transformer block are kept in memory. For deep transformers (RealRestorer's transformer has many layers), this can consume significant memory.

### Inference-Time Activation Strategies

**Strategy 1: Sequential Block Execution with Cache Clearing**

```python
# Instead of keeping all block activations in memory:
def forward_with_memory_clearing(blocks, hidden_states):
    for block in blocks:
        hidden_states = block(hidden_states)
        # Clear MPS cache between blocks
        if hidden_states.device.type == "mps":
            torch.mps.empty_cache()
    return hidden_states
```

**Memory saving:** Reduces peak activation memory from O(num_blocks) to O(1)
**Speed cost:** Negligible — just cache clearing between blocks

**Strategy 2: Selective Activation Caching**

For models that reuse activations (e.g., skip connections, cross-attention with shared keys):

```python
# Cache only cross-attention K/V tensors, recompute self-attention
# This saves the quadratic self-attention activations
# while preserving the linear cross-attention cache
```

**Strategy 3: torch.utils.checkpoint for Inference**

Even during inference, `torch.utils.checkpoint` can reduce memory by recomputing activations:

```python
import torch.utils.checkpoint as checkpoint

class MemoryEfficientBlock(nn.Module):
    def forward(self, x, *args):
        # Recompute block activations instead of storing them
        return checkpoint.checkpoint(self._forward_impl, x, *args, use_reentrant=False)
```

**Trade-off:** ~20-30% slower due to recomputation, but reduces activation memory by ~50-70%
**Recommendation:** Only use for very high-resolution inference (>2048px) where activation memory exceeds available headroom

---

## 6. Model Sharding Across CPU/GPU on Unified Memory

### Why Traditional Sharding is Unnecessary

On discrete GPU systems, model sharding splits layers between CPU RAM and GPU VRAM. On Apple Silicon unified memory, both "CPU" and "GPU" tensors reside in the same physical DRAM. There is no benefit to splitting a model across devices.

### What Matters Instead: Metal Allocation Tracking

The MPS backend tracks GPU memory allocations separately from CPU allocations, even though they share physical memory. The GPU has a ~48GB allocation limit (75% of 64GB). Sharding helps only to stay under this allocation limit.

### Smart Component Placement for RealRestorer

```python
# For 64GB unified memory, keep everything on MPS:
pipe.to("mps")

# If hitting the 48GB Metal limit with high-res + large models:
# Option A: Keep text encoder on CPU (it only runs once per inference)
pipe.text_encoder.to("cpu")
pipe.transformer.to("mps")
pipe.vae.to("mps")

# Option B: Use diffusers' model offloading (moves components as needed)
pipe.enable_model_cpu_offload(device="mps")
# This moves text_encoder→transformer→vae sequentially
# On unified memory, the "movement" is just a residency hint change
# Much faster than on discrete GPU (no actual data copy)
```

### MLX: True Zero-Copy Alternative

MLX arrays live in unified memory by design — no concept of "CPU" vs "GPU" tensors:

```python
import mlx.core as mx

# Arrays are created in unified memory
# Operations automatically dispatch to CPU or GPU
weights = mx.load("model.safetensors")
# No .to("device") needed — it's all unified
```

**Recommendation:** For maximum memory efficiency on Apple Silicon, the MLX path eliminates all overhead from device placement decisions. The PyTorch MPS path works well but carries some allocation tracking overhead.

---

## 7. Page-Locked (Pinned) Memory on Apple Silicon

### CUDA vs Metal Pinned Memory

On CUDA systems, `pin_memory()` locks pages in physical RAM, preventing the OS from swapping them and enabling faster DMA transfers via PCIe.

On Apple Silicon:
- **There is no PCIe bus** between CPU and GPU
- **Unified memory means all memory is "pinned"** from the GPU's perspective — the GPU can access any physical page directly
- `pin_memory()` in PyTorch on MPS has **no performance benefit** and may waste memory by preventing the OS from paging

### What Matters Instead: Memory Pressure Management

```python
# Monitor memory pressure programmatically
import subprocess
result = subprocess.run(["memory_pressure"], capture_output=True, text=True)
# Green = safe, Yellow = caution, Red = danger

# Proactive cache clearing
import gc
gc.collect()
torch.mps.empty_cache()
```

### Best Practices for 64GB Systems

1. **Do not use `pin_memory=True`** in DataLoaders on Apple Silicon
2. **Do not set `non_blocking=True`** for MPS transfers — there are synchronization issues
3. **Do call `torch.mps.empty_cache()`** between pipeline stages (text encoding → denoising → VAE decode)
4. **Do use `gc.collect()`** before heavy operations to free Python-side references

---

## 8. Key Projects and Frameworks

### MLX (ml-explore/mlx)
- Apple's native array framework for Apple Silicon
- True unified memory — arrays live in shared memory, no device placement
- Native quantization support (4-bit, 8-bit)
- Stable Diffusion 40% faster than PyTorch MPS; FLUX 50-70% faster via ComfyUI MLX
- **Status:** Production-ready for image generation (SDXL, FLUX via mflux)

### mflux (filipstrand/mflux)
- MLX-native FLUX implementation
- `--mlx-cache-limit-gb` for fine-grained memory control
- `--low-ram` mode for constrained systems
- Outperforms PyTorch MPS by 50-70% for FLUX inference

### Metal FlashAttention (philipturner/metal-flash-attention)
- Native Metal implementation of FlashAttention 2.0
- Used in Draw Things for fastest-in-ecosystem image generation
- 43-120% speedup over standard attention
- O(1) memory in sequence length
- Available in Swift and C++ (via ccv)

### vllm-mlx (waybarrios/vllm-mlx)
- Production-grade inference server for Apple Silicon
- Zero-copy cache management exploiting unified memory
- Paged attention for efficient KV cache
- Content-based prefix caching
- Up to 525 tok/s on M4 Max (Qwen3-0.6B)

### Draw Things
- Most optimized diffusion inference app on Apple Silicon
- Integrates Metal FlashAttention, CoreML, ANE offloading
- Demonstrates what is achievable with full Apple Silicon optimization

### vllm-metal (vllm-project/vllm-metal)
- Community-maintained vLLM plugin for Apple Silicon
- Metal backend with MPS acceleration
- Docker Model Runner integration

---

## 9. Benchmarking Different Offloading Strategies

### Expected Performance Profile for RealRestorer on 64GB M4 Pro/Max

Based on research and hardware specifications:

| Strategy | Peak Memory | Latency (1024x1024, 28 steps) | Notes |
|----------|-------------|-------------------------------|-------|
| Full MPS (no offload) | ~15-20GB | Baseline (fastest) | Recommended default |
| Model CPU offload | ~12-15GB | +10-20% | Unnecessary on 64GB |
| Sequential CPU offload | ~8-10GB | +200-400% | Never use on unified memory |
| Full MPS + attention slicing | ~10-15GB | +10-20% | Good safety margin |
| Full MPS + VAE tiling | ~12-16GB | +5-10% | Essential for >1024px |
| Full MPS + bf16 | ~10-12GB | Baseline | Default dtype for M-series |
| MLX native (mflux-style) | ~8-15GB | -30-50% vs MPS | Optimal path |

### Memory Bandwidth: The Real Bottleneck

Diffusion model inference on Apple Silicon is **memory-bandwidth bound**, not compute-bound:

| Chip | Memory Bandwidth | Effective for Inference |
|------|-----------------|----------------------|
| M4 | 120 GB/s | Adequate for SD/SDXL |
| M4 Pro | 273 GB/s | Good for FLUX/RealRestorer |
| M4 Max | 546 GB/s | Excellent for large models |
| M5 (2025) | ~630 GB/s (est.) | Best available |

**Key insight:** On memory-bandwidth-bound workloads, reducing model size through quantization has a **double benefit**: less memory usage AND faster inference (fewer bytes to read from memory).

---

## 10. Recommended Memory Optimization Stack for RealRestorer

### Tier 1: Essential (Always Enable)

1. **Use bfloat16 dtype** — halves model memory vs float32, natively supported on M-series
2. **Enable VAE tiling** for images >1024px — makes VAE memory constant regardless of resolution
3. **Use safetensors format** with `low_cpu_mem_usage=True` for loading — reduces peak init memory
4. **Call `torch.mps.empty_cache()` + `gc.collect()`** between pipeline stages

### Tier 2: Recommended (Significant Gains)

5. **Apply 4-bit or 8-bit quantization** to the text encoder (Qwen2.5-VL) — largest component, runs once per inference
6. **Use `scaled_dot_product_attention`** (SDPA) — built into PyTorch 2.x, MPS-optimized
7. **Keep all components on MPS** — no offloading on 64GB unified memory
8. **Enable attention slicing** as fallback for very high-res (>2048px)

### Tier 3: Advanced (Maximum Performance)

9. **MLX conversion** of transformer and VAE for zero-copy unified memory access
10. **Metal FlashAttention** integration for O(1) attention memory
11. **Tiled diffusion** for ultra-high-resolution (>4096px) output
12. **Selective activation checkpointing** for the transformer during high-res inference
13. **GGUF quantization** of text encoder for maximum compression with quality preservation

### Tier 4: System-Level

14. **Close memory-heavy applications** before inference (browsers, etc.)
15. **Monitor memory pressure** — stay in "green" zone
16. **Do NOT override `recommendedMaxWorkingSetSize`** unless benchmarks prove stability
17. **Do NOT use `pin_memory` or `non_blocking`** on MPS

---

## 11. Inference Scheduling Strategy

### Pipeline Stage Scheduling

RealRestorer inference has three distinct stages with different memory profiles:

```
Stage 1: Text Encoding (Qwen2.5-VL)
  - Runs ONCE per inference
  - Memory: 3-7GB for weights + activations
  - Can be quantized aggressively (4-bit)
  - Output: text embeddings (~small tensor)

Stage 2: Denoising Loop (Transformer, N steps)
  - Runs N times (default 28)
  - Memory: 5-12GB weights + 2-8GB activations (resolution-dependent)
  - Most time-consuming stage
  - Benefits most from attention optimization

Stage 3: VAE Decode
  - Runs ONCE per inference
  - Memory: 0.2-1GB weights, but 4-64GB activations (resolution-dependent!)
  - Tiling is essential for large images
  - Output: final pixel image
```

### Optimal Scheduling for 64GB

```python
# Recommended inference flow for 64GB Apple Silicon
def optimized_inference(pipe, image, prompt, steps=28):
    # Stage 1: Text encoding
    text_embeds = pipe.encode_prompt(prompt)
    torch.mps.empty_cache()  # Free text encoder activations

    # Stage 2: Denoising (transformer)
    # Keep transformer on MPS throughout all steps
    latents = pipe.denoise(text_embeds, image, num_steps=steps)
    torch.mps.empty_cache()  # Free denoising activations

    # Stage 3: VAE decode with tiling
    pipe.enable_vae_tiling()
    output = pipe.vae.decode(latents)
    torch.mps.empty_cache()

    return output
```

### Batch Processing Strategy

For processing multiple images:
- **Do NOT increase batch size** — memory scales linearly with batch size
- **Process sequentially** with cache clearing between images
- **Reuse text embeddings** if the same prompt is used across images
- **Pre-encode all prompts** before starting the denoising loop to free text encoder memory

---

## 12. Comparison: PyTorch MPS vs MLX for RealRestorer

| Aspect | PyTorch MPS | MLX |
|--------|------------|-----|
| Memory overhead | ~10-15% from device tracking | ~0% (true unified) |
| Attention | SDPA (limited for long seq) | Native efficient attention |
| Quantization | Via bitsandbytes (limited MPS) | Native 4/8-bit support |
| Model ecosystem | Full HuggingFace compatibility | Requires conversion |
| Performance | Good | 30-70% faster |
| Memory management | `empty_cache()` needed | Lazy eval handles it |
| Flash Attention | Via Metal FA (custom integration) | Built-in |
| Maturity for diffusion | Proven (diffusers) | Growing (mflux, ComfyUI-MLX) |

**Recommendation:** Start with PyTorch MPS (immediate compatibility with diffusers/RealRestorer), provide MLX as an optimized backend for users who want maximum performance.

---

## Sources

- [Apple Silicon vs NVIDIA CUDA AI Comparison 2025](https://scalastic.io/en/apple-silicon-vs-nvidia-cuda-ai-2025/)
- [Profiling Apple Silicon Performance for ML Training](https://arxiv.org/pdf/2501.14925)
- [Explore LLMs on Apple Silicon with MLX - WWDC25](https://developer.apple.com/videos/play/wwdc2025/298/)
- [MPS-Accelerated Image Generation Deep Dive](https://medium.com/@michael.hannecke/unleashing-apple-silicons-hidden-ai-superpower-a-technical-deep-dive-into-mps-accelerated-image-9573ba90570a)
- [Metal FlashAttention 2.0 - Draw Things](https://engineering.drawthings.ai/p/metal-flashattention-2-0-pushing-forward-on-device-inference-training-on-apple-silicon-fe8aac1ab23c)
- [Metal FlashAttention GitHub](https://github.com/philipturner/metal-flash-attention)
- [Integrating Metal FlashAttention](https://engineering.drawthings.ai/p/integrating-metal-flashattention-accelerating-the-heart-of-image-generation-in-the-apple-ecosystem-16a86142eb18)
- [Optimizing PyTorch MPS Attention](https://medium.com/@rakshekaraj/optimizing-pytorch-mps-attention-memory-efficient-large-sequence-processing-without-accuracy-5239f565f07b)
- [Diffusers MPS Optimization](https://huggingface.co/docs/diffusers/en/optimization/mps)
- [Diffusers Memory Reduction](https://huggingface.co/docs/diffusers/en/optimization/memory)
- [Tiled Diffusion & VAE](https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111)
- [ComfyUI TiledDiffusion](https://github.com/shiimizu/ComfyUI-TiledDiffusion)
- [PyTorch Activation Checkpointing](https://pytorch.org/blog/activation-checkpointing-techniques/)
- [MLX Framework](https://github.com/ml-explore/mlx)
- [mflux - MLX FLUX](https://github.com/filipstrand/mflux)
- [vllm-mlx](https://github.com/waybarrios/vllm-mlx)
- [vllm-metal](https://github.com/vllm-project/vllm-metal)
- [Native LLM Inference at Scale on Apple Silicon](https://arxiv.org/html/2601.19139v1)
- [Benchmarking On-Device ML on Apple Silicon with MLX](https://arxiv.org/html/2510.18921v1)
- [Production-Grade Local LLM Inference on Apple Silicon](https://arxiv.org/abs/2511.05502)
- [Apple M5 Announcement](https://www.apple.com/newsroom/2025/10/apple-unleashes-m5-the-next-big-leap-in-ai-performance-for-apple-silicon/)
- [PyTorch MPS Memory Management](https://github.com/pytorch/pytorch/issues/104188)
- [Leveraging Unified Memory for MPS Tensors](https://github.com/pytorch/pytorch/issues/172987)
- [Metal FlashAttention v2.5 with Neural Accelerators](https://releases.drawthings.ai/p/metal-flashattention-v25-w-neural)
- [ComfyUI MLX Extension Guide](https://apatero.com/blog/comfyui-mlx-extension-70-faster-apple-silicon-guide-2025)
- [Exploring LLMs with MLX and M5 Neural Accelerators](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)
- [FLUX on Apple Silicon Complete Guide](https://www.apatero.com/blog/flux-apple-silicon-m1-m2-m3-m4-complete-performance-guide-2025)
- [PyTorch mmap Loading](https://discuss.pytorch.org/t/offloading-mmap-state-dict-to-cpu/206728)
- [llama.cpp Unified Memory Discussion](https://github.com/ggml-org/llama.cpp/discussions/3083)
- [MPS High Watermark Ratio](https://discuss.pytorch.org/t/mps-backend-out-of-memory/183879)
- [Apple Silicon Limitations for Local LLM](https://stencel.io/posts/apple-silicon-limitations-with-usage-on-local-llm%20.html)
