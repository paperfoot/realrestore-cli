# Quantization Techniques for RealRestorer Vision Models on Apple Silicon

**Research Date:** 2026-03-29
**Scope:** Quantization methods applicable to diffusion-based image restoration models running on Apple Silicon (M-series, 64GB unified memory)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [GPTQ Quantization](#gptq-quantization)
3. [AWQ (Activation-aware Weight Quantization)](#awq-activation-aware-weight-quantization)
4. [Google TurboQuant](#google-turboquant)
5. [GGUF Format](#gguf-format)
6. [bitsandbytes on Apple Silicon](#bitsandbytes-on-apple-silicon)
7. [PyTorch Native Quantization (torch.ao / Quanto)](#pytorch-native-quantization)
8. [Quality vs Speed Tradeoffs by Bit Width](#quality-vs-speed-tradeoffs)
9. [Component-Level Quantization Strategy](#component-level-quantization-strategy)
10. [GitHub Library Activity](#github-library-activity)
11. [Recommendations for RealRestorer](#recommendations-for-realrestorer)

---

## Executive Summary

For RealRestorer running on Apple Silicon with 64GB unified memory, the quantization landscape as of March 2026 offers several viable paths, each with distinct tradeoffs:

- **Best Apple Silicon native path:** Quanto (int8 weights, MPS-compatible, device-agnostic) or GGUF (Metal backend via llama.cpp ecosystem)
- **Best quality preservation:** AWQ (only 0.7% accuracy drop at 4-bit) or GPTQ via GPTQModel (Apple CPU support)
- **Most memory savings:** torchao int4_weight_only (66% reduction from BF16) or bitsandbytes NF4 (60% reduction)
- **Emerging technique:** Google TurboQuant (KV cache compression, already ported to MLX for Apple Silicon)
- **Key constraint:** MPS backend lacks native int4/int8 hardware acceleration -- operations are emulated via upcast to BF16/FP32. MLX provides better native quantization performance on Apple Silicon.

**Primary recommendation:** Use Quanto int8 for the transformer/UNet backbone (MPS-compatible, minimal quality loss) with GGUF Q8_0 as an alternative for pre-quantized model distribution. Reserve int4 for memory-constrained scenarios only, as diffusion models show sharper quality degradation below 8-bit compared to LLMs.

---

## GPTQ Quantization

### Overview
GPTQ (Generative Pre-trained Transformer Quantization) is a one-shot post-training quantization method based on approximate second-order information. Originally published at ICLR 2023, it can quantize models with 175B+ parameters in ~4 GPU hours with negligible accuracy degradation at 4-bit.

### Technical Approach
- Uses layer-wise quantization with second-order (Hessian) information
- Calibrates each layer independently using a small calibration dataset
- Supports 2/3/4/8-bit weight quantization
- Primarily weight-only quantization (activations remain in higher precision)

### Apple Silicon Support
**GPTQModel** (successor to AutoGPTQ) provides explicit Apple Silicon support:
- Apple CPU backend with hardware-optimized kernels
- MPS (Metal Performance Shaders) backend support
- macOS platform builds available
- Supports multimodal models including Qwen2-VL and Ovis1.6-VL

**Limitations:**
- GPTQ is primarily optimized for CUDA tensor cores -- Apple Silicon runs via CPU fallback or emulated MPS ops
- Calibration step requires representative data from the target domain
- Linux-focused development; macOS is secondary

### Applicability to RealRestorer
- **Applicable to:** Transformer/DiT backbone components, text encoders (CLIP/T5)
- **Less suitable for:** VAE encoder/decoder (small model, quantization overhead not justified)
- **Quality at 4-bit:** ~1.6% accuracy drop on language benchmarks; vision model degradation likely higher for pixel-level tasks like restoration

### Sources
- [GPTQ Paper (arXiv)](https://arxiv.org/abs/2210.17323)
- [GPTQModel GitHub](https://github.com/ModelCloud/GPTQModel) (1,075 stars, last push 2026-03-28)
- [HuggingFace GPTQ Docs](https://huggingface.co/docs/transformers/en/quantization/gptq)

---

## AWQ (Activation-aware Weight Quantization)

### Overview
AWQ (MLSys 2024 Best Paper Award) is a hardware-friendly low-bit weight-only quantization method from MIT Han Lab. It identifies that only ~1% of weights are "salient" based on activation distributions, and applies an equivalent mathematical transformation to protect those channels during quantization.

### Technical Approach
- Does NOT rely on backpropagation or reconstruction
- Identifies salient weight channels by observing activation magnitude distributions
- Applies per-channel scaling to protect important weights before quantization
- Generalizes across domains and modalities without overfitting calibration data

### Latest Implementation (2025-2026)
- DeepSeek-R1-Distilled model support (April 2025)
- BF16 precision support (February 2025)
- TinyChat 2.0 with 1.5-1.7x faster prefilling (October 2024)
- Active support for multi-modal LMs (VLMs)

### Apple Silicon Support
- **Limited direct support** -- AWQ kernels are primarily CUDA-optimized
- AWQ 4-bit via vLLM supported on AMD ROCm but not Metal/MPS
- CPU fallback available but without hardware acceleration
- GPTQModel includes AWQ TorchFusedAWQ kernel but optimized for Intel/AMD CPU

### Quality Metrics
- Only 0.7% accuracy drop at 4-bit on Llama 3.1 8B (vs 1.6% for GGUF Q4_K_M)
- Best-in-class quality preservation due to activation-aware approach
- Particularly strong for "coding/creative assistants where coherence matters"

### Applicability to RealRestorer
- **Best candidate for:** Pre-quantizing transformer backbone weights for distribution
- **Advantage:** Activation-aware approach may better preserve the fine-grained detail needed for image restoration
- **Challenge:** No native MPS acceleration; would need GGUF or Quanto for runtime on Apple Silicon

### Sources
- [AWQ Paper (arXiv)](https://arxiv.org/abs/2306.00978)
- [MIT Han Lab AWQ GitHub](https://github.com/mit-han-lab/llm-awq) (3,479 stars, last push 2025-07-17)
- [AWQ Project Page](https://hanlab.mit.edu/projects/awq)

---

## Google TurboQuant

### Overview
TurboQuant is a compression algorithm from Google Research (presenting at ICLR 2026) that reduces key-value cache memory by at least 6x and delivers up to 8x speedup with zero accuracy loss. It is primarily a KV cache compression technique, not a weight quantization method.

### Technical Approach
1. **PolarQuant:** Converts vector pairs into polar coordinates (length + angle), eliminating per-block scaling constants
2. **QJL (Quantized Johnson-Lindenstrauss):** Applies 1-bit error correction on residual error to maintain accurate attention scores
3. Works without retraining or fine-tuning -- drops in under existing models
4. Compresses KV cache to as few as 3 bits per value

### Apple Silicon Compatibility (Already Working)
Three community implementations exist as of March 2026:

1. **turboquant_mlx** -- Native Apple Silicon via MLX framework
   - 1-3 bit extreme KV cache compression
   - Asymmetric PolarQuant caching
   - OpenAI-compatible server
   - Tested with Qwen 3.5 35B on M5 Max
   - GitHub: 20 stars, very active (last push 2026-03-25)

2. **turboquant_plus** -- llama.cpp/Metal integration
   - turbo3 and turbo4 KV-cache types
   - End-to-end serving on Apple Silicon
   - Prefill throughput at q8_0 parity with 4.6x KV cache compression
   - GitHub: 481 stars, very active (last push 2026-03-29)

3. **HuggingFace integration** -- Community models available (e.g., flovflo/turboquant-mlx-qwen35-kv)

### Applicability to RealRestorer
- **Directly applicable:** If RealRestorer uses transformer/DiT models with attention-based KV caches
- **Not applicable to:** UNet-only architectures without explicit KV caching, or VAE components
- **Key benefit:** Enables longer context / larger batch processing within 64GB memory budget
- **Timeline:** Mainstream tooling support expected Q2 2026

### Sources
- [Google Research Blog](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)
- [VentureBeat Coverage](https://venturebeat.com/infrastructure/googles-new-turboquant-algorithm-speeds-up-ai-memory-8x-cutting-costs-by-50/)
- [turboquant_mlx GitHub](https://github.com/helgklaizar/turboquant_mlx)
- [turboquant_plus GitHub](https://github.com/TheTom/turboquant_plus)
- [Google Research @X announcement](https://x.com/GoogleResearch/status/2036533564158910740)

---

## GGUF Format

### Overview
GGUF (GPT-Generated Unified Format) originated in the llama.cpp LLM ecosystem but has been widely adopted for diffusion models including FLUX, SDXL, Stable Diffusion, and image generation models. It is a file format for storing quantized model weights with flexible block-level quantization.

### Quantization Levels Available
| Level | Bits | Compression | Quality | Use Case |
|-------|------|-------------|---------|----------|
| Q8_0 | 8-bit | 2x | Near-lossless | Production quality |
| Q6_K | 6-bit | 2.7x | Excellent | Balanced |
| Q5_K_S/M | 5-bit | 3.2x | Very good (~95% quality) | Good balance |
| Q4_K_S/M | 4-bit | 4x | Good | Memory-constrained |
| Q4_1 | 4-bit | 4x | Good | Simple 4-bit |
| Q3_K | 3-bit | 5.3x | Acceptable | Tight memory |
| Q2_K | 2-bit | 8x | Noticeable artifacts | Extreme compression |

### Diffusion Model Benchmarks (Flux-dev via HuggingFace Diffusers)
| GGUF Level | Memory (loaded) | Peak Memory | Inference Time |
|------------|----------------|-------------|----------------|
| Q8_0 | 21.5 GB | 26.0 GB | 15 sec |
| Q4_1 | 16.8 GB | 21.3 GB | 23 sec |
| Q2_K | 13.3 GB | 17.8 GB | 26 sec |

Baseline BF16: 31.4 GB loaded, 36.2 GB peak, 12 sec inference.

### Apple Silicon Support
- **Metal backend** supported via llama.cpp and ComfyUI-GGUF
- Recommended format for Apple Silicon: Q4_K_M with "Metal backend + CPU fallback"
- ComfyUI-GGUF has 3,426 stars and active development (last push 2026-01-12)
- GGUF is the most portable format -- works on CPU, CUDA, Metal, ROCm

### Integration with HuggingFace Diffusers
```python
from diffusers import FluxTransformer2DModel, GGUFQuantizationConfig

transformer = FluxTransformer2DModel.from_single_file(
    "flux1-dev-Q4_1.gguf",
    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
    torch_dtype=torch.bfloat16,
)
```

### Applicability to RealRestorer
- **Strong candidate for:** Model distribution format (users download pre-quantized GGUF files)
- **Advantage:** Universal hardware compatibility, well-tested with diffusion models
- **Advantage:** Direct diffusers integration via `from_single_file` and `GGUFQuantizationConfig`
- **Consideration:** Q8_0 recommended for restoration tasks where pixel-level fidelity matters

### Sources
- [HuggingFace Diffusers GGUF Docs](https://huggingface.co/docs/diffusers/quantization/gguf)
- [ComfyUI-GGUF GitHub](https://github.com/city96/ComfyUI-GGUF) (3,426 stars)
- [HuggingFace Diffusers Quantization Blog](https://huggingface.co/blog/diffusers-quantization)
- [FLUX GGUF Guide](https://apatero.com/blog/flux-gguf-quantization-8gb-vram-guide-2026)

---

## bitsandbytes on Apple Silicon

### Current Status (March 2026)
**Experimental / Early Stage.** bitsandbytes has 8,081 GitHub stars and is very active (last push 2026-03-27), but Apple Silicon support remains limited.

### MPS Backend Development
- PR #1853 adds MPS backend by delegating to external `mps-bitsandbytes` package
- Provides NF4/FP4/FP8/INT8 quantization for PyTorch on Apple Silicon with Metal GPU acceleration
- **Security concern:** The `mps-bitsandbytes` package was flagged for supply chain risks (unlicensed, very new publisher)
- Apple Silicon support was planned for Q4/2024 - Q2/2025 but timeline slipped

### Available Quantization on Apple Silicon
| Method | Status | Notes |
|--------|--------|-------|
| NF4 (4-bit) | Via mps-bitsandbytes | Experimental |
| FP4 (4-bit) | Via mps-bitsandbytes | Experimental |
| FP8 (8-bit) | Via mps-bitsandbytes | Experimental |
| INT8 (8-bit) | Via mps-bitsandbytes | Experimental |
| 8-bit optimizers | Via mps-bitsandbytes | Adam, RMSprop, Lion, Momentum, AdEMAMix |

### Diffusion Model Benchmarks (CUDA, for reference)
| Precision | Memory (loaded) | Peak Memory | Inference Time |
|-----------|----------------|-------------|----------------|
| NF4 (4-bit) | 12.6 GB | 17.3 GB | 12 sec |
| INT8 (8-bit) | 19.3 GB | 24.4 GB | 27 sec |

NF4 is notable for matching BF16 inference speed while cutting memory by 60%.

### Applicability to RealRestorer
- **Not recommended as primary path** due to experimental MPS support and supply chain concerns
- **Useful for:** Reference benchmarking on CUDA if cross-platform support is needed
- **Alternative:** Use Quanto for MPS-safe quantization with similar capabilities

### Sources
- [bitsandbytes GitHub](https://github.com/bitsandbytes-foundation/bitsandbytes) (8,081 stars)
- [MPS Backend PR #1853](https://github.com/bitsandbytes-foundation/bitsandbytes/pull/1853)
- [Apple Silicon Support Issue #252](https://github.com/bitsandbytes-foundation/bitsandbytes/issues/252)
- [Multi-backend Discussion #1340](https://github.com/bitsandbytes-foundation/bitsandbytes/discussions/1340)

---

## PyTorch Native Quantization

### torch.ao (torchao)

**Status:** Functional but limited on MPS.

MPS executes int4/int8 quantized operations as emulated (upcast to BF16/FP32 for computation). Apple Silicon has no equivalent to NVIDIA's NVFP4 Tensor Cores. Using float8 on MPS raises an error.

**Diffusion Model Benchmarks (CUDA):**
| Method | Memory (loaded) | Peak Memory | Inference Time |
|--------|----------------|-------------|----------------|
| int4_weight_only | 10.6 GB | 14.7 GB | 109 sec* |
| int8_weight_only | 17.0 GB | 21.5 GB | 15 sec |
| float8_weight_only | 17.0 GB | 21.5 GB | 15 sec |

*int4 is slow without torch.compile. With torch.compile: 6 sec (but 285 sec compile time).

**MPS Limitations:**
- torch.compile support for MPS is incomplete -- complex fusions often fallback to CPU
- No native int4/int8 Metal kernels
- float8 not supported on MPS at all

### Quanto (optimum-quanto)

**Status:** Best MPS-compatible option for PyTorch quantization.

Key advantages:
- **Device agnostic:** Works on CPU, CUDA, and MPS (Apple Silicon)
- **torch.compile friendly** (where compile is supported)
- Supports int2, int4, int8, and float8 weights and activations
- No calibration data required -- quantizes on the fly

**Diffusion Model Benchmarks (CUDA):**
| Precision | Memory (loaded) | Peak Memory | Inference Time |
|-----------|----------------|-------------|----------------|
| INT4 | 12.3 GB | 16.1 GB | 109 sec |
| INT8 | 17.3 GB | 21.8 GB | 15 sec |
| FP8 | 16.4 GB | 20.9 GB | 16 sec |

**MPS-specific notes:**
- float8 will error on MPS devices
- int8 and int4 work via emulation (no hardware acceleration)
- Still the safest choice for MPS compatibility

**Usage with Diffusers:**
```python
from diffusers import QuantoConfig as DiffusersQuantoConfig
from transformers import QuantoConfig as TransformersQuantoConfig
from diffusers.quantizers import PipelineQuantizationConfig

pipeline_quant_config = PipelineQuantizationConfig(
    quant_mapping={
        "transformer": DiffusersQuantoConfig(weights_dtype="int8"),
        "text_encoder_2": TransformersQuantoConfig(weights_dtype="int8"),
    }
)
```

### SmoothQuant

**Status:** Relevant for W8A8 (8-bit weight + 8-bit activation) quantization.

SmoothQuant (ICML 2023, MIT Han Lab) addresses activation outliers in transformer models by offline migrating quantization difficulty from activations to weights via a mathematically equivalent transformation. Achieves 1.56x speedup and 2x memory reduction with negligible accuracy loss.

**Relevance to RealRestorer:**
- Useful if both weights AND activations need quantization (W8A8)
- Complementary to weight-only methods like AWQ/GPTQ
- No native MPS support; primarily CUDA via TensorRT

### Sources
- [PyTorch MPS Documentation](https://developer.apple.com/metal/pytorch/)
- [Quanto Introduction (HuggingFace)](https://huggingface.co/blog/quanto-introduction)
- [optimum-quanto GitHub](https://github.com/huggingface/optimum-quanto) (1,035 stars)
- [SmoothQuant GitHub](https://github.com/mit-han-lab/smoothquant)
- [State of PyTorch Hardware Acceleration 2025](https://tunguz.github.io/PyTorch_Hardware_2025/)

---

## Quality vs Speed Tradeoffs

### By Bit Width

| Bit Width | Memory Reduction | Quality Impact | Best For |
|-----------|-----------------|----------------|----------|
| **FP16/BF16** | Baseline | None | Reference / full quality |
| **FP8** | 50% from FP16 | Minimal | Production with Hopper/Ada GPUs |
| **INT8 / Q8_0** | 50% from FP16 | Near-lossless | **Recommended default for restoration** |
| **INT4 / NF4 / Q4_K** | 75% from FP16 | Moderate; acceptable for generation | Memory-constrained deployment |
| **INT2 / Q2_K** | 87.5% from FP16 | Substantial; visible artifacts | Not recommended for restoration |

### Diffusion Model Specifics

The quality-performance relationship is **non-linear** for diffusion models:
- FP32 -> FP16/INT8: Minimal perceptual degradation
- INT8 -> INT4: **Sharp quality cliff** for pixel-level tasks
- INT4 -> INT2: Severe degradation with visible artifacts

**Critical insight:** Diffusion models used for image restoration have stricter quality requirements than image generation. A generation model at Q4 may produce "acceptable" outputs, but a restoration model at Q4 may introduce artifacts that defeat the purpose of restoration.

### Component Sensitivity

Not all components are equally sensitive to quantization:

| Component | Sensitivity | Safe Minimum | Notes |
|-----------|-------------|-------------|-------|
| **Transformer/UNet backbone** | Medium | INT8 / Q8_0 | Largest component; most memory savings here |
| **Text Encoder (T5/CLIP)** | Low | INT4 / NF4 | Text encoding tolerates aggressive quantization |
| **VAE** | **High** | FP16 (no quantization) | Small model (~168 MB); quantization adds artifacts to pixel-level output |

### Techniques for Mitigating Quality Loss at Lower Bit Widths
1. **AWQ:** Activation-aware scaling protects salient 1% of weights
2. **SmoothQuant:** Migrates quantization difficulty from activations to weights
3. **Mixed precision:** Different bit widths per layer based on sensitivity analysis
4. **NF4 (Normal Float 4-bit):** Information-theoretically optimal 4-bit data type
5. **Calibration data:** Domain-specific calibration (restoration images) improves accuracy

---

## Component-Level Quantization Strategy

### Architecture: Diffusion Model Pipeline

```
Text Encoder (CLIP/T5) -> Transformer/UNet -> VAE Decoder -> Output Image
     ~10 GB                  ~24 GB            ~168 MB
```

### Recommended Quantization Map

```python
PipelineQuantizationConfig(
    quant_mapping={
        # Largest component -- quantize aggressively for memory savings
        "transformer": QuantoConfig(weights_dtype="int8"),  # 24 GB -> ~12 GB

        # Text encoder tolerates lower precision well
        "text_encoder_2": QuantoConfig(weights_dtype="int4"),  # 10 GB -> ~2.5 GB

        # VAE is tiny and quality-critical -- DO NOT quantize
        # "vae": None  (keep at BF16/FP16)
    }
)
```

**Expected memory profile:**
- Baseline BF16: ~31.4 GB loaded, ~36.2 GB peak
- With quantization: ~14.7 GB loaded, ~19 GB peak
- Savings: ~53% memory reduction

### Combined Optimization Options

For maximum memory efficiency on 64GB Apple Silicon:

1. **Quantization + CPU offloading:**
   - Quantize transformer to int8 + enable_model_cpu_offload()
   - Peak memory: ~12.4 GB (demonstrated with bitsandbytes NF4)

2. **FP8 layerwise casting + group offloading:**
   - Peak memory: ~14.2 GB with 58 sec inference
   - Viable on MPS if float8 support lands

3. **torchao int4 + torch.compile (CUDA only):**
   - 6 sec inference but requires ~285 sec compile warmup
   - Not viable on MPS due to limited torch.compile support

---

## GitHub Library Activity

| Library | Stars | Last Push | Apple Silicon | Notes |
|---------|-------|-----------|---------------|-------|
| [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) | 8,081 | 2026-03-27 | Experimental (mps-bitsandbytes) | Most popular; MPS WIP |
| [AWQ (llm-awq)](https://github.com/mit-han-lab/llm-awq) | 3,479 | 2025-07-17 | No native | MLSys Best Paper; CUDA-focused |
| [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) | 3,426 | 2026-01-12 | Yes (Metal) | Best diffusion GGUF support |
| [GPTQ (original)](https://github.com/IST-DASLab/gptq) | 2,274 | 2024-03-27 | No | Academic reference; inactive |
| [GPTQModel](https://github.com/ModelCloud/GPTQModel) | 1,075 | 2026-03-28 | Yes (CPU + MPS) | Active; multimodal support |
| [optimum-quanto](https://github.com/huggingface/optimum-quanto) | 1,035 | 2025-11-21 | Yes (MPS) | Best MPS compatibility |
| [turboquant_plus](https://github.com/TheTom/turboquant_plus) | 481 | 2026-03-29 | Yes (Metal) | KV cache; very active |
| [turboquant_mlx](https://github.com/helgklaizar/turboquant_mlx) | 20 | 2026-03-25 | Yes (MLX native) | Early stage; MLX native |

---

## Recommendations for RealRestorer

### Tier 1: Implement Now

1. **Quanto INT8 for transformer backbone** -- MPS-compatible, minimal quality loss, no calibration needed, direct diffusers integration. This is the safest path for Apple Silicon.

2. **GGUF Q8_0 for model distribution** -- Pre-quantize models in GGUF format for users to download. Universal compatibility (Metal, CPU, CUDA). Established ecosystem via ComfyUI-GGUF and HuggingFace diffusers.

3. **No VAE quantization** -- Keep VAE at FP16/BF16. It is only ~168 MB and directly affects pixel-level output quality.

### Tier 2: Implement When Mature

4. **TurboQuant for KV cache** (if using DiT/transformer architecture) -- Already working on Apple Silicon via MLX and llama.cpp. Monitor turboquant_plus (481 stars, actively developed). Will matter most for batch processing or high-resolution inputs.

5. **Mixed-precision quantization** -- INT4 for text encoders, INT8 for transformer, FP16 for VAE. Reduces total pipeline memory from ~31 GB to ~15 GB.

### Tier 3: Monitor / Avoid

6. **bitsandbytes on MPS** -- Wait for official Apple Silicon support. The mps-bitsandbytes external package has supply chain concerns. Revisit when bitsandbytes natively supports MPS.

7. **AWQ/GPTQ direct** -- Excellent quality but CUDA-optimized. Use GPTQModel if CPU fallback is acceptable, but prefer Quanto for MPS runtime.

8. **torch.ao on MPS** -- Emulated operations negate the speed benefits. Wait for native Metal int8 kernels or use MLX instead.

9. **INT4/INT2 for restoration backbone** -- Quality degradation too severe for pixel-level restoration tasks. Only use for text encoders or non-critical components.

### Decision Matrix

| Scenario | Recommended Approach |
|----------|---------------------|
| Default (quality-first) | Quanto INT8 transformer + FP16 VAE |
| Memory-constrained (<16 GB) | GGUF Q4_K_M transformer + Quanto INT4 text encoder + FP16 VAE |
| Maximum throughput | Quanto INT8 + CPU offloading |
| Model distribution | Pre-quantized GGUF Q8_0 files on HuggingFace |
| Future (Q2+ 2026) | TurboQuant KV cache + Quanto INT8 weights |
