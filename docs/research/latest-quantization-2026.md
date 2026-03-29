# Latest Quantization Techniques for Diffusion Models on Apple Silicon

**Research Date:** 2026-03-29
**Scope:** Quantization methods for running large vision/diffusion models efficiently on Apple Silicon (MPS/MLX/ANE)

---

## Table of Contents

1. [Google TurboQuant](#1-google-turboquant)
2. [TorchAO (PyTorch AO)](#2-torchao-pytorch-ao)
3. [GGUF for Diffusion Models](#3-gguf-for-diffusion-models)
4. [MLX Quantization](#4-mlx-quantization)
5. [Quanto (optimum-quanto) on MPS](#5-quanto-optimum-quanto-on-mps)
6. [Apple ml-ane-transformers](#6-apple-ml-ane-transformers)
7. [Draw Things + Metal FlashAttention](#7-draw-things--metal-flashattention)
8. [AQLM, QuIP#, SqueezeLLM](#8-aqlm-quip-squeezellm)
9. [SpinQuant (Meta)](#9-spinquant-meta)
10. [Q-DiT: Diffusion Transformer Quantization](#10-q-dit-diffusion-transformer-quantization)
11. [bitsandbytes MPS Backend](#11-bitsandbytes-mps-backend)
12. [Apple Intelligence Foundation Models Quantization](#12-apple-intelligence-foundation-models-quantization)
13. [Summary Matrix](#13-summary-matrix)
14. [Recommendations for realrestore-cli](#14-recommendations-for-realrestore-cli)

---

## 1. Google TurboQuant

**Status:** Published March 2026 | ICLR 2026
**Paper:** https://arxiv.org/abs/2504.19874
**Blog:** https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/
**Official Code:** No official repository released yet
**Apple Silicon:** Not tested/targeted (H100-focused benchmarks)

### What It Is

TurboQuant is a **KV-cache compression** algorithm (not a weight quantization method). It quantizes the key-value cache to 3 bits with zero accuracy loss and no training/calibration required. It is a two-stage pipeline combining PolarQuant and QJL (Quantized Johnson-Lindenstrauss).

### How It Works

**Stage 1 -- PolarQuant (AISTATS 2026):**
- Converts vectors from Cartesian to polar coordinates (radius + angle)
- Radius captures data strength; angle captures data direction/meaning
- Angle patterns are concentrated and predictable, eliminating expensive normalization
- Applies standard scalar quantization to each component individually
- Paper: https://arxiv.org/abs/2502.02617

**Stage 2 -- QJL:**
- Takes the residual error from PolarQuant
- Projects through a random Gaussian matrix (Johnson-Lindenstrauss Transform)
- Stores only the sign (+1 or -1) of each projection = exactly 1 bit per dimension
- Zero memory overhead; eliminates inner-product bias
- Paper: https://dl.acm.org/doi/10.1609/aaai.v39i24.34773

### Performance Numbers

| Metric | Result |
|--------|--------|
| KV cache compression | 6x memory reduction |
| Bit rate | 3-bit KV cache, zero accuracy loss |
| 4-bit TurboQuant vs 32-bit | Up to 8x performance increase (H100) |
| Models tested | Gemma, Mistral |
| Benchmarks | LongBench, Needle In A Haystack, ZeroSCROLLS, RULER, L-Eval |

### Community Implementations

| Repo | Language | Notes |
|------|----------|-------|
| [tonbistudio/turboquant-pytorch](https://github.com/tonbistudio/turboquant-pytorch) | Python/PyTorch | From-scratch implementation, 5x compression at 3-bit |
| [RecursiveIntell/turbo-quant](https://github.com/RecursiveIntell/turbo-quant) | Rust | PolarQuant + QJL residual sketch |
| [OnlyTerp/turboquant](https://github.com/OnlyTerp/turboquant) | Python | First open-source implementation |
| [TheTom/turboquant_plus](https://github.com/TheTom/turboquant_plus) | Python | Based on Google's paper |

### Relevance to Diffusion Models on Apple Silicon

**Low.** TurboQuant targets KV-cache in LLM inference, not diffusion model weights or activations. However, the PolarQuant stage's data-oblivious approach could theoretically be adapted for quantizing attention layers in DiT (Diffusion Transformer) architectures like FLUX. The llama.cpp community is already discussing integration: https://github.com/ggml-org/llama.cpp/issues/20979

---

## 2. TorchAO (PyTorch AO)

**Status:** Active development | Paper accepted at CodeML @ ICML 2025
**Repo:** https://github.com/pytorch/ao
**Docs:** https://docs.pytorch.org/ao/stable/workflows/inference.html
**Apple Silicon:** Partial (ARM CPU kernels, limited MPS GPU acceleration)

### What It Offers

TorchAO is PyTorch's official quantization and sparsity library. Supports INT4, INT8, FP8, MXFP4, MXFP6, MXFP8 data types plus 2:4 sparsity. Integrates with HuggingFace Transformers, vLLM, Diffusers, and ExecuTorch.

### Apple Silicon Support Details

**ARM CPU Kernels (January 2025):**
- 1-8 bit ARM CPU kernels for linear and embedding ops
- 8-bit dynamic quantization of activations
- uintx groupwise quantization of weights
- Experimental, runs on any ARM CPU (including Apple Silicon Macs)

**MPS GPU Support:**
- `int4_weight_only` can be used on Mac, BUT MPS **emulates** operations
- Low-bit ops are **upcast to BF16 or Float32** for computation on MPS
- No native Metal kernels for quantized matmul
- No Apple-specific installation variant (CPU-only or CUDA)

### Usage on Apple Silicon

```python
import torch
import torchao

# INT8 weight-only quantization (works on ARM CPU)
from torchao.quantization import int8_weight_only
model = torchao.quantize_(model, int8_weight_only())

# INT4 weight-only (works but emulated on MPS)
from torchao.quantization import int4_weight_only
model = torchao.quantize_(model, int4_weight_only())
# Note: MPS will upcast to float32 for computation
```

### Key Limitation

**There is no true MPS acceleration for quantized ops.** The quantization reduces memory usage (which helps on unified-memory Macs), but does not provide the ~2x compute speedup you get on CUDA with dedicated INT4/INT8 kernels. The ARM CPU kernels give actual speedup for CPU-bound workloads.

---

## 3. GGUF for Diffusion Models

**Status:** Production-ready for FLUX/SD3 | Active development
**Key Repos:**
- ComfyUI-GGUF: https://github.com/city96/ComfyUI-GGUF
- HF Diffusers GGUF: https://huggingface.co/docs/diffusers/quantization/gguf
- ComfyUI-ModelQuantizer: https://github.com/lum3on/ComfyUI-ModelQuantizer
**Apple Silicon:** Works (with caveats on macOS Sequoia)

### How GGUF Works for Diffusion

GGUF stores quantized weights in a single memory-mapped binary file. Originally designed for LLMs (llama.cpp), it has been adapted for diffusion model weights. Key insight: **transformer/DiT architectures (FLUX, SD3) are far less affected by quantization than UNet-based models (SDXL, SD1.5)**.

Weights are stored in low-memory dtype (typically `torch.uint8`) and dynamically dequantized during each forward pass to a configured `compute_dtype`.

### Supported Quantization Types

| Type | Bits | Quality | Notes |
|------|------|---------|-------|
| Q8_0 | 8 | Excellent | Sweet spot for FLUX |
| Q6_K | 6 | Very good | K-quant, layer-aware |
| Q5_K | 5 | Good | K-quant |
| Q5_0 / Q5_1 | 5 | Good | Legacy |
| Q4_K | 4 | Acceptable | K-quant, recommended balance |
| Q4_0 / Q4_1 | 4 | Acceptable | Legacy |
| Q3_K | 3 | Noticeable loss | K-quant |
| Q2_K | 2 | Significant loss | Extreme compression |

K-quant variants use layer-importance-aware quantization: critical layers get higher precision, less important layers get more compression.

### Usage with HuggingFace Diffusers

```python
import torch
from diffusers import FluxPipeline, FluxTransformer2DModel, GGUFQuantizationConfig

ckpt_path = "https://huggingface.co/city96/FLUX.1-dev-gguf/blob/main/flux1-dev-Q4_K.gguf"

transformer = FluxTransformer2DModel.from_single_file(
    ckpt_path,
    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
    torch_dtype=torch.bfloat16,
)

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    transformer=transformer,
    torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()

image = pipe("A cat holding a sign", generator=torch.manual_seed(0)).images[0]
```

### macOS/Apple Silicon Notes

- On **macOS Sequoia**: torch 2.4.1 required; 2.6.x nightly causes "M1 buffer is not large enough" error
- GGUF loading works on MPS, but optimized CUDA kernels (via `kernels` package) are CUDA-only
- Quantization itself requires significant RAM (96GB+ for large models)
- Pre-quantized GGUF files available on HuggingFace (no need to quantize locally)

### Supported Diffusion Models

- FLUX.1-dev and FLUX.1-schnell (transformer/DiT architecture)
- Stable Diffusion 3.x (transformer architecture)
- SDXL (UNet, more quality loss from quantization)
- SD 1.x (UNet, more quality loss)
- T5 text encoder (quantized separately)

---

## 4. MLX Quantization

**Status:** Rapidly evolving | Strategic Apple initiative
**Repo:** https://github.com/ml-explore/mlx
**WWDC25:** https://developer.apple.com/videos/play/wwdc2025/298/
**Apple Silicon:** Native (purpose-built)

### MLX Quantization for LLMs

MLX natively supports weight quantization via `mlx_lm.convert`. A 7B model can be quantized to 4-bit in seconds:

```bash
# Quantize a model to 4-bit
python -m mlx_lm.convert \
    --hf-path mistralai/Mistral-7B-v0.1 \
    --mlx-path mlx-model-4bit \
    -q --q-bits 4
```

### MLX for Diffusion Models

**DiffusionKit** (https://github.com/argmaxinc/DiffusionKit):
- Python + Swift packages for on-device diffusion with Core ML and MLX
- Supports: Stable Diffusion 3 (2B), FLUX.1-schnell, FLUX.1-dev
- Pre-quantized models on HuggingFace:
  - `argmaxinc/mlx-stable-diffusion-3.5-large-4bit-quantized`
  - `argmaxinc/mlx-FLUX.1-schnell-4bit-quantized`

```python
from diffusionkit.mlx import DiffusionPipeline

pipeline = DiffusionPipeline(
    shift=3.0,
    model_version="argmaxinc/mlx-stable-diffusion-3-medium",
    low_memory_mode=True,  # For constrained devices
    a16=True,   # 16-bit activations
    w16=True,   # 16-bit weights
)
image, _ = pipeline.generate_image("a photo of a cat", num_steps=50)
```

**mflux** (https://github.com/filipstrand/mflux):
- MLX-native implementations of state-of-the-art generative image models
- Supports 7 model families: Z-Image (6B), FLUX.2 (4B/9B), FIBO (8B), SeedVR2 (3B/7B), Qwen Image (20B), Depth Pro, FLUX.1 (12B)
- 4-bit and 8-bit quantization support
- Pre-quantized models on HuggingFace eliminate manual quantization

```bash
# Generate with 8-bit quantization
mflux-generate-z-image-turbo \
    --prompt "A puffin standing on a cliff" \
    --width 1280 --height 500 \
    --seed 42 --steps 9 -q 8

# Save quantized model for reuse
mflux-save --model flux1-schnell -q 4 --path ./flux-4bit
```

### M5 Performance (2025-2026)

- FLUX-dev-4bit (12B) on M5 is **3.8x faster** than on M4
- MLX now supports CUDA GPUs in addition to Apple Silicon
- Metal FlashAttention v2.5 with Neural Accelerators on M5

### Chip-Specific Quantization Guidelines

| Chip | Recommended Quant | Framework |
|------|-------------------|-----------|
| M1/M2 (16GB) | 4-bit | mflux, DiffusionKit |
| M1/M2 Pro (32GB) | 4-bit or 8-bit | mflux, DiffusionKit |
| M2 Ultra (192GB) | Full precision | mflux |
| M3 Pro | 4-bit | mflux, DiffusionKit |
| M4 Pro | 4-8 bit | mflux, DiffusionKit |
| M5 | 4-8 bit | mflux, DiffusionKit |

---

## 5. Quanto (optimum-quanto) on MPS

**Status:** MAINTENANCE MODE (no major new features)
**Repo:** https://github.com/huggingface/optimum-quanto
**PyPI:** https://pypi.org/project/optimum-quanto/
**Apple Silicon:** Deploys on MPS, but NO accelerated Metal kernels

### MPS Support Reality

The documentation claims "quantized models can be placed on any device (including CUDA and MPS)." However:

- **CUDA:** Accelerated kernels for int8-int8, fp16-int4, bf16-int8, bf16-int4
- **MPS:** No equivalent accelerated kernels documented or implemented
- Operations on MPS likely **fall back to dequantize-then-compute in float**

### Supported Quantization

| Weight Type | Activation Type | CUDA Accel | MPS Accel |
|-------------|-----------------|------------|-----------|
| float8 | float8 | Yes | No |
| int8 | int8 | Yes | No |
| int4 | none | Yes | No |
| int2 | none | Partial | No |

### Verdict

**Quanto on MPS provides memory savings but NOT compute acceleration.** The library is in maintenance mode. HuggingFace recommends migrating to bitsandbytes or torchao for active development.

```python
from optimum.quanto import quantize, freeze
from optimum.quanto import qint8

# This works on MPS but won't be faster than float16
quantize(model, weights=qint8)
freeze(model)
model.to("mps")  # Memory reduced, but no Metal kernel speedup
```

---

## 6. Apple ml-ane-transformers

**Status:** STALE (last commit August 2022, v0.1.3)
**Repo:** https://github.com/apple/ml-ane-transformers
**Vision Transformers:** https://github.com/apple/ml-vision-transformers-ane
**Apple Silicon:** Native ANE optimization (A14+, M1+)

### What It Is

Reference implementation of Transformers optimized for Apple Neural Engine (ANE). Demonstrates up to 10x faster and 14x lower peak memory compared to baseline implementations. 602/606 operations run on ANE for DistilBERT example.

### Key Optimizations

1. Replace linear layers with convolution layers (ANE runs conv ops well)
2. Use local attention blocks for spatial efficiency
3. Restructure operations to maximize ANE utilization

### Vision Transformers on ANE

**ml-vision-transformers-ane** (https://github.com/apple/ml-vision-transformers-ane):
- Efficient attention module for vision transformers on ANE
- DeiT and MOAT architectures optimized
- Window partitioning for efficient attention
- Optimized MOAT is "multiple times faster" than third-party implementations

### Quantization Support

- No explicit quantization in the reference implementations
- The research papers mention quantization and pruning as complementary optimization strategies
- Apple's internal models use 2-bit QAT (see Section 12 below)

### Relevance to Diffusion Models

**Low-medium.** These repos target classification/NLP transformers, not diffusion. However, the optimization patterns (conv replacement, attention restructuring) are directly applicable to diffusion transformer (DiT) architectures when converting to Core ML for ANE deployment.

### Community Extension

**more-ane-transformers** (https://github.com/smpanaro/more-ane-transformers):
- Extends Apple's work to run LLMs on ANE
- More actively maintained than Apple's official repos

---

## 7. Draw Things + Metal FlashAttention

**Status:** Most performant diffusion framework on Apple Silicon
**App:** https://apps.apple.com/us/app/draw-things-offline-ai-art/id6444050820
**Engineering Blog:** https://engineering.drawthings.ai/
**Apple Silicon:** Native, deeply optimized

### Metal FlashAttention 2.0 and 2.5

Draw Things implements custom Metal FlashAttention kernels that are the fastest available for Apple Silicon diffusion inference:

- **20% faster** inference on M3/M4 vs earlier implementations
- **25% faster** than mflux for FLUX models
- **94% faster** than ggml-based implementations for FLUX
- **v2.5 with Neural Accelerators:** 4.6x improvement on M5 over M4
- Video generation: 5-second 480p video with Wan 2.2 A14B on M5 iPad (16GB RAM)

### Quantization in Draw Things

| Chip | Draw Things Quant | Notes |
|------|-------------------|-------|
| M3 Pro | 5-bit | Custom quantization |
| M4 Pro | 5-bit | Custom quantization |
| M5 | Higher precision | 3.5x over M4 via MPSGraph API |

### Supported Models

- Stable Diffusion 1.x, 2.x, XL
- Stable Diffusion 3.x
- FLUX.1-dev, FLUX.1-schnell
- AuraFlow
- Wan 2.2 (video generation)
- LoRA/LyCORIS fine-tuning support

### Why It Matters

Draw Things represents the **ceiling of what's achievable** on Apple Silicon for diffusion inference. Its Metal FlashAttention implementation is proprietary but demonstrates that Metal can achieve near-CUDA performance with the right kernel engineering. The blog posts provide architectural insights:

- Metal FlashAttention 2.0: https://engineering.drawthings.ai/p/metal-flashattention-2-0-pushing-forward-on-device-inference-training-on-apple-silicon-fe8aac1ab23c
- Neural Accelerator integration: https://releases.drawthings.ai/p/metal-flashattention-v25-w-neural

---

## 8. AQLM, QuIP#, SqueezeLLM

### AQLM (Additive Quantization for Language Models)

**Status:** LLM-focused, no diffusion/vision support
**Apple Silicon:** Not specifically targeted

AQLM uses additive quantization with learned codebooks for extreme compression (2-3 bit). Primarily used for language models. No documented use for vision or diffusion models. Supported by vLLM and Aphrodite inference engines but without Apple Silicon optimization.

### QuIP# (Quantization with Incoherence Processing, Sharp)

**Status:** ICLR 2024 | LLM-focused
**Repo:** https://github.com/Cornell-RelaxML/quip-sharp
**Paper:** https://arxiv.org/abs/2402.04396
**Apple Silicon:** CUDA-only (requires GPU capability > 7)

Three key innovations:
1. **Randomized Hadamard Transform** for incoherence processing (faster than QuIP's original)
2. **E8 lattice codebooks** for 8-dimensional vector quantization (optimal packing)
3. **Fine-tuning** for fidelity restoration

Performance: First PTQ method where 3-bit models scale better than 4-bit. Achieves 50%+ of peak memory bandwidth on NVIDIA RTX 4090.

**Verdict for Apple Silicon:** Not applicable. The E8 lattice codebook lookup requires custom CUDA kernels with no Metal equivalent. Would need significant porting effort.

### SqueezeLLM

**Status:** ICML 2024 | LLM-focused
**Repo:** https://github.com/SqueezeAILab/SqueezeLLM
**Apple Silicon:** No MPS/Metal support

Dense-and-Sparse quantization framework. Supports LLaMA family (7B-65B), LLaMA-2, Vicuna, XGen, OPT. No vision model or diffusion model support. CUDA-only implementation.

### Verdict

**None of AQLM, QuIP#, or SqueezeLLM are practically applicable to vision/diffusion models on Apple Silicon.** They are all LLM-specific, CUDA-dependent methods. The research insights (incoherence processing, lattice codebooks, dense-and-sparse patterns) could theoretically be adapted, but no such work exists.

---

## 9. SpinQuant (Meta)

**Status:** ICLR 2025 | Production (Meta Connect demo)
**Repo:** https://github.com/facebookresearch/SpinQuant
**Paper:** https://arxiv.org/abs/2405.16406
**License:** CC-BY-NC 4.0 (non-commercial)
**Apple Silicon:** No MPS support (requires CUDA + fast-hadamard-transform)

### How Learned Rotations Work

SpinQuant applies learned rotation matrices to weight/activation tensors before quantization:

1. Random rotations can remove outliers that hurt quantization, but quality varies wildly (up to 13 point difference)
2. SpinQuant optimizes rotation matrices using **Cayley parameterization on the Stiefel manifold**
3. The learned rotations specifically minimize activation outliers before quantization

### Quantization Configurations

| Config | Weights | Activations | KV Cache |
|--------|---------|-------------|----------|
| W4A16KV16 | 4-bit | 16-bit | 16-bit |
| W4A4KV16 | 4-bit | 4-bit | 16-bit |
| W4A4KV4 | 4-bit | 4-bit | 4-bit |

### Performance

LLaMA-2 7B W4A4KV4: accuracy gap narrows to **2.9 points** vs full precision, surpassing:
- LLM-QAT by 19.1 points
- SmoothQuant by 25.0 points

### Apple Silicon Compatibility

**Not compatible.** Key blocker:
- `fast-hadamard-transform` library requires CUDA
- PyTorch CUDA tensors throughout
- No Metal/MPS backend support
- ExecuTorch integration exists for edge deployment but targets mobile (non-Mac)

Pre-quantized models available on HuggingFace:
- `meta-llama/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8`
- `meta-llama/Llama-3.2-3B-Instruct-SpinQuant_INT4_EO8`

---

## 10. Q-DiT: Diffusion Transformer Quantization

**Status:** CVPR 2025 | Most relevant academic work for our use case
**Repo:** https://github.com/Juanerx/Q-DiT
**Paper:** https://arxiv.org/abs/2406.17343
**Project Page:** https://q-dit.github.io/
**Apple Silicon:** Not tested (CUDA-focused), but techniques are backend-agnostic

### Why This Matters

Q-DiT is the first dedicated post-training quantization method for Diffusion Transformers (DiTs). This directly applies to FLUX-architecture models.

### Key Techniques

1. **Automatic Quantization Granularity Allocation:** Uses evolutionary search to find optimal group sizes per layer, accounting for the significant spatial variance in DiT weights and activations
2. **Sample-wise Dynamic Activation Quantization:** Adaptively captures activation changes across both timesteps and samples (diffusion models have time-varying activations unlike LLMs)

### Results

| Setting | Model | Resolution | Result |
|---------|-------|------------|--------|
| W6A8 | DiT-XL/2 | 256x256 | FID reduced by 1.09 vs baseline |
| W4A8 | DiT-XL/2 | 256x256 | High fidelity maintained |

### Related Academic Work

- **QuEST** (ICCV 2025): Low-bit diffusion model quantization via efficient selective finetuning
  - Paper: https://openaccess.thecvf.com/content/ICCV2025/papers/Wang_QuEST_Low-bit_Diffusion_Model_Quantization_via_Efficient_Selective_Finetuning_ICCV_2025_paper.pdf
- **BinaryAttention** (2026): One-bit QK-attention for vision and diffusion transformers
  - Paper: https://arxiv.org/abs/2603.09582
- **VIDIT-Q** (ICLR 2025): Efficient diffusion transformer quantization
  - Paper: https://nicsefc.ee.tsinghua.edu.cn/nics_file/pdf/429cbac0-9463-4978-941b-c3c8ef5daa01.pdf

---

## 11. bitsandbytes MPS Backend

**Status:** In development (PR open, security concerns raised)
**PR:** https://github.com/bitsandbytes-foundation/bitsandbytes/pull/1853
**Issue:** https://github.com/bitsandbytes-foundation/bitsandbytes/issues/252
**Discussion:** https://github.com/bitsandbytes-foundation/bitsandbytes/discussions/1340

### Current Status

PR #1853 adds MPS backend via an external `mps-bitsandbytes` package:
- 8-bit optimizers (Adam, RMSprop, Lion, Momentum, AdEMAMix) using dynamic codebook quantization
- Metal kernels for 4-bit quantization/dequantization (NF4/FP4)
- `gemv_4bit` Metal kernel

### Security Concerns

The PR has NOT been merged due to architectural review issues:
- External `mps-bitsandbytes` package executes compiled native Metal shaders + Obj-C++ extension
- No license on the external package
- Only ~18 days old at time of review, insufficient trust
- Reviewers recommend in-tree implementation or plugin system instead of hard import

### Timeline

Apple Silicon MPS support was planned for Q4/2024 - Q2/2025 but remains unmerged. The core issue is whether Metal shader code should live inside bitsandbytes or as an external dependency.

---

## 12. Apple Intelligence Foundation Models Quantization

**Status:** Production (deployed across Apple devices)
**Paper:** https://arxiv.org/abs/2507.13575
**Blog:** https://machinelearning.apple.com/research/apple-foundation-models-tech-report-2025

### Apple's Internal Quantization Approach

Apple's ~3B on-device model uses a combination that is far more aggressive than any open-source method:

| Component | Precision | Method |
|-----------|-----------|--------|
| Model weights | **2-bit** | QAT with learnable weight clipping + special initialization |
| Embedding table | **4-bit** | Standard quantization |
| KV cache | **8-bit** | Runtime quantization |

### Key Innovation: 2-bit QAT

Unlike post-training quantization, Apple trains the model knowing it will be inferred at 2-bit:
- Learnable weight clipping boundaries
- Special weight initialization for low-bit training stability
- Result: ~8x smaller than full precision with minimal quality loss

### Vision Components

The on-device model includes:
- **ViTDet-L backbone** with Register-Window mechanism
- Vision-language adaptation module that compresses visual features to fixed token count
- Efficient extraction of both local and global visual features

### Relevance

Apple's 2-bit QAT approach represents the gold standard for on-device quantization but is not available as a public library. The techniques described in the tech report could guide custom quantization implementations for diffusion models targeting Apple Silicon.

---

## 13. Summary Matrix

### Practical Options for Diffusion Models on Apple Silicon (Ranked)

| Rank | Method | Memory Savings | Speed Benefit | Maturity | Effort |
|------|--------|---------------|---------------|----------|--------|
| 1 | **MLX (mflux)** | 4x (4-bit) | Native Metal | Production | Low |
| 2 | **GGUF + Diffusers** | 2-8x | Memory only | Production | Low |
| 3 | **Draw Things** | 3-6x | Best perf | Production | N/A (app) |
| 4 | **DiffusionKit + MLX** | 2-4x | Native Metal | Beta | Medium |
| 5 | **TorchAO ARM kernels** | 2-4x | CPU only | Experimental | Medium |
| 6 | **Core ML + ANE** | 2-8x | ANE accel | Mature | High |
| 7 | **Quanto on MPS** | 2-4x | None | Maintenance | Low |

### Methods NOT Applicable to Apple Silicon Diffusion

| Method | Reason |
|--------|--------|
| TurboQuant | KV-cache only, no weight quant, H100-focused |
| SpinQuant | CUDA-only (fast-hadamard-transform) |
| QuIP# | CUDA-only (E8 lattice kernels) |
| SqueezeLLM | CUDA-only, LLM-only |
| AQLM | CUDA-only, LLM-only |
| bitsandbytes MPS | PR not merged, security concerns |

---

## 14. Recommendations for realrestore-cli

### Immediate (can implement now)

1. **GGUF loading via Diffusers** for the restoration pipeline. Pre-quantized FLUX GGUF files (Q4_K, Q8_0) are available on HuggingFace. This is the lowest-effort path to reduce memory from ~24GB to ~6-12GB for the transformer component.

2. **mflux integration** if switching to MLX backend is feasible. Native 4-bit and 8-bit quantization with pre-quantized models available. Best performance on Apple Silicon.

### Medium-term (requires development)

3. **Core ML conversion with mixed-bit quantization** using Apple's ml-stable-diffusion tools. Achieves 2.81-bit average with layer-aware precision allocation. Requires pre-analysis run but yields the best memory/quality tradeoff.

4. **Q-DiT techniques** for custom quantization of the restoration model's transformer layers. The evolutionary search for per-layer granularity and dynamic activation quantization are directly applicable.

### Watch

5. **bitsandbytes MPS** -- if PR #1853 gets merged with in-tree Metal kernels, this becomes the easiest path for NF4/FP4 quantization with real GPU acceleration.

6. **TorchAO MPS kernels** -- PyTorch team may add native Metal quantized matmul kernels. This would make INT4/INT8 actually fast on MPS instead of emulated.

7. **Metal FlashAttention** open-sourcing -- Draw Things' implementation is closed source but the engineering blog provides enough detail to implement custom Metal attention kernels.

---

## X/Twitter Discussion Highlights (2025-2026)

- **@awnihannun** (MLX lead): DeepSeek R1 671B running on 2x M2 Ultra with 3-bit quantization (~4 bpw), faster than reading speed. Demonstrates MLX + quantization at extreme scale on consumer hardware.
  - https://x.com/awnihannun/status/1881412271236346233

- **@ggerganov** (ggml/llama.cpp creator): "The future of on-device inference is ggml + Apple Silicon"
  - https://x.com/ggerganov/status/1665403955801739267

- Active discussion of MoE-Quant, QuEST (1-bit QAT), BATQuant (MXFP4), and various quantization methods in the ML research community on X.

---

## Key References

### Papers
- TurboQuant: https://arxiv.org/abs/2504.19874 (ICLR 2026)
- PolarQuant: https://arxiv.org/abs/2502.02617 (AISTATS 2026)
- SpinQuant: https://arxiv.org/abs/2405.16406 (ICLR 2025)
- QuIP#: https://arxiv.org/abs/2402.04396 (ICLR 2024)
- Q-DiT: https://arxiv.org/abs/2406.17343 (CVPR 2025)
- BinaryAttention: https://arxiv.org/abs/2603.09582 (2026)
- Apple Foundation Models: https://arxiv.org/abs/2507.13575 (2025)
- MLX Benchmarks on Apple Silicon: https://arxiv.org/abs/2510.18921 (2025)
- LLM Inference on Apple Silicon: https://arxiv.org/abs/2508.08531 (2025)
- LLM/MLLM at Scale on Apple Silicon: https://arxiv.org/abs/2601.19139 (2026)

### Repositories
- TorchAO: https://github.com/pytorch/ao
- MLX: https://github.com/ml-explore/mlx
- mflux: https://github.com/filipstrand/mflux
- DiffusionKit: https://github.com/argmaxinc/DiffusionKit
- ComfyUI-GGUF: https://github.com/city96/ComfyUI-GGUF
- SpinQuant: https://github.com/facebookresearch/SpinQuant
- QuIP#: https://github.com/Cornell-RelaxML/quip-sharp
- SqueezeLLM: https://github.com/SqueezeAILab/SqueezeLLM
- Q-DiT: https://github.com/Juanerx/Q-DiT
- ml-ane-transformers: https://github.com/apple/ml-ane-transformers
- ml-vision-transformers-ane: https://github.com/apple/ml-vision-transformers-ane
- more-ane-transformers: https://github.com/smpanaro/more-ane-transformers
- Apple ml-stable-diffusion: https://github.com/apple/ml-stable-diffusion
- Mochi Diffusion: https://github.com/MochiDiffusion/MochiDiffusion
