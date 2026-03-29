# GitHub Research: Running Large Diffusion Models on Apple Silicon MPS

**Date**: 2026-03-29
**Scope**: Comprehensive GitHub search for MPS dtype handling, offloading, and large model optimization

---

## 1. Diffusers Models on MPS with float16 -- MPSNDArray Dtype Assertion

### The Core Problem

When running diffusers pipelines with `torch_dtype=torch.float16` on MPS, multiple dtype assertion failures occur at the Metal framework level. The root errors come from Apple's Metal Performance Shaders framework itself, not from PyTorch or diffusers.

**Key error messages:**
- `MPSNDArrayConvolutionA14.mm:3976: failed assertion 'destination datatype must be fp32'` -- Convolution ops require fp32 output
- `error: input types 'tensor<1x77x1xf16>' and 'tensor<1xf32>' are not broadcast compatible` -- LayerNorm/broadcast mismatch
- `MPSNDArrayMatrixMultiplication.mm:644: failed assertion` -- MatMul dtype issues

### PyTorch Tracking Issues

**[pytorch/pytorch#119108](https://github.com/pytorch/pytorch/issues/119108)** -- [OPEN] "[MPS] Tracking issue for ModuleInfo failures when enabling testing for torch.float16"
- Comprehensive list of float16 failures on MPS
- Conv1d, Conv2d, ConvTranspose1d, ConvTranspose2d ALL fail with "destination datatype must be fp32"
- AvgPool2d, BCELoss fail with broadcast incompatibility between f16 and f32
- GroupNorm, LayerNorm, InstanceNorm fail with NaN outputs or crashes
- **Status**: OPEN since Feb 2024, no fix

**[pytorch/pytorch#78168](https://github.com/pytorch/pytorch/issues/78168)** -- [CLOSED] "MPS 16Bit Not Working correctly"
- Original 2022 report showing `.half().item()` returns wrong values on MPS
- `bfloat16` also returned garbage values
- Fixed in later PyTorch nightlies for basic ops, but convolutions remain broken
- URL: https://github.com/pytorch/pytorch/issues/78168

**[pytorch/pytorch#96113](https://github.com/pytorch/pytorch/issues/96113)** -- "[mps] [PyTorch 2.0] LayerNorm crashes when input is in float16"
- Reported by HuggingFace maintainer @pcuenca after diffusers user reports
- LayerNorm with float16 input causes MPS graph compilation failure

**[pytorch/pytorch#160332](https://github.com/pytorch/pytorch/issues/160332)** -- [CLOSED] "Use FP32 for ConvTranspose3D when using autocast on MPS"
- Fix: Register ConvTranspose3D for FP32 in MPS autocast policy
- **Solution pattern**: Add operations to `KERNEL_MPS(..., fp32_cast_policy)` in `aten/src/ATen/autocast_mode.cpp`
- This is the template for how MPS autocast should handle unsupported fp16 ops

```cpp
// The fix in PyTorch source (autocast_mode.cpp):
// Register ConvTranspose3D as requiring fp32 on MPS
KERNEL_MPS(conv_transpose3d, fp32_cast_policy)
```

### Diffusers-Specific Issues

**[huggingface/diffusers#2521](https://github.com/huggingface/diffusers/issues/2521)** -- [CLOSED] "MPS fails with float16 and PyTorch 2.0"
- Basic StableDiffusionPipeline with `torch_dtype=torch.float16` crashes on MPS
- Error: `input types 'tensor<1x77x1xf16>' and 'tensor<1xf32>' are not broadcast compatible`
- Root cause traced to PyTorch MPS LayerNorm not handling mixed dtypes
- **No diffusers-level fix possible** -- this is a PyTorch MPS bug

**[huggingface/diffusers#7426](https://github.com/huggingface/diffusers/issues/7426)** -- [CLOSED] "[MPS] SDXL pipeline fails inference in fp16 mode"
- DDIM scheduler changes latent dtype from float16 to float32 between steps
- Causes broadcast mismatch on step 2+
- **Workaround found by @bghira**:
```python
# Force dtype preservation after scheduler step
old_dtype = latents.dtype
latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
if latents.dtype is not old_dtype:
    latents = latents.to(old_dtype)
```
- Also: **disabling autocast on MPS** prevents the dtype drift

**[huggingface/diffusers#9047](https://github.com/huggingface/diffusers/issues/9047)** -- [CLOSED] "flux does not work on MPS devices"
- Flux uses `torch.float64` for rotary embeddings which MPS doesn't support
- `enable_model_cpu_offload()` also doesn't work
- Model too large for CPU fallback
- 41 comments, significant community frustration
- **Root cause**: `torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device)` -- MPS can't do float64

**[huggingface/diffusers#7563](https://github.com/huggingface/diffusers/issues/7563)** -- [OPEN] "[mps] training / inference dtype issues"
- Two-phase problem: First NDArray >2^32 crash, then Input type float vs BFloat16 mismatch
- **Attention slicing resolves the NDArray size crash**
- Training with bf16 requires a special AdamW optimizer that handles bf16 weights
- Key finding: `accelerator.native_amp = False` needed for MPS

```python
# MPS training setup pattern
if torch.backends.mps.is_available():
    accelerator.native_amp = False
results = accelerator.prepare(unet, lr_scheduler, optimizer, *train_dataloaders)
unet = results[0]
if torch.backends.mps.is_available():
    unet.set_attention_slice()
```

**[huggingface/diffusers#13227](https://github.com/huggingface/diffusers/issues/13227)** -- [OPEN] "[Bug] GlmImagePipeline silently corrupts weights on MPS accelerator"
- **CRITICAL finding**: `device_map="mps"` silently corrupts model weights
- float32 + MPS direct load: weights corrupted, bias OK
- float16 + MPS direct load: bias corrupted, weights OK
- Results in extreme values (~1e37), LayerNorm overflow, NaN outputs
- **Workaround**: ALWAYS load on CPU first, then `.to("mps")`

```python
# WRONG -- corrupts weights
pipe = Pipeline.from_pretrained(model, torch_dtype=torch.float32, device_map="mps")

# CORRECT -- load CPU first, then move
pipe = Pipeline.from_pretrained(model, torch_dtype=torch.float32)
pipe.to("mps")
```

### How Projects Handle the Dtype Assertion

**Consensus solution across all projects**: Use float32 on MPS, not float16.

From the [Medium deep-dive by @michaelhannecke](https://medium.com/@michael.hannecke/unleashing-apple-silicons-hidden-ai-superpower-a-technical-deep-dive-into-mps-accelerated-image-9573ba90570a):
- Float32: 18-20 sec/image
- Float16: 22-25 sec/image (actually SLOWER on Apple Silicon)
- Mixed precision: crashes or produces artifacts
- "Apple's GPU architecture doesn't have the same float16 optimization paths as NVIDIA's Tensor Cores"

**Official HuggingFace MPS documentation** (https://huggingface.co/docs/diffusers/en/optimization/mps):
- Recommends float16 in example code BUT warns about NDArray >2^32
- Recommends attention slicing for all systems <64GB RAM
- Does NOT mention the convolution fp32 assertion issue

---

## 2. MPSNDArray "different datatype" Fixes in PyTorch

### The 4GB (2^32 bytes) NDArray Limit

**[pytorch/pytorch#143859](https://github.com/pytorch/pytorch/issues/143859)** -- [OPEN] "MPSNDArray limits single NDArray memory to 4GB"
- `MPSNDArray.mm:850: failed assertion '[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: total bytes of NDArray > 2**32'`
- Affects M4 Pro 64GB, macOS 15.2
- **This is an Apple MPS framework limitation, not a PyTorch bug**
- No workaround for single arrays >4GB

**[pytorch/pytorch#149261](https://github.com/pytorch/pytorch/issues/149261)** -- [OPEN] "MPS Error: NDArray > 2^32 bytes in scaled_dot_product_attention"
- Reproducible with: `q = torch.randn(1, 12, 29640, 128).to("mps")`
- Attention computation creates intermediate buffers exceeding 4GB
- PyTorch 2.8.0.dev, M4 Max
- **No fix available** -- hard Apple framework limit

**[pytorch/pytorch#146769](https://github.com/pytorch/pytorch/issues/146769)** -- [OPEN] "MPS Error on sequoia 15.3: NDArray dimension length > INT_MAX"
- Related 32-bit dimension limit

**[pytorch/pytorch#84520](https://github.com/pytorch/pytorch/issues/84520)** -- [OPEN] "MPS backend appears to be limited to 32 bits"
- Original 2022 report of the fundamental 32-bit limitation in MPS
- Still open in 2026

### MatMul Bugs on M4

**[pytorch/pytorch#178056](https://github.com/pytorch/pytorch/issues/178056)** -- [CLOSED] "MPS: mm/addmm SEGFAULTS on M4 if 2nd matrix is padded with LORADOWN GEMV"
- Crash in `MPSNDArrayMatrixMultiplication.mm:644`
- Specific to M4 chip with padded matrices
- Affects PyTorch 2.10/2.11/nightly
- **Fix merged** into PyTorch

### vllm-metal Workaround for 4GB Limit

**[vllm-project/vllm-metal#43](https://github.com/vllm-project/vllm-metal/issues/43)** -- [CLOSED]
- Detailed analysis of the 4GB `MPSTemporaryNDArray` limit
- Even vocab_size=151,936 in float32 (~608MB base) can trigger it due to internal MPS temp buffers
- **Fix merged as [PR #51](https://github.com/vllm-project/vllm-metal/pull/51)**: Conservative 1GB threshold for tensor transfers to MPS
- Tensors >1GB are chunked or kept on CPU to avoid hitting the 4GB limit
- vllm-metal repo: 782 stars, very actively maintained (updated 2026-03-29)
- URL: https://github.com/vllm-project/vllm-metal

---

## 3. Qwen2-VL / Qwen2.5-VL on MPS

### Qwen2-VL MPS Issues

**[huggingface/transformers#33399](https://github.com/huggingface/transformers/issues/33399)** -- [CLOSED] "Qwen2-VL-7B-Instruct on MPS"
- Mac M3 Max, `torch_dtype=torch.bfloat16` to MPS device
- 48 comments -- significant community effort
- Multiple dtype-related crashes when using MPS
- **Root cause**: Qwen2-VL uses operations not supported in MPS (bfloat16, specific attention patterns)
- Key comment from @zucchini-nlp: "We had several problems with size mismatch in LLaVA so the error might be related to that"

### Qwen2.5-VL / Qwen3-VL on M4

**[QwenLM/Qwen3-VL#992](https://github.com/QwenLM/Qwen3-VL/issues/992)** -- [CLOSED] "cannot run on M4 Pro"
- Qwen2.5-VL with `torch_dtype=torch.bfloat16, device_map="auto"` hangs/crashes on M4 Pro
- Same issue confirmed for Qwen2.5-3B-Instruct on M4 Pro
- **No working MPS solution provided** -- issue was closed without resolution

### BFloat16 on MPS -- Widespread Problem

Multiple repos report "BFloat16 is not supported on MPS":
- [stable-diffusion-webui-forge#2399](https://github.com/lllyasviel/stable-diffusion-webui-forge/issues/2399)
- [ComfyUI#6254](https://github.com/Comfy-Org/ComfyUI/issues/6254)
- [oobabooga/text-generation-webui#5216](https://github.com/oobabooga/text-generation-webui/issues/5216)
- [microsoft/lida#85](https://github.com/microsoft/lida/issues/85)

**Note**: PyTorch added bfloat16 support for MPS in later versions, but many operations still don't support it. The error typically comes from older PyTorch versions or unsupported op paths.

---

## 4. RealRestorer + MPS / Apple Silicon

### Official RealRestorer Repo

- **Repo**: [yfyang007/RealRestorer](https://github.com/yfyang007/RealRestorer) -- 103 stars
- **Last pushed**: 2026-03-29 (actively maintained)
- **No Apple Silicon / MPS port exists** in the official repo or forks
- No issues filed about MPS support
- Official quick start requires CUDA with bfloat16

### Known Forks and Ports

- **[StartHua/Comfyui_RealRestorer](https://github.com/StartHua/Comfyui_RealRestorer)** -- 7 stars, ComfyUI integration
- **[199-biotechnologies/realrestore-cli](https://github.com/199-biotechnologies/realrestore-cli)** -- Our project, only known MPS port attempt
- **No other MPS-specific ports found anywhere on GitHub**

### Architecture Challenges for MPS

RealRestorer uses:
1. **Step1X-Edit** backbone -- DiT (Diffusion Transformer) architecture
2. **QwenVL** for semantic encoding -- the Qwen2-VL models that have MPS issues (see section 3)
3. **Dual-stream diffusion network** -- large attention computation
4. **Official dtype: bfloat16** -- problematic on MPS (see bfloat16 section above)

**Implication**: All three major subsystems have known MPS compatibility issues.

---

## 5. apple/ml-stable-diffusion Updates

### Repo Status

- **URL**: https://github.com/apple/ml-stable-diffusion
- **Stars**: 17,831
- **Last pushed**: 2025-07-03 (no updates in 9 months)
- **Latest release**: v1.1.1 (2024-05-04) -- MIT license change + minor fixes
- **Open issues**: 186

### What It Supports

- Stable Diffusion v1.x, v2.x, SDXL via Core ML conversion
- Neural Engine (ANE) acceleration via `SPLIT_EINSUM_V2` attention
- 6-bit weight compression (palettization)
- Swift and Python inference

### How It Handles Large Models

The project does **NOT** handle 30GB+ models. Key limitations:
- Designed for SD-scale models (~2-4GB parameters)
- Core ML conversion requires the entire model to fit in a `.mlmodelc` package
- No streaming or chunked loading support
- No equivalent of `enable_model_cpu_offload` for Core ML
- SDXL is the largest model officially supported

### Apple's Approach to Dtype

- Core ML handles all dtype conversion internally during compilation
- Models are converted to optimized Metal-native formats
- float16 is the default for ANE; float32 for GPU
- 6-bit and mixed-bit quantization for further compression
- **Key insight**: Apple bypasses PyTorch MPS entirely by using Core ML

---

## 6. Popular Repos Optimizing Diffusers for Apple Silicon (2025-2026)

### vllm-project/vllm-metal -- 782 stars
- URL: https://github.com/vllm-project/vllm-metal
- Community-maintained vLLM plugin for Apple Silicon
- Actively addresses MPS limitations (4GB NDArray limit, tensor bridging)
- Updated: 2026-03-29

### MochiDiffusion/MochiDiffusion
- URL: https://github.com/MochiDiffusion/MochiDiffusion
- Native macOS app for Stable Diffusion using Core ML
- Bypasses MPS entirely by using Apple's Core ML framework
- One of the most successful Apple Silicon diffusion projects

### michaelhannecke/apple-silicon
- URL: https://github.com/michaelhannecke/apple-silicon
- Practical guides for running LLMs and image generation on macOS with MPS
- Detailed benchmarks showing float32 > float16 performance on Apple Silicon

### ivanfioravanti/z-image-mps
- URL: https://github.com/ivanfioravanti/z-image-mps
- Z-Image-Turbo optimized for Apple Silicon
- Treats MPS as first-class citizen with CUDA/CPU fallbacks

### Key Observation

**No repo with >100 stars has successfully run 30GB+ diffusion models on MPS with PyTorch.** The successful projects either:
1. Use Core ML conversion (apple/ml-stable-diffusion, MochiDiffusion)
2. Use MLX framework instead (various ml-explore projects)
3. Stay within smaller model sizes (<10GB)

---

## 7. enable_model_cpu_offload vs enable_sequential_cpu_offload on MPS

### enable_model_cpu_offload -- DOES NOT WORK ON MPS

**[huggingface/diffusers#4197](https://github.com/huggingface/diffusers/issues/4197)** -- [CLOSED] "Feature Request: enable_model_cpu_offload for MPS"
- Filed 2023-07, closed as stale without implementation
- The function hardcodes CUDA device references: `torch.device(f"cuda:{gpu_id}")`
- Proposed fix was simple but never merged:

```python
# Proposed but never merged
if gpu_id == "mps":
    device = torch.device(gpu_id)
else:
    device = torch.device(f"cuda:{gpu_id}")

if self.device.type != "cpu":
    self.to("cpu", silence_dtype_warnings=True)
    if self.device.type == "mps":
        torch.mps.empty_cache()
    else:
        torch.cuda.empty_cache()
```

- **Status**: Still not implemented in diffusers as of 2026-03

### enable_sequential_cpu_offload -- PARTIALLY WORKS but VERY SLOW

- Uses `accelerate` hooks to move individual layers between CPU and accelerator
- Can technically work with MPS since it uses generic `.to(device)` calls
- But: 3x+ slowdown compared to keeping everything on device
- URL: https://github.com/huggingface/diffusers/issues/2266

### FitDiT-ComfyUI Confirms the Problem

**[BoyuanJiang/FitDiT-ComfyUI#5](https://github.com/BoyuanJiang/FitDiT-ComfyUI/issues/5)** -- [OPEN]
- `with_offload=True` or `with_aggressive_offload=True` triggers `AssertionError: Torch not compiled with CUDA enabled`
- Root cause: `enable_model_cpu_offload()` / `enable_sequential_cpu_offload()` contain CUDA-specific code paths
- **Only workaround**: Set `with_offload=False` on MPS

### Practical Impact for RealRestorer

Since RealRestorer's Step1X-Edit + QwenVL pipeline is ~30GB:
- Cannot use `enable_model_cpu_offload()` on MPS (CUDA-only)
- `enable_sequential_cpu_offload()` technically possible but extremely slow
- Must implement custom offloading strategy

---

## 8. PyTorch MPS Issues: Meta Tensors and Accelerate Hooks

### Meta Tensor + MPS Device Transfer

**[huggingface/accelerate#3617](https://github.com/huggingface/accelerate/issues/3617)** -- [CLOSED]
- `NotImplementedError: Cannot copy out of meta tensor; no data! Please use torch.nn.Module.to_empty() instead of torch.nn.Module.to()`
- Occurs with `device_map="auto"` on models with tied weights
- Accelerate 1.7.0 + PyTorch 2.6.0
- **Key finding**: safetensors format doesn't provide tied parameters, leaving them on meta device
- **Workaround**: Use `use_safetensors=False` or manually handle tied weights

### MPS-Specific Meta Tensor Issues

When using accelerate's offloading hooks with MPS:
1. `device_map="auto"` doesn't properly detect MPS as a valid accelerator
2. Meta tensors created during model init can't be directly moved to MPS
3. The `AlignDevicesHook` from accelerate uses CUDA-specific logic

### PYTORCH_ENABLE_MPS_FALLBACK Limitations

- Setting `PYTORCH_ENABLE_MPS_FALLBACK=1` allows unsupported ops to fall back to CPU
- BUT: This creates performance cliff edges where a single op forces CPU round-trip
- Many fallback operations also fail due to the meta tensor issue
- [pytorch/pytorch#86195](https://github.com/pytorch/pytorch/issues/86195) -- MPS_FALLBACK has no effect for some ops

### MPS Memory Leak Issues

**[pytorch/pytorch#154329](https://github.com/pytorch/pytorch/issues/154329)** -- [OPEN] "MPS Memory Leak"
- Qwen2 models specifically mentioned
- Memory grows unbounded during inference on MPS

**[meta-pytorch/torchtune#2473](https://github.com/meta-pytorch/torchtune/issues/2473)** -- [OPEN] "MPS memory leak"
- Confirmed across multiple model types

---

## Summary of Findings

### What Works on MPS for Diffusion Models

1. **float32 inference** for small-medium models (<10GB)
2. **Attention slicing** to reduce peak memory
3. **CPU-first loading** then `.to("mps")` (never `device_map="mps"`)
4. **CPU-based random generators** for reproducibility
5. **Core ML conversion** for Apple-optimized inference (bypasses MPS entirely)
6. **MLX framework** as an alternative to PyTorch MPS

### What Does NOT Work on MPS

1. **float16 inference** -- convolution ops crash with "destination datatype must be fp32"
2. **bfloat16 inference** -- many ops unsupported
3. **enable_model_cpu_offload()** -- hardcoded CUDA references
4. **Single tensors >4GB** -- hard Apple MPS framework limit
5. **Attention with sequences >~30K tokens** -- NDArray >2^32 bytes
6. **device_map="mps"** -- silently corrupts weights (diffusers#13227)
7. **Mixed precision / autocast** -- scheduler dtype drift causes crashes

### Recommended Strategy for RealRestorer on Apple Silicon

Based on all findings, the viable approaches are:

1. **float32 with manual component offloading** -- Load components individually to MPS in float32, move back to CPU when done. Implement custom hooks instead of relying on diffusers' broken MPS offloading.

2. **Chunked attention** -- Use attention slicing and/or tiled VAE to keep individual tensor allocations under 4GB.

3. **Core ML conversion** (long-term) -- Convert Step1X-Edit DiT to Core ML for Apple-native inference. No existing tooling for this model architecture.

4. **MLX port** (alternative) -- Port critical components to MLX framework which handles Apple Silicon dtype and memory natively.

### Key Code Patterns from the Community

```python
# Pattern 1: Safe MPS pipeline loading (from diffusers#13227 workaround)
pipe = Pipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipe.to("mps")  # Never use device_map="mps"

# Pattern 2: Attention slicing for memory (from diffusers MPS docs)
pipe.enable_attention_slicing(slice_size=1)
pipe.enable_vae_slicing()

# Pattern 3: MPS-safe random generation
generator = torch.Generator(device='cpu').manual_seed(42)

# Pattern 4: Memory cleanup
torch.mps.empty_cache()
torch.mps.synchronize()

# Pattern 5: Disable native AMP for MPS (from diffusers#7563)
if torch.backends.mps.is_available():
    accelerator.native_amp = False

# Pattern 6: Force dtype preservation after scheduler (from diffusers#7426)
old_dtype = latents.dtype
latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
if latents.dtype != old_dtype:
    latents = latents.to(old_dtype)

# Pattern 7: vllm-metal 4GB limit workaround (from vllm-metal PR#51)
MAX_MPS_TENSOR_BYTES = 1 * 1024 * 1024 * 1024  # 1GB conservative limit
if tensor.nelement() * tensor.element_size() > MAX_MPS_TENSOR_BYTES:
    # Process in chunks or keep on CPU
    pass

# Pattern 8: Environment variable for unsupported op fallback
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
```
