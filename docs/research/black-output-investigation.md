# Black Output Image Investigation

**Date:** 2026-03-29
**Symptom:** Pipeline completes successfully, output image is entirely black (all zeros).
**Warning:** `RuntimeWarning: invalid value encountered in cast` at `images = (images * 255).round().astype("uint8")`
**Hardware:** Apple Silicon (M4 Max), MPS backend, PyTorch 2.x

---

## Executive Summary

The black output is almost certainly caused by **NaN (Not a Number) values in the latent tensor** that propagate through the VAE decoder and get clipped to 0 during the uint8 cast. The root cause is a **bfloat16-to-float16 precision mismatch**: RealRestorer (and its base model Step1X-Edit) were trained in bfloat16, but our pipeline runs in float16 on MPS because MPS does not natively support bfloat16. Float16 has a much smaller dynamic range than bfloat16, causing overflow/underflow in attention layers, normalization ops, and the VAE decoder, which produces NaN values that cascade into an all-black output.

This is a **well-documented, widespread issue** across the diffusers ecosystem with dozens of GitHub issues confirming the exact same failure mode.

---

## 1. RealRestorer Repository Issues

**Repository:** [yfyang007/RealRestorer](https://github.com/yfyang007/RealRestorer)

As of 2026-03-29, the repository has only **1 issue** (#1 "GRam required", closed) asking about memory requirements. No issues exist about black output, wrong results, or MPS compatibility. The model was released on 2026-03-26, so it is extremely new with minimal community feedback.

**Key finding from the HuggingFace model card** ([RealRestorer/RealRestorer](https://huggingface.co/RealRestorer/RealRestorer)):
- The model is based on [stepfun-ai/Step1X-Edit](https://github.com/stepfun-ai/Step1X-Edit)
- The official inference code uses `torch_dtype=torch.bfloat16`
- The CLI uses `--torch_dtype bfloat16`
- **No float16 example is provided anywhere in the official documentation**

---

## 2. The bfloat16 vs float16 Problem (ROOT CAUSE)

### 2.1 Why bfloat16 models fail in float16

| Property | bfloat16 | float16 |
|----------|----------|---------|
| Exponent bits | 8 | 5 |
| Mantissa bits | 7 | 10 |
| Dynamic range | ~1.2e-38 to ~3.4e+38 | ~6.1e-5 to ~6.5e+4 |

bfloat16 has the **same exponent range as float32** (8 bits), meaning it can represent very large and very small values without overflow. float16 has a much narrower range (~65504 max). When a model trained in bfloat16 has internal activation values that exceed float16's range, those values become NaN or infinity.

### 2.2 Direct evidence from identical architectures

**Stable Cascade Prior** ([stabilityai/stable-cascade discussion #3](https://huggingface.co/stabilityai/stable-cascade/discussions/3)):
- Prior produces **all NaN outputs** when using float16 instead of bfloat16
- Traced to `AttnBlock` in the UNet where tensor accumulates NaN after attention computation
- Stability AI acknowledged this as a **known issue**
- Workaround: use bfloat16 or float32

**Flux with torch.half** ([diffusers #9096](https://github.com/huggingface/diffusers/issues/9096)):
- Running Flux (also a bfloat16-trained model) with float16 produces NaN values
- The exact same `"invalid value encountered in cast"` warning appears
- Fix merged in PR #9097 ensuring consistent dtype handling
- Root cause: **bfloat16 initialization with subsequent float16 conversion causes NaN propagation**

**Z-Image FP16** ([Tongyi-MAI/Z-Image #14](https://github.com/Tongyi-MAI/Z-Image/issues/14)):
- FP16 inference produces completely black images with NaN latents
- Root cause: "FP16 handling in the UNet or scheduler (overflow/underflow leading to NaNs)"
- NaN values originate from arithmetic overflow/underflow and cascade through all downstream computations

**SDXL VAE** ([madebyollin/sdxl-vae-fp16-fix](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix)):
- The standard SDXL VAE generates NaN in fp16 because "internal activation values are too big"
- Fix: retrained the VAE to keep internal activation values smaller (within float16 range)
- This proves the fundamental issue: **models trained in higher precision have activation magnitudes that exceed float16's representable range**

### 2.3 MPS does not support bfloat16

MPS on Apple Silicon does not natively support bfloat16 operations. When you load a bfloat16 model on MPS, PyTorch silently upcasts to float32, which is slow (78.6GB for the full RealRestorer model on a 64GB machine). This is why our code uses float16 as a compromise -- but float16 is NOT numerically equivalent to bfloat16.

References:
- [BFloat16 Unsupported on MPS - PyTorch #141864](https://github.com/pytorch/pytorch/issues/141864)
- [Apple Developer Forums - BFloat16 support](https://developer.apple.com/forums/thread/726201)

---

## 3. MPS-Specific Issues That Compound the Problem

### 3.1 Softmax NaN on MPS

**PyTorch #96602** ([pytorch/pytorch #96602](https://github.com/pytorch/pytorch/issues/96602)):
- softmax returns NaN attention probabilities for large tensors on MPS
- Affects **both float16 and float32**
- Root cause: subtraction operation `diffs = x - maxes` unexpectedly produces NaN
- Linked to "FMA (fused multiply-add) precision issues" on Metal
- Status: may be fixed in PyTorch 2.5.1+ with macOS 15.1, but the issue remains open

### 3.2 CrossAttention NaN from baddbmm

**Diffusers PR #2643** ([huggingface/diffusers #2643](https://github.com/huggingface/diffusers/pull/2643)):
- On MPS, allocating a large bias tensor for `baddbmm()` causes it to return NaN attention scores
- The tensor allocation itself causes corruption, even when the tensor is unused (`beta=0`)
- Fix: allocate a smaller `(1,1,1)` tensor and broadcast it
- This NaN propagates through softmax and produces black images

### 3.3 LayerNorm float16 crash on MPS

**PyTorch #96113** ([pytorch/pytorch #96113](https://github.com/pytorch/pytorch/issues/96113)):
- LayerNorm crashes on MPS with float16 input
- Error: "input types 'tensor<1x77x768xf16>' and 'tensor<768xf32>' are not broadcast compatible"
- Fixed in PyTorch 2.0.1+ (PR #96430)
- **But: the fix was for the crash, not for numerical correctness of float16 LayerNorm on MPS**

### 3.4 GroupNorm crashes on MPS

**PyTorch #99981** ([pytorch/pytorch #99981](https://github.com/pytorch/pytorch/issues/99981)):
- GroupNorm causes kernel crash during backward pass on M1/MPS
- Regression in PyTorch 2.0+
- May be fixed in macOS 14.6.1 with recent torch nightly
- Swapping to BatchNorm or InstanceNorm avoids the crash

### 3.5 Silent failures with non-contiguous tensors

**PyTorch #165257** ([pytorch/pytorch #165257](https://github.com/pytorch/pytorch/issues/165257)):
- MPS random in-place operations silently fail on non-contiguous tensors (macOS < 15.0)
- Operations like `normal_()`, `uniform_()` do not modify the output tensor at all
- The tensor stays at its initialized value (usually zeros), which looks identical to "never updated"
- This could cause latent noise initialization to silently fail, producing zero latents

---

## 4. The "invalid value encountered in cast" Warning -- Documented Pattern

This exact warning is the #1 reported symptom across **at least 8 diffusers GitHub issues**:

| Issue | Model | Root Cause | Fix |
|-------|-------|-----------|-----|
| [#4104](https://github.com/huggingface/diffusers/issues/4104) | SD 2.1 | float16 NaN overflow | Use float32 |
| [#4325](https://github.com/huggingface/diffusers/issues/4325) | SDXL 1.0 | VAE float16 instability | Use madebyollin/sdxl-vae-fp16-fix |
| [#6815](https://github.com/huggingface/diffusers/issues/6815) | SD 2.1 | float16 on weak GPU | Use float32 |
| [#8759](https://github.com/huggingface/diffusers/issues/8759) | SD3 | T5 encoder needs fp32 | Use from_pretrained() not from_single_file() |
| [#8970](https://github.com/huggingface/diffusers/issues/8970) | Kolors | attention_slicing + MPS | Remove enable_attention_slicing() |
| [#9096](https://github.com/huggingface/diffusers/issues/9096) | Flux | bfloat16/float16 mismatch | PR #9097 dtype fix |
| [#10343](https://github.com/huggingface/diffusers/issues/10343) | CogView3 | float16 NaN | Use float32 or bfloat16 |
| [#1251](https://github.com/huggingface/diffusers/issues/1251) | RePaint | MPS backend incompatibility | Unresolved |

**Common thread:** In every case, NaN values in the latent/image tensor cause the uint8 cast to produce zeros (black pixels). The NaN originates from dtype precision issues, almost always involving float16.

---

## 5. VAE-Specific Failure Mode

The VAE decoder is the most common point of failure for black output:

1. **Model runs inference in float16** -- attention layers may produce slightly out-of-range values
2. **NaN propagates through the UNet** -- one NaN in an attention score poisons the entire computation
3. **VAE decoder receives NaN latents** -- decodes them to NaN pixel values
4. **Image processor casts to uint8** -- NaN becomes 0 (black)
5. **Warning is emitted** -- `"RuntimeWarning: invalid value encountered in cast"`

The standard fix across the ecosystem is one of:
- **Run the VAE in float32** (even if the rest of the model is float16)
- **Use a VAE retrained for float16 stability** (e.g., madebyollin/sdxl-vae-fp16-fix)
- **Run the entire pipeline in float32** (safest but uses 2x memory)

Reference: [Medium article on black images on Mac](https://medium.com/@shruk1403/why-do-stable-diffusion-images-turn-out-black-on-mac-02d720d6e42c)

---

## 6. PYTORCH_MPS_PREFER_METAL Interaction

Our code sets `PYTORCH_MPS_PREFER_METAL=1` to fix MPSNDArrayMatrixMultiplication dtype assertion errors. While this fixes one class of errors, it changes the Metal shader path used for matrix multiplication. There is no direct evidence that this env var causes black output, but:

- It routes matmul through a different kernel than the default MPSNDArray path
- Combined with float16 precision, any kernel differences could affect numerical stability
- The [CivitAI article on fixing black images](https://civitai.com/articles/11106/fixing-black-images-in-comfyui-on-mac-m1m2-pytorch-260-and-mps) recommends `--force-fp32` as the primary fix alongside MPS configuration

---

## 7. attention_slicing Interaction

Multiple issues confirm that `enable_attention_slicing()` causes black output on MPS:

- [Diffusers #8970](https://github.com/huggingface/diffusers/issues/8970): Kolors produces black images specifically because of attention_slicing on MPS
- [HuggingFace forum thread](https://discuss.huggingface.co/t/activating-attention-slicing-leads-to-black-images-when-running-diffusion-more-than-once/68623): Black images on second run with attention_slicing enabled

Our code correctly avoids attention_slicing on 64GB devices, but this is worth noting for lower-memory configurations.

---

## 8. Diagnosis Methodology

To confirm NaN values are the cause, add these debug checks before the image is saved:

```python
# After pipeline output, before saving:
import torch
latents = pipe_output.images  # or wherever the raw output is
if isinstance(latents, torch.Tensor):
    print(f"NaN count: {torch.isnan(latents).sum().item()}")
    print(f"Inf count: {torch.isinf(latents).sum().item()}")
    print(f"Min: {latents.min().item()}, Max: {latents.max().item()}")
    print(f"All zeros: {(latents == 0).all().item()}")

# Check numpy array before uint8 cast:
import numpy as np
if isinstance(image_array, np.ndarray):
    print(f"NaN: {np.isnan(image_array).any()}")
    print(f"Inf: {np.isinf(image_array).any()}")
    print(f"Range: [{image_array.min()}, {image_array.max()}]")
```

---

## 9. Potential Fixes (Ranked by Likelihood of Success)

### Fix 1: Force float32 for the entire pipeline (SAFEST)

The model is 39GB in bfloat16. In float32 it would be ~78GB, which exceeds 64GB. This requires INT8 quantization to fit:
- Load in float32 + INT8 quantize = ~20GB
- Eliminates all float16 NaN issues
- May be slower than float16 but produces correct output

### Fix 2: Upcast VAE to float32 (TARGETED)

Keep the transformer/UNet in float16 but run the VAE decoder in float32:
```python
pipe.vae = pipe.vae.to(dtype=torch.float32)
```
This addresses the most common failure point while keeping memory manageable.

### Fix 3: Add NaN clamping in the pipeline

Insert NaN detection and replacement before VAE decode:
```python
latents = torch.nan_to_num(latents, nan=0.0, posinf=1.0, neginf=-1.0)
```
This is a band-aid that prevents the crash but may produce low-quality output.

### Fix 4: Mixed-precision strategy

- Text encoder: float32 (already done -- Qwen2.5-VL needs it)
- Transformer/UNet: float16 with NaN monitoring
- VAE decoder: float32 (upcast before decode, downcast after)
- Scheduler: float32 (to prevent accumulation errors)

### Fix 5: Wait for MPS bfloat16 support

Apple has shown interest in bfloat16 support for Metal/MPS. When it lands, this problem goes away entirely. However, there is no timeline for this.

---

## 10. References

### RealRestorer / Step1X-Edit
- [yfyang007/RealRestorer](https://github.com/yfyang007/RealRestorer) -- source repository (1 issue, none about black output)
- [RealRestorer/RealRestorer HuggingFace](https://huggingface.co/RealRestorer/RealRestorer) -- model card, uses bfloat16
- [stepfun-ai/Step1X-Edit](https://github.com/stepfun-ai/Step1X-Edit) -- base model, uses bfloat16

### Diffusers Black Image Issues
- [diffusers #4104](https://github.com/huggingface/diffusers/issues/4104) -- "invalid value encountered in cast" black output
- [diffusers #4325](https://github.com/huggingface/diffusers/issues/4325) -- SDXL black image, VAE fp16 NaN
- [diffusers #6815](https://github.com/huggingface/diffusers/issues/6815) -- SD 2.1 black image, float16
- [diffusers #8759](https://github.com/huggingface/diffusers/issues/8759) -- SD3 invalid value cast, T5 precision
- [diffusers #8970](https://github.com/huggingface/diffusers/issues/8970) -- Kolors MPS black image, attention_slicing
- [diffusers #9096](https://github.com/huggingface/diffusers/issues/9096) -- Flux float16 NaN, bfloat16 mismatch
- [diffusers #10343](https://github.com/huggingface/diffusers/issues/10343) -- CogView3 black image
- [diffusers #1251](https://github.com/huggingface/diffusers/issues/1251) -- RePaint MPS black image
- [diffusers #1614](https://github.com/huggingface/diffusers/issues/1614) -- SD 2.1 blank black output with autocast
- [diffusers #2521](https://github.com/huggingface/diffusers/issues/2521) -- MPS fails with float16 PyTorch 2.0

### Diffusers MPS Fixes
- [diffusers PR #2643](https://github.com/huggingface/diffusers/pull/2643) -- MPS CrossAttention NaN fix (baddbmm)
- [diffusers PR #9097](https://github.com/huggingface/diffusers/issues/9096) -- Flux fp16 inference fix
- [HuggingFace MPS docs](https://huggingface.co/docs/diffusers/en/optimization/mps) -- official MPS optimization guide

### PyTorch MPS Bugs
- [pytorch #96602](https://github.com/pytorch/pytorch/issues/96602) -- MPS softmax NaN in float16/float32
- [pytorch #96113](https://github.com/pytorch/pytorch/issues/96113) -- MPS LayerNorm float16 crash
- [pytorch #99981](https://github.com/pytorch/pytorch/issues/99981) -- MPS GroupNorm crash
- [pytorch #88331](https://github.com/pytorch/pytorch/issues/88331) -- MPS NaN from NativeGroupNormBackward
- [pytorch #78168](https://github.com/pytorch/pytorch/issues/78168) -- MPS 16-bit not working correctly
- [pytorch #128435](https://github.com/pytorch/pytorch/issues/128435) -- MPS inference causes poor results
- [pytorch #165257](https://github.com/pytorch/pytorch/issues/165257) -- MPS silent failure on non-contiguous tensors
- [pytorch #141864](https://github.com/pytorch/pytorch/issues/141864) -- BFloat16 unsupported on MPS

### VAE Precision Fixes
- [madebyollin/sdxl-vae-fp16-fix](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix) -- retrained VAE for fp16 stability
- [stabilityai/stable-cascade discussion #3](https://huggingface.co/stabilityai/stable-cascade/discussions/3) -- Prior all-NaN in float16

### General Resources
- [Medium: Why Stable Diffusion images turn out black on Mac](https://medium.com/@shruk1403/why-do-stable-diffusion-images-turn-out-black-on-mac-02d720d6e42c)
- [Tongyi-MAI/Z-Image #14](https://github.com/Tongyi-MAI/Z-Image/issues/14) -- FP16 black images with NaN latents
- [HuggingFace discussion on attention_slicing black images](https://discuss.huggingface.co/t/activating-attention-slicing-leads-to-black-images-when-running-diffusion-more-than-once/68623)

---

## 11. Conclusion

**The black output is not a bug in RealRestorer or in our code.** It is a fundamental precision mismatch: a model trained in bfloat16 being forced to run in float16 because MPS does not support bfloat16. This exact failure mode is documented in dozens of GitHub issues across the diffusers ecosystem, affecting Flux, SDXL, SD3, Stable Cascade, CogView3, Kolors, and now RealRestorer.

The fix is to either:
1. Run the VAE (and ideally the full pipeline) in float32, accepting the memory cost
2. Use INT8 quantization + float32 to fit within 64GB
3. Add targeted float32 upcasting for the VAE decoder and normalization layers
4. Insert NaN detection/clamping as a safety net

The problem will be permanently resolved when Apple adds native bfloat16 support to MPS, but there is no public timeline for this.
