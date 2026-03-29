# RealRestorer on Apple M4 Max 64GB: Codex Optimization Deep Dive

Date: 2026-03-29

## Executive Summary

The current failure mode is not fundamentally "the model is too large for Apple Silicon." The real problems are:

1. The upstream RealRestorer pipeline is not MPS-clean. It hardcodes `torch.bfloat16` for latents/noise on the MPS path and ties prompt-embedding output dtype to the text encoder dtype.
2. The Qwen2.5-VL text encoder is the only component that genuinely needs `float32` for stability on MPS. The transformer already casts `encoder_hidden_states` to its own dtype internally, so cross-component mixed precision is possible.
3. `enable_sequential_cpu_offload()` is the wrong tool here. In this pipeline it collides with explicit `self.text_encoder.to(...)` calls inside `_get_qwenvl_embeds()`, which explains the meta-tensor/offload-hook failures.
4. Google TurboQuant is not a solution to the main RealRestorer problem. TurboQuant is a KV-cache compression method for autoregressive LLM serving. RealRestorer does not maintain a growing KV cache across decoding the way an LLM does.
5. If "no quality compromise" is literal, the best path is still full-precision weights where they matter, with correct staging and dtype normalization. If "no quality compromise" means "no measurable visual degradation," the best practical accelerator is quantizing only the Qwen text encoder with an Apple-native path.

My recommendation:

- Exact-quality baseline:
  - `text_encoder`: `float32`
  - `transformer`: `float16`
  - `vae`: `float32`
  - `prompt_embeds`, `noise`, `ref_latents`, `img_ids`, `txt_ids`: transformer dtype on MPS (`float16`)
  - No `bfloat16` anywhere on MPS
  - Stage Qwen deliberately; do not assume it can remain permanently co-resident with the denoiser without squeezing activation headroom
  - No sequential/model CPU offload
  - No attention slicing on 64GB unless profiling proves otherwise
- Fastest near-lossless path:
  - Keep transformer and VAE unquantized
  - Move only Qwen2.5-VL to an Apple-native 8-bit path (`transformers` Metal quantization or MLX)

## What Is Actually Broken in This Repo

The repo and upstream pipeline already expose the key root causes:

- `pipeline_realrestorer.py:393` uses `dtype=self.text_encoder.dtype` when creating prompt embeddings.
- `pipeline_realrestorer.py:586` and `pipeline_realrestorer.py:600` hardcode `torch.bfloat16` for reference latents and initial noise.
- `transformer_realrestorer.py:139` already casts `encoder_hidden_states` to the transformer's dtype internally:

```python
llm_embedding=encoder_hidden_states.to(device=model_device, dtype=model_dtype)
```

- `_get_qwenvl_embeds()` explicitly moves the text encoder with `.to(...)` and later moves it back to CPU when `_offload_device` is set. That is exactly the kind of behavior that conflicts with Accelerate's offload hooks and leads to meta/offload errors.

Bottom line: the correct fix is boundary dtype normalization and manual staging, not generic offload.

## Direct Answers

### 1. What is Google TurboQuant? How does it work? Is there a repo? Can it work on MPS/Apple Silicon?

TurboQuant is a Google Research compression method for KV-cache-heavy autoregressive inference, not a weight quantizer for large diffusion checkpoints.

How it works:

- It compresses KV cache entries to extremely low bit widths.
- Its core ideas are PolarQuant plus error correction via Quantized Johnson-Lindenstrauss projections.
- The point is to reduce KV-cache memory and attention bandwidth while preserving downstream accuracy.

Repo status:

- I did not find an official Google code release linked from the Google Research blog post.
- I did find active community implementations:
  - `TheTom/turboquant_plus` for `llama.cpp`/Metal
  - `helgklaizar/turboquant_mlx` for MLX

Apple Silicon status:

- Yes, community ports already run on Apple Silicon through MLX or Metal-backed llama.cpp.
- No, that does not make TurboQuant directly useful for RealRestorer.

Why it does not solve the main RealRestorer problem:

- RealRestorer is a diffusion transformer. Its attention tensors are recomputed at each denoising step.
- It does not maintain a long-lived, growing KV cache across autoregressive token generation.
- TurboQuant does not shrink the 23GB transformer weights.
- TurboQuant does not fix MPS mixed-dtype matmul assertions.

Verdict: interesting, real, and already on Apple Silicon, but not the lever to pull for this project.

### 2. What are the latest torchao quantization techniques that work on MPS?

The short answer is that torchao has moved forward, but MPS is still not the mature path for large-model diffusion inference.

What is current:

- torchao has modern low-bit configs such as int4 weight-only and int8 dynamic-activation/intx-weight quantization.
- Recent torchao releases mention low-bit CPU and MPS kernels becoming pip-installable from source.
- Arm CPU support is more mature than Apple GPU support.

What matters for this project:

- Stable torchao workflows are still documented and optimized primarily around CUDA and CPU.
- MPS remains experimental enough that I would not make torchao-on-MPS the foundation of a "must work" RealRestorer runtime.
- Float8-style paths are not the answer on MPS here.
- For Apple hardware, Hugging Face's newer Metal quantization path and MLX-native quantization are currently more Apple-specific than torchao.

Practical ranking for RealRestorer:

1. `transformers` Metal quantization for Qwen2.5-VL if you accept near-lossless, not exact, behavior.
2. MLX 8-bit or 4-bit Qwen2.5-VL if you want the most Apple-native text-encoder path.
3. Quanto only as a memory-saving experiment, not as a guaranteed speedup on MPS.
4. torchao MPS as a prototype path, not the default production answer.

What I would not do:

- I would not quantize the RealRestorer transformer first if the requirement is no quality compromise.
- I would not rely on MPS int4 as the primary production path for a restoration model.

### 3. How do ComfyUI and other tools handle 30GB+ models on macOS?

The successful macOS tools do not generally solve this by "just load the raw bf16 checkpoint on MPS and pray."

They use one of four patterns:

#### A. GGUF or other pre-quantized transformer checkpoints

ComfyUI's large-model workflows on macOS commonly use `ComfyUI-GGUF` for transformer-heavy models such as FLUX/Qwen-image-class systems.

- This is primarily a memory-fit strategy.
- Q8 is the closest thing to near-lossless.
- Q4 and below are for fit, not strict fidelity.

This pattern is useful if you decide that "no quality compromise" can mean "no visible regression after validation," not bitwise equivalence.

#### B. MLX-native inference

Projects like MFLUX and MLX-based ports keep the entire runtime inside Apple's stack:

- unified memory aware
- custom Metal kernels
- Apple-friendly quantization
- compiled execution without the PyTorch MPS abstraction layer

This is currently the cleanest Apple-native path for very large generative models, especially when the model family already has MLX ports.

#### C. Custom Metal attention kernels

Draw Things is the clearest example. It does not depend on stock PyTorch MPS attention performance. It uses custom Metal FlashAttention work and heavily tuned Apple-specific kernels.

This is how macOS-native high-performance diffusion apps get ahead of generic PyTorch pipelines.

#### D. Core ML export with chunking/palettization

Apple's `ml-stable-diffusion` path uses:

- component splitting
- attention rewrites like `SPLIT_EINSUM`
- chunking
- optional weight compression

This is viable when the model architecture can be exported cleanly. It is not a quick fix for RealRestorer's current custom DiT + Qwen2.5-VL pipeline.

Verdict: the macOS winners either quantize the large transformer, move to MLX, or ship custom Metal/Core ML kernels. They do not rely on Accelerate offload hooks to make a 30GB+ diffusion stack feel native.

### 4. What is the correct fix for the MPSNDArrayMatrixMultiplication dtype assertion in diffusers?

The correct fix is not "load everything in float32" and not "turn on sequential offload."

The correct fix is:

1. Remove all `bfloat16` from the MPS path.
2. Run Qwen2.5-VL in `float32`.
3. Cast Qwen outputs to the transformer's dtype exactly once at the component boundary.
4. Keep the transformer in `float16` on MPS.
5. Keep the VAE in `float32`.
6. Stop using `enable_sequential_cpu_offload()` on this pipeline.

Why this is the right fix:

- MPS matmul kernels require dtype consistency inside each operation.
- Qwen2.5-VL is the unstable component; the transformer is already written to downcast `encoder_hidden_states` to its own dtype.
- The upstream pipeline's hardcoded `bfloat16` latents/noise inject extra dtype inconsistency on MPS for no benefit.
- Offload hooks create a second problem: meta tensors and hook-managed device transitions colliding with manual `.to(...)` calls.

Concrete patch targets:

#### Patch 1: prompt embedding output dtype

Current:

```python
txt, mask = self._get_qwenvl_embeds(
    ...,
    dtype=self.text_encoder.dtype,
)
```

Recommended:

```python
target_dtype = self.transformer.dtype
txt, mask = self._get_qwenvl_embeds(
    ...,
    dtype=target_dtype,
)
```

This keeps Qwen internal compute in `float32` while emitting `float16` prompt embeddings for the denoiser.

#### Patch 2: remove hardcoded `torch.bfloat16`

Current:

```python
ref_latents = self._pack_latents(ref_latents_tensor.to(device=device, dtype=torch.bfloat16))
...
noise = randn_tensor(..., dtype=torch.bfloat16)
```

Recommended:

```python
latent_dtype = self.transformer.dtype
ref_latents = self._pack_latents(ref_latents_tensor.to(device=device, dtype=latent_dtype))
...
noise = randn_tensor(..., dtype=latent_dtype)
```

#### Patch 3: no sequential CPU offload

Do not call:

```python
pipe.enable_sequential_cpu_offload()
```

Instead:

- place the text encoder deliberately
- place the transformer deliberately
- let the VAE stay full precision

#### Patch 4: only benchmark `PYTORCH_MPS_PREFER_METAL=1` as a secondary tuning knob

It may help select better matmul kernels on Apple GPUs.
It is not the primary correctness fix.

### 5. Can we use mixed precision where `text_encoder=float32` and `transformer=float16` on MPS simultaneously?

Yes.

This is already supported by the RealRestorer transformer design:

- the transformer takes `encoder_hidden_states`
- then immediately casts them to `hidden_states.dtype`

So the mixed-precision rule is:

- inside the text encoder: `float32`
- at the boundary to the transformer: cast once to `float16`
- inside the transformer: stay `float16`

That is the correct architecture for MPS here.

The caveat is memory policy:

- Keeping a 30GB `float32` Qwen resident alongside a 23GB `float16` transformer is risky on a 64GB machine once activations and OS pressure are included.
- Therefore, "mixed precision is possible" is not the same as "keep both hot and resident forever."

The production interpretation should be:

- exact-quality mode: stage Qwen carefully and do not rely on generic offload hooks
- fast mode: move Qwen to an Apple-native 8-bit runtime and keep the denoiser full precision

In other words: mixed precision is valid, but a permanently hot 30GB `float32` Qwen plus a hot 23GB `float16` transformer is not the memory posture I would design around on a 64GB machine.

### 6. What about DeepCache, Token Merging (ToMe), or step distillation?

#### DeepCache

DeepCache was designed around the observation that later diffusion steps reuse deep features and that caching them can skip repeated compute.

For this project:

- It is not the first thing I would deploy.
- The classic DeepCache integration story is strongest for UNet-based Stable Diffusion pipelines.
- RealRestorer is a custom DiT-like transformer denoiser, so a direct drop-in path is much weaker.

Recommendation:

- Not a short-term production answer.
- If you want a caching-based acceleration for transformer denoisers, look at newer DiT-oriented cache work rather than assuming vanilla DeepCache will translate cleanly.

#### ToMe / token merging

ToMe is more structurally relevant because RealRestorer is transformer-based.

But:

- ToMe deliberately merges tokens.
- RealRestorer is a restoration model, not a text-to-image model where small generative drift may be acceptable.
- Restoration quality is patch-sensitive; token merging can erase the exact local detail you are trying to reconstruct.

Recommendation:

- Do not enable ToMe by default if "no quality compromise" is a hard requirement.
- Only test it behind image-quality metrics such as PSNR, SSIM, LPIPS, and restoration-specific visual regression sets.

#### Step distillation

This is the highest-upside speed path, but it is not a runtime optimization.

- It requires training a student, consistency model, or distilled scheduler/model.
- It can deliver the biggest throughput jump because it reduces the denoising loop itself.
- It is a medium-term research project, not a loading fix.

Recommendation:

- This is the right medium-term bet if you want an order-of-magnitude speedup.
- It is not the right immediate answer for "make the current checkpoint run on M4 Max without quality loss."

### 7. Are there any Apple-specific Metal shader tricks for transformer inference?

Yes, but only some are practical in the current stack.

#### Practical today

- `PYTORCH_MPS_PREFER_METAL=1`
- `PYTORCH_MPS_FAST_MATH=1` only if you accept numerical drift; do not enable this for strict fidelity work
- `torch.mps.compile_shader()` for targeted custom kernels
- `torch.mps.profiler` signposts for Instruments-based profiling

#### High-value custom kernel targets in this model

The best places to fuse are not generic elementwise ops. They are:

1. `qkv split -> q/k norm -> RoPE -> prepack for attention`
2. latent pack/unpack around `einops.rearrange`
3. CFG merge plus `process_diff_norm`
4. `img_ids` generation

These are attractive because the current RealRestorer layers do a lot of reshape/transpose/upcast/downcast work around attention.

#### What already exists externally

- Philip Turner's Metal FlashAttention work
- Draw Things style custom Metal attention kernels
- Apple's Core ML `SPLIT_EINSUM` / ANE transformer rewrites

#### What is not realistic as a quick fix

- Dropping a custom Metal FlashAttention implementation directly into stock diffusers/PyTorch without native extension work
- Assuming ANE will accelerate this custom transformer stack without a serious Core ML export effort

## Recommended Strategy for This Project

## Phase 1: Correctness and full-quality performance

This is the path I would ship first.

1. Patch the upstream pipeline so the MPS path is dtype-clean:
   - no `bfloat16`
   - Qwen internal compute `float32`
   - denoiser inputs `float16`
2. Remove sequential/model CPU offload from the Apple Silicon path.
3. Keep attention slicing off on the 64GB M4 Max unless profiling proves a net gain.
4. Keep VAE tiling only for large images.
5. Add shape bucketing so MPS shader caches are reused.
6. Profile before changing kernels.

Expected result:

- this should solve the dtype assertion correctly
- this should remove the meta-tensor offload failure mode
- this should give the fastest exact-quality PyTorch/diffusers baseline you can reasonably expect on M4 Max

## Phase 2: Fast path if tiny, validated drift is acceptable

This is the best practical speed/memory win.

1. Keep the RealRestorer transformer and VAE full precision.
2. Move only Qwen2.5-VL to:
   - Hugging Face Metal 8-bit quantization, or
   - MLX 8-bit quantization
3. Validate prompt-embedding cosine similarity and output-image regressions on a real restoration benchmark set.

Why this is the right place to quantize first:

- Qwen runs once per image
- it is the largest source of dtype instability
- it is the largest single weight block after upcasting to `float32`
- moving it to an Apple-native 8-bit runtime makes the whole system much easier to fit and stage

## Phase 3: Long-horizon Apple-native acceleration

If the project outgrows PyTorch MPS:

1. MLX port of the transformer hot path
2. custom Metal attention kernels
3. potential Core ML export of the denoiser only, not the whole pipeline
4. model-side step distillation

## What I Would Not Recommend

- Do not use TurboQuant as the headline optimization for this project.
- Do not keep `enable_sequential_cpu_offload()` in the Apple path.
- Do not standardize on `bfloat16` for MPS here.
- Do not quantize the restoration transformer to int4 first.
- Do not enable ToMe by default on a restoration model.
- Do not assume ANE is a drop-in accelerator for this custom pipeline.

## Final Conclusion

The best answer on a 64GB M4 Max is not "find a magic quantizer." It is:

- make the pipeline MPS-correct
- run Qwen in `float32`
- run the denoiser in `float16`
- keep the VAE in `float32`
- stage Qwen instead of relying on offload hooks or assuming permanent co-residency
- eliminate `bfloat16` and offload-hook misuse
- then, only if you accept tiny validated drift, quantize Qwen with an Apple-native 8-bit path

That gives you the best current balance of:

- correctness
- memory fit
- speed
- preserved restoration quality

## References

- Google Research blog: TurboQuant
  - https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/
- Community Apple/Metal TurboQuant implementations
  - https://github.com/TheTom/turboquant_plus
  - https://github.com/helgklaizar/turboquant_mlx
- torchao docs and releases
  - https://docs.pytorch.org/ao/stable/
  - https://github.com/pytorch/ao/releases
- PyTorch MPS documentation and environment variables
  - https://docs.pytorch.org/docs/stable/mps.html
  - https://docs.pytorch.org/docs/stable/mps_environment_variables.html
- Hugging Face Metal quantization
  - https://huggingface.co/docs/transformers/main/en/quantization/metal
- Hugging Face Diffusers quantization and GGUF
  - https://huggingface.co/docs/diffusers/main/en/quantization/overview
  - https://huggingface.co/docs/diffusers/main/en/quantization/gguf
- ComfyUI and ComfyUI-GGUF
  - https://github.com/Comfy-Org/ComfyUI
  - https://github.com/city96/ComfyUI-GGUF
- Apple's Core ML Stable Diffusion tooling
  - https://github.com/apple/ml-stable-diffusion
- Apple transformer and Stable Diffusion optimization notes
  - https://machinelearning.apple.com/research/neural-engine-transformers
  - https://machinelearning.apple.com/research/stable-diffusion-coreml-apple-silicon
- Metal FlashAttention / Draw Things related work
  - https://github.com/philipturner/metal-flash-attention
- DeepCache and ToMe
  - https://github.com/horseee/DeepCache
  - https://github.com/dbolya/tomesd
