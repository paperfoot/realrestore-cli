# RealRestore CLI Innovations Review

Date: 2026-03-29

## Bottom Line

The biggest missing wins are not generic "more MPS tuning". They are:

1. Reduce memory churn inside the denoising loop.
2. Bucket image shapes so Apple kernels and any Core ML export path can stay static.
3. Add a real Tree-Ring attack path instead of generic image-spectrum cleanup.
4. Treat `torch.export + CoreML` as a denoiser-only project, not a full-pipeline conversion.

The current code already has some useful modules, but several of the best ideas are either missing, only partially researched, or not wired into the execution path.

## 1. Missing Optimizations That Could Matter Most

### 1.1 Shape Bucketing Is Missing

The research discusses MPS, MLX, and Core ML broadly, but it does not push hard enough on shape policy. The upstream pipeline resizes to arbitrary multiples of 16, preserving aspect ratio, which means a large number of unique shapes can hit the runtime ([`upstream-realrestorer/diffusers/src/diffusers/pipelines/realrestorer/pipeline_realrestorer.py:357`](../../upstream-realrestorer/diffusers/src/diffusers/pipelines/realrestorer/pipeline_realrestorer.py#L357)).

Why this matters:

- MPSGraph and Metal shader caches work much better with a small set of repeated shapes.
- `torch.export` and Core ML both become far more practical with fixed buckets.
- Warmup cost and benchmark variance drop materially.

Recommendation:

- Introduce a small bucket set such as `512`, `768`, `1024`, plus 2-3 common aspect ratios.
- Snap resize targets to those buckets rather than arbitrary `16`-aligned sizes.

### 1.2 Per-Step Allocation Churn Is a Bigger Problem Than the Research Implies

In the denoising loop, the pipeline repeatedly allocates with `repeat`, `cat`, and fresh timestep/guidance tensors on every step ([`upstream-realrestorer/diffusers/src/diffusers/pipelines/realrestorer/pipeline_realrestorer.py:443`](../../upstream-realrestorer/diffusers/src/diffusers/pipelines/realrestorer/pipeline_realrestorer.py#L443), [`upstream-realrestorer/diffusers/src/diffusers/pipelines/realrestorer/pipeline_realrestorer.py:491`](../../upstream-realrestorer/diffusers/src/diffusers/pipelines/realrestorer/pipeline_realrestorer.py#L491)).

Why this matters on Apple Silicon:

- Unified memory removes PCIe cost, but it does not make allocator churn free.
- This is exactly the kind of bandwidth and residency pressure that slows MPS.

Recommendation:

- Preallocate `latent_model_input`, `model_input`, `t_vec`, and `guidance_vec`.
- Fill views in-place instead of rebuilding tensors every iteration.
- Fuse CFG post-processing (`cond/uncond`, `diff_norm`, scaling) into one tensor pass or one custom kernel.

### 1.3 The Research Stops at Scheduler Tuning, but the Real Win Is Step Distillation

The scheduling module is sensible, but it only changes step count heuristically. It is not actually integrated into guidance decay, and the engine only uses it to overwrite `steps` ([`python/realrestore_cli/engine.py:211`](../../python/realrestore_cli/engine.py#L211)). The bigger missing idea is a true low-step student model or consistency-style distillation for the RealRestorer backbone.

Recommendation:

- Treat `4-8` step distillation as a higher-value medium-term bet than additional int4/int8 work on MPS.
- For a restoration model, fewer denoising passes usually beats marginal kernel-level wins.

### 1.4 Precision Consistency Needs to Be Fixed Before More Tuning

The CLI selects `float16` on MPS ([`python/realrestore_cli/engine.py:58`](../../python/realrestore_cli/engine.py#L58)), but the upstream pipeline still hardcodes `bfloat16` for reference latents and noise ([`upstream-realrestorer/diffusers/src/diffusers/pipelines/realrestorer/pipeline_realrestorer.py:586`](../../upstream-realrestorer/diffusers/src/diffusers/pipelines/realrestorer/pipeline_realrestorer.py#L586), [`upstream-realrestorer/diffusers/src/diffusers/pipelines/realrestorer/pipeline_realrestorer.py:600`](../../upstream-realrestorer/diffusers/src/diffusers/pipelines/realrestorer/pipeline_realrestorer.py#L600)).

That mismatch makes profiling noisy and complicates export work.

Recommendation:

- Normalize the MPS path to one latent dtype.
- Benchmark that clean baseline before chasing more platform tricks.

### 1.5 Some Already-Built Optimizations Are Not Wired In

There is useful work in the repo that is not actually used in the main path:

- `tile`, `tile_size`, and `tile_overlap` exist in `restore_image`, but tiling is not invoked ([`python/realrestore_cli/engine.py:182`](../../python/realrestore_cli/engine.py#L182)).
- CLI parsing does not expose `quality` or tiling flags ([`python/realrestore_cli/engine.py:288`](../../python/realrestore_cli/engine.py#L288)).
- `guidance_decay` exists in scheduling, but the main engine does not apply it to the diffusion loop ([`python/realrestore_cli/optimizations/scheduling.py:44`](../../python/realrestore_cli/optimizations/scheduling.py#L44)).

These are not "innovations", but they are high-ROI gaps.

## 2. Apple Silicon Metal Shader Tricks We Are Not Using

### 2.1 PyTorch MPS Knobs Missing in Code

The research mentions `PYTORCH_MPS_FAST_MATH=1` and `PYTORCH_MPS_PREFER_METAL=1`, but the runtime only sets watermark ratios and fallback ([`python/realrestore_cli/optimizations/mps_backend.py:63`](../../python/realrestore_cli/optimizations/mps_backend.py#L63)).

From PyTorch docs:

- `PYTORCH_MPS_FAST_MATH=1` enables fast math for MPS kernels.
- `PYTORCH_MPS_PREFER_METAL=1` prefers Metal kernels over MPSGraph for matmul-heavy workloads.

Recommendation:

- Add both as opt-in flags and benchmark them on the denoiser specifically.

### 2.2 `torch.mps.compile_shader` Is an Underused Escape Hatch

PyTorch now exposes `torch.mps.compile_shader`, which can compile and invoke custom Metal compute shaders directly from Python. That is the cleanest way to prototype targeted kernels without leaving the PyTorch runtime.

Best candidates in this codebase:

- Latent pack/unpack replacing repeated `einops.rearrange`.
- `img_ids` grid generation.
- CFG merge and `process_diff_norm`.
- Watermark FFT-domain suppression for batch cleanup.

This is the most interesting platform-specific trick currently absent from both the research and implementation.

### 2.3 Profile with MPS Signposts Before Writing Kernels

PyTorch also exposes `torch.mps.profiler.start`, which emits OS signposts viewable in Instruments. That should be used before any custom Metal work.

Recommendation:

- Profile one `1024` bucket end-to-end.
- Confirm whether the top cost is SDPA, QKV/RoPE prep, CFG merge, pack/unpack, or VAE decode.

### 2.4 Native Attention Rewrite Is Still the Highest-Ceiling Metal Bet

The denoiser stack does many separate memory-moving ops around attention: `qkv` projection, `rearrange`, Q/K norm, RoPE application, concatenation, attention, then projection ([`upstream-realrestorer/diffusers/src/diffusers/models/realrestorer/layers.py:279`](../../upstream-realrestorer/diffusers/src/diffusers/models/realrestorer/layers.py#L279), [`upstream-realrestorer/diffusers/src/diffusers/models/realrestorer/layers.py:386`](../../upstream-realrestorer/diffusers/src/diffusers/models/realrestorer/layers.py#L386)).

If you ever go native, the best fused kernel target is:

- `qkv split -> qk norm -> rope -> attention prepack`

That is more valuable than writing a custom kernel for generic elementwise ops.

## 3. Accelerate for Image Preprocessing

Short answer: yes, but mostly for preprocess, tiling, and watermark paths, not the main denoiser.

### Where Accelerate Helps

- `vImage` for resize, colorspace conversion, planar/interleaved conversion, normalization, and tile blending.
- `vDSP` for FFT/DCT and reductions in watermark detection and removal.
- `BNNSGraph` / `BNNSGraphBuilder` for small low-latency pre/post graphs, especially the auto-detect path and image cleanup utilities.

Why this is relevant here:

- Current preprocessing is mostly PIL/NumPy.
- Watermark removal and detection are entirely NumPy/SciPy based ([`python/realrestore_cli/watermark/remover.py:18`](../../python/realrestore_cli/watermark/remover.py#L18), [`python/realrestore_cli/watermark/detector.py:20`](../../python/realrestore_cli/watermark/detector.py#L20)).
- Auto-detect also uses SciPy filters and hand-built kernels ([`python/realrestore_cli/optimizations/auto_detect.py:30`](../../python/realrestore_cli/optimizations/auto_detect.py#L30)).

### Best Accelerate Uses for This Project

1. Replace PIL + NumPy resize / channel conversion with a `vImage` path.
2. Replace watermark FFT and DCT routines with `vDSP`.
3. Use `vImage` for tile extraction and Gaussian overlap blending.
4. Keep denoiser compute on MPS/MLX/Core ML; do not try to run the main model in Accelerate.

### Why This Is Also a Zero-Copy Opportunity

Apple's `vImage.PixelBuffer` APIs can reference existing raw data or `CVPixelBuffer` storage, which is the right substrate for a low-copy input pipeline.

## 4. Zero-Copy Unified Memory Patterns Worth Trying

### 4.1 The Current MLX Bridge Is Not Zero-Copy

`python/realrestore_cli/optimizations/mlx_backend.py` describes a "NumPy zero-copy bridge", but the current implementation does:

- `tensor.detach().cpu().float().numpy()`
- `mx.array(np_array)`
- `np.array(mlx_array)`
- `torch.from_numpy(np_array).to(device)`

([`python/realrestore_cli/optimizations/mlx_backend.py:104`](../../python/realrestore_cli/optimizations/mlx_backend.py#L104))

Unified memory helps physical residency, but this is still a framework boundary with material conversion overhead.

### 4.2 Better Patterns

1. Keep an entire stage in one framework.
   - Example: MLX for Qwen embedding generation only, then one handoff.
2. Use `CVPixelBuffer` / `IOSurface` for native preprocess/postprocess shared across vImage, Metal, and Core ML.
3. Use memory-mapped safetensors plus `low_cpu_mem_usage=True` during model load.
4. Preallocate persistent buffers for latents, CFG inputs, and tile accumulators.
5. Use `torch.mps.recommended_max_memory()` instead of a hardcoded working-set assumption for runtime budgeting.

I did not verify a production-ready DLPack bridge between MLX and PyTorch from the sources I checked, so I would treat that as an experiment, not a plan assumption.

## 5. Watermark Removal: What Actually Works for Tree-Ring

### Current State

The current remover is mostly generic image-domain cleanup:

- spectral suppression
- DWT
- DCT
- a lightweight "adversarial purification"
- optional restore pass

([`python/realrestore_cli/watermark/remover.py:18`](../../python/realrestore_cli/watermark/remover.py#L18))

That is not the right primary attack surface for Tree-Ring. Tree-Ring is detected after diffusion inversion into latent noise space, not by looking at the final image FFT. The current detector is also only a proxy image-spectrum detector, not a real DDIM-inversion detector ([`python/realrestore_cli/watermark/detector.py:228`](../../python/realrestore_cli/watermark/detector.py#L228)).

### Best Attack, by Practical Setting

If you want one black-box default:

- Use small spatial translation / crop-pad first.
- This is cheap and specifically disrupts the Tree-Ring phase structure.

If you want the strongest targeted attack with good quality:

- Use the public-VAE surrogate attack from *A Crack in the Bark*.
- That is the best model-aware route if you have the relevant VAE or a compatible public surrogate.

If you want the best product strategy:

1. Targeted Tree-Ring pre-attack: `7px` shift or subpixel affine shift.
2. Boundary repair / copy-back.
3. Low-noise regeneration through the restoration model.
4. Metadata strip.

### Recommendation

Add a dedicated `tree_ring_targeted` method and evaluate it against a real inversion-based detector. Until that exists, the current Tree-Ring scores are not trustworthy enough for product claims.

## 6. `torch.export + CoreML` for the Denoiser Backbone

Short answer: yes, but only for the denoiser, and probably GPU-first on Mac.

### Why It Is Promising

Core ML Tools now recommends `torch.export.export` for PyTorch export, but also states that the path is new and still beta. The export path was added in Core ML Tools 8, and Apple says op translation coverage is roughly `~70%`.

That still makes the RealRestorer denoiser a reasonable target because its core ops are mostly:

- Linear
- LayerNorm / RMS-like norms
- GELU(tanh)
- concatenation
- SDPA
- reshape / permute

The denoiser itself has far fewer non-convertible concerns than the full pipeline.

### Why the Whole Pipeline Is the Wrong Export Target

The full pipeline includes:

- PIL and string preprocessing
- Qwen processor templating
- Python-side loops over prompt chunks
- scheduler orchestration
- arbitrary resize logic

That is all poor export territory.

### Best Scope

Export one denoiser step as a standalone module with fixed-shape buckets:

- `hidden_states`
- `encoder_hidden_states`
- `prompt_mask`
- `img_ids`
- `txt_ids`
- `timestep`
- `guidance`

Keep the scheduler loop, VAE, and Qwen processing outside initially.

### ANE Caution

A naive export is unlikely to become a good ANE model. The backbone still uses `nn.Linear` in `(B, L, C)` layout, whereas Apple's ANE-optimized transformer guidance is built around:

- 4D channels-first tensors
- `Linear -> 1x1 Conv`
- split-einsum attention
- chunked query processing

So the realistic path is:

1. `torch.export + CoreML` for Mac GPU deployment first.
2. Use `TorchExportMLModelComparator` and `MLModelBenchmarker`.
3. Only pursue ANE after a deliberate rewrite toward Apple's transformer layout.

### Compression Strategy for Core ML

If export works, Core ML's mixed-bit and grouped-channel palettization are more interesting than PyTorch int4 on MPS for this model. That is where Apple actually has a deployment-quality compression stack.

## Recommended Experiment Order

1. Fix the dtype mismatch and add shape buckets.
2. Preallocate denoising-loop buffers and remove `repeat`/`cat` churn.
3. Benchmark `PYTORCH_MPS_FAST_MATH=1` and `PYTORCH_MPS_PREFER_METAL=1`.
4. Add a real Tree-Ring attack path and real detector evaluation.
5. Replace watermark FFT/DCT preprocessing with Accelerate (`vDSP` / `vImage`).
6. Prototype `torch.export + CoreML` on a single denoiser bucket at `1024`.

## Sources

- Local research: [`docs/research/mps-optimizations.md`](../research/mps-optimizations.md)
- Local research: [`docs/research/memory-optimization.md`](../research/memory-optimization.md)
- Local research: [`docs/research/ane-offloading.md`](../research/ane-offloading.md)
- Local research: [`docs/research/mlx-conversion.md`](../research/mlx-conversion.md)
- Local research: [`docs/research/watermark-removal.md`](../research/watermark-removal.md)
- PyTorch MPS environment variables: <https://docs.pytorch.org/docs/stable/mps_environment_variables.html>
- PyTorch `torch.mps.compile_shader`: <https://docs.pytorch.org/docs/2.8/generated/torch.mps.compile_shader.html>
- PyTorch `torch.mps.recommended_max_memory`: <https://docs.pytorch.org/docs/stable/generated/torch.mps.recommended_max_memory.html>
- PyTorch MPS profiler: <https://docs.pytorch.org/docs/2.9/generated/torch.mps.profiler.start.html>
- Apple Accelerate overview: <https://developer.apple.com/accelerate/>
- Apple `vImage.PixelBuffer` creation/reference APIs: <https://developer.apple.com/documentation/accelerate/creating-vimage-pixel-buffers>
- Apple Core ML `torch.export` guide: <https://apple.github.io/coremltools/docs-guides/source/model-exporting.html>
- Apple Core ML graph passes: <https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.passes.defs.html>
- Apple Core ML SDPA op docs: <https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html>
- Apple Core ML palettization overview: <https://apple.github.io/coremltools/docs-guides/source/opt-palettization-overview.html>
- Apple / Hugging Face Core ML Stable Diffusion repo: <https://github.com/apple/ml-stable-diffusion>
- Apple ANE transformer guidance: <https://machinelearning.apple.com/research/neural-engine-transformers>
- "A Crack in the Bark" abstract / record: <https://www.research.ed.ac.uk/en/publications/a-crack-in-the-bark-leveraging-public-knowledge-to-remove-tree-ri>
- "A Crack in the Bark" dataset / implementation record: <https://www.research.ed.ac.uk/en/datasets/a-crack-in-the-bark-leveraging-public-knowledge-to-remove-tree-ri/>
- "UnMarker" paper record: <https://arxiv.org/abs/2405.08363>
