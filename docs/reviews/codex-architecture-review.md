# Codex Architecture Review

Date: 2026-03-29

## Executive Summary

The high-level split is correct: keep Rust for CLI UX and Python for model execution. I would **not** replace the Python boundary with Rust FFI or PyO3. The current implementation is not yet the optimal version of that pattern, though. Right now the project is best described as a good control plane wrapped around an incomplete and partly speculative Apple Silicon execution layer.

The biggest gaps are not micro-optimizations:

1. The runtime contract between Rust and Python is still loose and non-hermetic.
2. The Python engine still reloads the pipeline on every call, so benchmarks mostly measure startup and model load.
3. The MPS path contains a few aggressive defaults that are not safe as production defaults.
4. The spec is ahead of the implementation: MLX and ANE are described architecturally, but the actual engine is still overwhelmingly a PyTorch/MPS engine.
5. The current watermark path is a useful heuristic baseline, but it is not sufficient if the goal is robust removal across modern invisible watermark schemes.

If the goal is to become the fastest restoration CLI on Apple Silicon, the highest-ROI work is:

- make the Python runtime hermetic and persistent,
- fix MPS correctness and observability,
- benchmark steady-state separately from cold start,
- reduce diffusion steps with a distilled/few-step student,
- only then invest in Core ML / MLX research backends.

## Current State Vs Spec

The design spec is aspirational and broader than the current implementation.

- The spec promises bundled Python and multiple optimization modules (`docs/superpowers/specs/2026-03-29-realrestore-cli-design.md:19-40`).
- The Rust CLI still resolves whichever `python3.12`, `python3`, or `python` appears first on `PATH` and injects `PYTHONPATH` manually (`src/main.rs:197-215`).
- `setup.py` installs into the active interpreter with `pip install`, but does not create or pin a project-owned virtualenv (`python/realrestore_cli/setup.py:16-45`, `python/realrestore_cli/setup.py:79-99`).
- The spec describes MLX and ANE modules, but the codebase currently has an MPS optimization module only; `backend="mlx"` or `backend="ane"` is still not a real backend abstraction in the engine (`src/main.rs:369-370`, `python/realrestore_cli/engine.py:227-241`).

That mismatch matters because architecture decisions should be judged against the real runtime path, not the target diagram.

## 1. Rust CLI Architecture (`src/main.rs`)

### Is the Python subprocess bridge pattern optimal?

**Yes as a boundary, no as currently implemented.**

I would keep the subprocess boundary for this project because:

- PyTorch, diffusers, safetensors, and Apple-specific ML tooling are all easiest to manage in Python.
- A subprocess gives crash isolation and keeps the Rust CLI small.
- A Rust-embedded Python runtime would make packaging, debugging, and dependency upgrades worse, not better.

### What is wrong with the current bridge

- The runtime is not hermetic. `find_python()` uses the first interpreter on `PATH` (`src/main.rs:197-203`), while `setup.py` installs into whichever interpreter invoked setup (`python/realrestore_cli/setup.py:25-45`). That can easily create “setup succeeded, restore fails” drift.
- `run_python()` spawns a fresh Python process for every command (`src/main.rs:206-236`). For a diffusion CLI this is acceptable for a prototype, but it is incompatible with “fastest” positioning because model load dominates.
- The bridge protocol is weak. Rust scans for the last stdout line beginning with `{` and assumes that is the result (`src/main.rs:228-236`). That is fragile once progress, warnings, or multiline JSON appear.
- The CLI advertises backend choices that are not fully implemented. `agent-info` lists `mlx` as a backend (`src/main.rs:369`), but the engine still treats backends largely as raw device strings (`python/realrestore_cli/engine.py:227-233`).
- The benchmark command is routed through the same bridge, but the Python benchmark runner calls `restore_image()` for each iteration, which reloads the model every time (`python/realrestore_cli/benchmarks/runner.py:31-70`, `python/realrestore_cli/engine.py:239-241`). Those numbers are not representative of warm inference.

### Recommended CLI architecture

Keep Rust as the front-end, but add a **resident local worker**:

```text
realrestore (Rust CLI)
  -> realrestored (local Python daemon over UDS / stdio JSON-RPC)
       -> pipeline cache keyed by (model, backend, quantize)
       -> prompt embedding cache
       -> warm scheduler / benchmark harness
```

Recommended behavior:

- `realrestore restore ...` tries to connect to a local worker first.
- If the worker is not running, Rust starts it and retries.
- Single-shot fallback remains available for constrained environments.

This keeps the good part of the subprocess pattern while removing most of its latency penalty.

### Concrete recommendations

- Replace `find_python()` with a project-owned interpreter path stored in a manifest after setup.
- Create a managed venv under `~/.cache/realrestore-cli/venv` or similar.
- Replace “last JSON line” parsing with a versioned request/response protocol.
- Add `realrestore serve` and `realrestore shutdown`.
- Split benchmark reporting into:
  - cold start latency,
  - warm inference latency,
  - end-to-end CLI latency.

## 2. Python Engine Design (`python/realrestore_cli/engine.py`)

### Overall assessment

The current engine is moving in the right direction. Avoiding `torch.compile` on MPS is the correct default today, and the move toward a dedicated `mps_backend.py` is much better than scattering Apple logic inside `engine.py`.

That said, several details are either unsafe, incomplete, or still unmeasured.

### The good parts

- Avoiding `torch.compile` on MPS is correct for now. The PyTorch MPS tracker still describes `torch.compile` on MPS as an early prototype rather than a production-ready path for end-to-end acceleration. This supports the decision in the engine comments (`python/realrestore_cli/engine.py:6-12`) and MPS module (`python/realrestore_cli/optimizations/mps_backend.py:3-11`).
- Not using CPU offload on a 64 GB unified-memory Mac is directionally correct for the main fast path (`python/realrestore_cli/optimizations/mps_backend.py:129-133`).
- Disabling attention slicing on high-memory Macs is also directionally correct. Diffusers explicitly notes that attention slicing is mainly a memory-pressure mitigation and is least necessary on 64 GB systems.

### The main issues

#### 2.1 The MPS environment defaults are too aggressive

`configure_mps_environment()` sets:

- `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`
- `PYTORCH_MPS_LOW_WATERMARK_RATIO=0.0`
- `PYTORCH_ENABLE_MPS_FALLBACK=1`

(`python/realrestore_cli/optimizations/mps_backend.py:63-80`)

This is not a safe production default.

Why:

- PyTorch documents that `0.0` disables the high-watermark limit and may cause system failure under OOM.
- `LOW_WATERMARK_RATIO=0.0` also disables adaptive commit / garbage collection.
- `PYTORCH_ENABLE_MPS_FALLBACK=1` silently routes unsupported ops to CPU, which is the opposite of what you want in a “fastest” benchmark path.

Recommendation:

- Default to explicit, bounded values like a 0.90-0.98 process fraction or use `torch.mps.set_per_process_memory_fraction()` directly.
- Make “unsafe full-memory mode” opt-in.
- In benchmark mode, disable MPS fallback or at least log every fallback.

#### 2.2 The engine still reloads the pipeline for every request

`restore_image()` calls `load_pipeline()` on every invocation (`python/realrestore_cli/engine.py:239-241`).

That is the single biggest avoidable cost in the current system. It also makes the benchmark suite misleading because `runner.py` repeats that load inside every iteration (`python/realrestore_cli/benchmarks/runner.py:45-52`).

Recommendation:

- Add a module-level pipeline cache in Python immediately.
- Move to a persistent worker after that.

#### 2.3 “Peak memory” is mislabeled

`get_peak_memory_mb()` returns current driver allocation or RSS at the end of the run (`python/realrestore_cli/engine.py:95-111`), not true peak memory.

That means benchmark output is not measuring what it claims to measure.

Recommendation:

- Rename it to `end_memory_mb` unless you actually sample or track a high-water mark.
- For MPS, expose `driver_allocated_memory`, `current_allocated_memory`, and `recommended_max_memory` separately.

#### 2.4 The current quantization path is incomplete

The engine now tries to use Quanto (`python/realrestore_cli/engine.py:170-211`), which is directionally much better than the earlier PyTorch dynamic-quantization-only path. However:

- `python/requirements.txt` does not include `optimum-quanto`, `psutil`, or `PyWavelets` (`python/requirements.txt:1-24`).
- Quantization is applied after the MPS pipeline is already constructed and moved (`python/realrestore_cli/engine.py:142-165`), which is not the cleanest or most testable integration path.
- The current code does not yet separate “quantized transformer” from “full-precision VAE/text encoder” as an explicitly benchmarked configuration matrix.

Recommendation:

- Add `optimum-quanto` to requirements.
- Prefer model/submodel quantization at load time via an explicit quantization config path where possible.
- Treat quantization as a first-class backend variant in benchmarks: `mps-fp16`, `mps-int8-transformer`, `mps-int4-transformer`.

#### 2.5 The upstream pipeline has a dtype mismatch risk on MPS

Your engine correctly chooses `float16` for MPS (`python/realrestore_cli/engine.py:58-69`), but the vendored upstream RealRestorer pipeline still hardcodes `torch.bfloat16` for `ref_latents` and `noise` in its call path (`upstream-realrestorer/diffusers/src/diffusers/pipelines/realrestorer/pipeline_realrestorer.py:586`, `upstream-realrestorer/diffusers/src/diffusers/pipelines/realrestorer/pipeline_realrestorer.py:600`).

That means the current MPS fast path is still exposed to implicit casts or fallback behavior in the hottest part of the denoising loop.

This is the most important correctness issue in the current Apple adaptation.

Recommendation:

- Patch the upstream pipeline so latent/noise dtype is backend-driven rather than hardcoded to `bfloat16`.
- Use one resolved dtype source of truth across engine + pipeline internals.

#### 2.6 The MPS optimization policy is internally inconsistent

`mps_backend.py` comments say VAE tiling should be used only for images larger than 1024, but `optimize_pipeline()` enables it unconditionally if the method exists (`python/realrestore_cli/optimizations/mps_backend.py:119-123`).

That is a red flag because it suggests the code and the benchmark narrative are already diverging.

Recommendation:

- Make tiling conditional on image size.
- Benchmark these combinations separately on M4 Max:
  - no slicing / no tiling
  - VAE slicing only
  - VAE tiling only for >1024
  - attention slicing only under memory pressure

## 3. Optimization Strategy Prioritization On M4 Max 64 GB

### My recommendation

For this specific machine, the priority order should be:

1. **MPS correctness + warm-path optimization**
2. **Algorithmic speedups (step reduction / distillation)**
3. **Selective quantization that actually works on Apple**
4. **Core ML backend for a frozen, shipping-optimized variant**
5. **MLX full port as a research track**

### Why this order

#### 3.1 MPS is the right first backend

It is already the native path of the current codebase and the least disruptive route to a fast product. The biggest speed wins still available here are not framework-level:

- persistent process,
- model cache,
- prompt embedding cache,
- dtype cleanup,
- better memory policy,
- honest benchmarking.

Until those are fixed, it is too early to declare MPS insufficient.

#### 3.2 Distillation matters more than backend churn

For diffusion systems, reducing the number of denoising steps is usually a bigger speed lever than moving from one reasonably optimized runtime to another. This is especially true here because the default path still runs 28 steps (`src/main.rs:49-55`, `python/realrestore_cli/engine.py:220`).

Inference: on this project, a good 6-12 step student is more likely to produce a step-function speedup than an early MLX or ANE port.

#### 3.3 Quantization should be treated as backend-specific, not generic

The design spec treats quantization as a general optimization layer (`docs/superpowers/specs/2026-03-29-realrestore-cli-design.md:73-80`), but on Apple Silicon it is not that simple.

- Bitsandbytes currently shows no MPS support in its own support matrix.
- Quanto is device-agnostic and includes MPS in the documented set of supported devices, so it is the most promising PyTorch-side path.
- Core ML weight palettization is the most mature low-bit path if you commit to a converted backend.

So quantization is important, but only when tied to a concrete runtime path.

#### 3.4 ANE is promising, but not automatically the winner on Mac

Core ML absolutely deserves a serious track. Apple documents `CPU_ONLY`, `CPU_AND_GPU`, `CPU_AND_NE`, and `ALL` compute-unit modes, and Apple’s own Stable Diffusion Core ML repo shows that the best choice is hardware- and model-specific.

Important nuance:

- On iPhone/iPad, `CPU_AND_NE` often wins.
- On Mac-class hardware, Apple’s own published Mac Stable Diffusion numbers in that repo use `CPU_AND_GPU` as the best path for 1024px generation.

So on an M4 Max desktop-class machine, I would frame ANE as part of a **Core ML experimental backend**, not as the immediate default target.

#### 3.5 MLX is high-upside, high-cost

MLX is attractive because of lazy execution, unified memory, and native Apple focus. But there is no cheap “switch to MLX” path for a custom diffusers-style restoration pipeline. A serious MLX effort is a partial reimplementation, not a small optimization patch.

MLX is worth doing if:

- you want a long-term Apple-native execution stack,
- you are willing to own a custom model port,
- profiling proves that PyTorch MPS has hit a real ceiling after the basic fixes above.

## 4. Innovative Ideas Beyond Standard Optimization

These are the highest-value non-obvious ideas for this project.

### 4.1 Cache prompt embeddings aggressively

This project has a tiny fixed prompt set (`python/realrestore_cli/engine.py:26-38`) plus a fixed negative prompt (`python/realrestore_cli/engine.py:40-46`).

That means most prompt encoding work is reusable. Precompute and cache:

- task prompt embeddings,
- negative prompt embeddings,
- prompt masks.

For normal CLI use, most requests should never re-run text encoding.

### 4.2 Add a draft/refine restoration mode

Proposal:

- run a cheap draft pass first, either with fewer steps or a distilled student,
- estimate restoration difficulty from residual energy / perceptual change,
- only escalate hard images or hard tiles to the full model.

This gives you better average latency than optimizing only for worst-case images.

### 4.3 Tile-level adaptive compute

Not every tile in an image is equally hard. A face, glass reflection, and flat sky do not need the same denoising budget.

Proposal:

- compute a degradation difficulty map,
- allocate more steps to hard tiles,
- early-exit easy tiles,
- blend in latent space with overlap windows.

This is more promising than global step reduction alone for restoration workloads.

### 4.4 Distill RealRestorer into a few-step student

This is the biggest strategic speed lever.

The most relevant pattern is a consistency / distilled student that approximates the current 28-step teacher with 2-8 steps. If you want the fastest CLI rather than the most elegant runtime, this should be a core roadmap item rather than a future research note.

### 4.5 Warm-latent parameter sweeps

For benchmark mode and advanced users:

- encode the input image once,
- reuse ref latents,
- sweep steps/guidance/task prompt variants without repeating preprocessing and model load.

That turns benchmarking into a real model study instead of repeated cold starts.

### 4.6 Route easy cases away from the full diffusion model

Some degradations do not need the full RealRestorer path all the time.

Proposal:

- train a tiny degradation router,
- use a classical or lightweight learned model for easy denoise/dehaze cases,
- reserve the full diffusion stack for hard or mixed degradations.

That is how you optimize for user-visible latency, not just per-backend benchmarks.

## 5. Watermark Removal: Is Spectral Analysis Enough?

**No. Spectral analysis is a good baseline, but it is not sufficient as the main strategy.**

### What the current approach does well

- It can detect or suppress periodic/global frequency anomalies.
- It can strip metadata-based provenance markers.
- It is simple, deterministic, and fast.

That is useful and worth keeping.

### Why it is not enough

- Many invisible watermark schemes are not just simple global periodic frequency spikes.
- Blind global suppression will also remove legitimate high-frequency image detail.
- The current remover evaluates PSNR against the original watermarked image (`python/realrestore_cli/watermark/remover.py:204-215`), which is not the right success metric. If the watermark is still present, PSNR can still look “good”.

### Recommended watermark architecture

Use a **two-stage system**:

1. **Detection / classification**
   - metadata,
   - periodic spectral markers,
   - known scheme signatures,
   - “unknown but suspicious” learned detector.

2. **Scheme-specific removal**
   - metadata strip for metadata-only schemes,
   - spectral / DWT notch filtering for periodic marks,
   - learned localizer + inpainting / restoration head for learned or spatially entangled marks,
   - diffusion sanitization only when necessary.

### What I would build

- Keep the current spectral/DWT path as a fast baseline.
- Add a learned watermark-presence classifier.
- Add a learned watermark-sanitizer or mask+inpaint model for the hard cases.
- Evaluate on:
  - watermark detectability after removal,
  - image fidelity to clean reference,
  - task performance after restoration.

That is the level required if “watermark removal” is meant as a real product feature rather than a heuristic convenience function.

## 6. Apple Silicon Optimization Techniques Worth Using

These are the most useful Apple-specific techniques that are not yet fully exploited here.

### 6.1 Dynamic MPS memory governance

Use PyTorch’s MPS memory APIs directly:

- `torch.mps.current_allocated_memory`
- `torch.mps.driver_allocated_memory`
- `torch.mps.recommended_max_memory`
- `torch.mps.set_per_process_memory_fraction`

These should drive tiling/slicing decisions dynamically rather than relying on static assumptions.

### 6.2 MPS signposts and Metal capture

PyTorch now exposes:

- `torch.mps.profiler.start/stop/profile`
- `torch.mps.profiler.metal_capture`

Use them in a `--profile-mps` mode and store traces alongside benchmark results. This is the shortest path to finding real kernel bottlenecks instead of guessing.

### 6.3 `PYTORCH_MPS_FAST_MATH` and `PYTORCH_MPS_PREFER_METAL`

PyTorch documents both environment variables:

- `PYTORCH_MPS_FAST_MATH=1` can trade precision for speed.
- `PYTORCH_MPS_PREFER_METAL=1` prefers Metal kernels over MPS Graph for matmul.

These should be exposed as experimental CLI flags and benchmarked on RealRestorer’s transformer blocks.

### 6.4 `torch.mps.compile_shader` for custom hot kernels

PyTorch now documents `torch.mps.compile_shader`, which means you can write custom Metal kernels without leaving Python entirely.

The most likely beneficiaries here are not the whole diffusion model, but small hot utilities:

- pack/unpack latents,
- tile overlap blending,
- custom normalization / postprocessing,
- watermark DSP kernels.

### 6.5 Core ML attention and palettization tricks

Apple’s `ml-stable-diffusion` repo is a useful precedent:

- hardware-specific compute-unit selection,
- attention implementation variants,
- low-bit palettization for real latency gains,
- just-in-time decompression on modern Apple runtimes.

If you build a Core ML backend, copy that playbook instead of inventing a custom ANE path from scratch.

## 7. What Would Make This Truly The Fastest?

### Phase 1: Fix the architecture you already have

- Add a persistent Python worker.
- Cache pipelines, prompt embeddings, and ref latents.
- Patch the upstream dtype mismatch so MPS is genuinely float16 end-to-end.
- Make benchmark mode fail fast on CPU fallbacks.
- Separate cold-start, warm-start, and steady-state numbers.

### Phase 2: Turn MPS into a real product backend

- Replace hardcoded memory env defaults with adaptive policy.
- Add real backend objects instead of raw backend strings.
- Add Quanto-based transformer-only quantization as an explicit benchmarked mode.
- Make VAE tiling conditional on image size.
- Add a proper representative benchmark corpus instead of a flat synthetic image (`python/realrestore_cli/benchmarks/runner.py:33-40`).

### Phase 3: Chase the ceiling

- Train a few-step distilled student.
- Build a Core ML backend for a frozen shipping model.
- Benchmark `CPU_AND_GPU`, `CPU_AND_NE`, and `ALL` on actual Mac hardware rather than assuming ANE wins.
- Add learned watermark sanitization.

### If I had to pick only three bets

1. Persistent worker + pipeline/prompt cache
2. Distilled 6-8 step model
3. Core ML experimental backend with palettized weights

Those three together are more likely to create a category-leading CLI than an early MLX rewrite.

## Recommended Immediate Changes

1. Make Python runtime hermetic and pinned.
2. Add a resident worker process.
3. Remove unsafe default `MPS_*_WATERMARK_RATIO=0.0`.
4. Patch upstream RealRestorer pipeline dtype handling for MPS.
5. Add missing dependencies: `optimum-quanto`, `psutil`, `PyWavelets`.
6. Rebuild the benchmark harness around warm inference and real images.
7. Move “MLX” and “ANE” from CLI marketing surface to experimental backends until they are real.

## External Sources Consulted

- PyTorch MPS API: https://docs.pytorch.org/docs/stable/mps.html
- PyTorch MPS memory fraction API: https://docs.pytorch.org/docs/stable/generated/torch.mps.set_per_process_memory_fraction
- PyTorch MPS environment variables: https://docs.pytorch.org/docs/stable/mps_environment_variables.html
- PyTorch MPS `torch.compile` tracker: https://github.com/pytorch/pytorch/issues/150121
- Diffusers MPS guidance: https://huggingface.co/docs/diffusers/main/optimization/mps
- Diffusers Quanto docs: https://huggingface.co/docs/diffusers/main/en/quantization/quanto
- Optimum Quanto repo: https://github.com/huggingface/optimum-quanto
- MLX repo: https://github.com/ml-explore/mlx
- Core ML Tools conversion docs: https://apple.github.io/coremltools/docs-guides/source/load-and-convert-model.html
- Core ML Tools palettization docs: https://apple.github.io/coremltools/source/coremltools.optimize.coreml.palettization.html
- Apple Core ML Stable Diffusion reference implementation: https://github.com/apple/ml-stable-diffusion
- Latent Consistency Models paper: https://arxiv.org/abs/2310.04378
