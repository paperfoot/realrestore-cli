# RealRestorer Quality Investigation

**Date:** 2026-03-29
**Status:** ROOT CAUSES IDENTIFIED -- 3 critical bugs, 1 major misconfiguration

## Executive Summary

RealRestorer is producing output that is worse than the input on Apple Silicon MPS.
The output is not black (that was fixed previously), but quality is degraded compared
to the original image. Investigation identified **three critical bugs and one major
misconfiguration** that compound to produce poor output.

---

## Bug #1: WRONG NEGATIVE PROMPT (Critical -- Quality Destroyer)

### The Problem

Our engine passes `DEFAULT_NEGATIVE_PROMPT` (a long string of quality-reducing terms)
for image editing/restoration mode. **The upstream code explicitly uses an empty string
for editing mode.**

### Evidence

**Upstream inference.py (lines 134-136):**
```python
negative_prompt = args.negative_prompt
if negative_prompt is None:
    negative_prompt = "" if image is not None else DEFAULT_T2I_NEGATIVE_PROMPT
```

This clearly shows:
- Image editing (restoration): `negative_prompt = ""` (EMPTY STRING)
- Text-to-image (no input image): `negative_prompt = DEFAULT_T2I_NEGATIVE_PROMPT`

**Our engine.py (lines 271-278):**
```python
result = pipe(
    ...
    negative_prompt=DEFAULT_NEGATIVE_PROMPT,  # WRONG for editing mode!
    ...
)
```

We unconditionally pass the full negative prompt for ALL modes, including editing.

**HuggingFace model card** confirms: `negative_prompt=""` in all examples.

**ComfyUI implementation** confirms: defaults to empty string `""` for restoration.

### Why This Destroys Quality

The pipeline uses classifier-free guidance (CFG). In `_denoise_edit` (line 478):
```python
cond, uncond = pred.chunk(2, dim=0)
pred = uncond + guidance_scale * (cond - uncond)
```

The `prompt_embeds` tensor is created from `[prompt, negative_prompt]` -- a batch of 2.
The model runs both through the transformer, producing `cond` (from prompt) and
`uncond` (from negative_prompt). CFG then steers the prediction AWAY from the negative
prompt's direction.

When the negative prompt is `""`, `uncond` is a neutral unconditional prediction --
standard CFG behavior. When the negative prompt is a long descriptive string about
"worst quality, blurry, low res", the `uncond` prediction is pulled toward those
concepts, and CFG then pushes AWAY from them. For text-to-image this improves quality.
**For image restoration, this fights the restoration process** because the model is
trying to both restore the image AND steer away from a complex semantic direction,
creating artifacts and degradation.

### Fix

```python
# In engine.py restore_image():
negative_prompt_to_use = "" if image is not None else DEFAULT_NEGATIVE_PROMPT
```

---

## Bug #2: WRONG VERSION -- v1.1 vs v1.0 (Critical -- Wrong Image IDs)

### The Problem

The pipeline defaults to `version="v1.1"` in its `__init__`, but the HuggingFace
model is actually `version="v1.0"`. When loaded via `from_pretrained`, the pipeline
gets the wrong version because there is no pipeline-level config file that stores
the version parameter.

### Evidence

**Transformer config.json (from HuggingFace cache):**
```json
{
  "version": "v1.0",
  "guidance_embeds": true,
  "use_mask_token": true
}
```

**Pipeline __init__ defaults (line 41):**
```python
def __init__(self, ..., version: str = "v1.1", ...):
```

The `version` parameter is not stored via `register_to_config` or in any
pipeline-level config file. When `from_pretrained` constructs the pipeline, it passes
the transformer, vae, text_encoder etc. but uses default values for `version`,
`model_guidance`, and `max_length`.

### Why This Destroys Quality

The version controls `ref_axis` in the image ID preparation (line 629):
```python
ref_axis = 0.0 if self.version == "v1.0" else 1.0
```

With v1.0 model but v1.1 code path:
- `ref_axis = 1.0` (WRONG -- should be 0.0 for v1.0)
- The reference image positional IDs get `axis0=1.0` instead of `0.0`
- This creates a mismatch between the positional encoding the model expects and
  what it receives
- The transformer cannot correctly attend to the reference image because its
  positional information is wrong
- Result: the model ignores or misinterprets the reference image

### Fix

Either hardcode the version or detect it from the transformer config:
```python
# In engine.py or pipeline loading:
# Option A: Hardcode for known HF model
pipe.version = "v1.0"

# Option B: Detect from transformer config
if hasattr(pipe.transformer, 'config'):
    pipe.version = pipe.transformer.config.get('version', 'v1.0')
```

---

## Bug #3: num_inference_steps=4 (Critical -- Insufficient Denoising)

### The Problem

The "WORKING" commit used only 4 inference steps for its test. The upstream
recommended setting is 28 steps.

### Evidence

**WORKING commit message:**
```
Inference (4 steps): 69.1s
```

**Our engine.py default is 28 steps**, which is correct. But the commit that
confirmed "working" only tested with 4. If subsequent testing used lower step
counts (e.g., from quality presets), quality would be severely degraded.

**Upstream README (line 58):**
```
Inference steps: 28
```

**The paper:**
> "a 28-step denoising process" -- noted as a limitation of computational cost

### Why Low Steps Destroy Quality

RealRestorer uses a custom Flow Matching scheduler (`RealRestorerFlowMatchScheduler`).
The scheduler generates timesteps with a time-shift function:
```python
timesteps = torch.linspace(1, 0, num_steps + 1)
mu = get_lin_function(y1=0.5, y2=1.15)(image_seq_len)
timesteps = time_shift(mu, 1.0, timesteps)
```

With 4 steps, you get 5 timestep values (4 intervals). The flow matching ODE is
approximated with only 4 Euler steps -- a very coarse approximation of the continuous
flow trajectory. With 28 steps (29 values, 28 intervals), the ODE integration is much
more accurate.

Additionally, the CFG norm processing applies `process_diff_norm` with power `k=0.4`
only when `t > timesteps_truncate` (default 0.93). With 4 steps, the time-shifted
schedule may only have 1-2 steps above this threshold. With 28 steps, many more steps
benefit from the norm-aware guidance.

**Note:** Our engine.py correctly defaults to 28 steps, but the scheduling.py module
offers presets as low as 4 steps (via task complexity adjustments). The model was NOT
trained with fewer steps -- it specifically requires 28 steps for full quality.

### Impact: RealRestorer is NOT a Turbo/LCM Model

Unlike distilled models (SDXL Turbo, LCM), RealRestorer was trained with a full
28-step schedule. There are no available distilled weights. The scheduling.py presets
that reduce steps to 4-8 using DPM++/UniPC/DDIM schedulers are based on research
for other diffusion models, NOT RealRestorer specifically.

The custom `RealRestorerFlowMatchScheduler` scheduler cannot simply be swapped for
DPM++ or UniPC because:
1. It uses flow matching (velocity prediction), not noise prediction
2. The time-shift function is calibrated for 28 steps
3. The scheduler.step() implements a simple Euler step: `x + (t_next - t) * v`
4. DPM++/UniPC assume noise-prediction parameterization

### Fix

Always use 28 steps. Remove or disable the scheduling presets that go below 28 steps
until a distilled model is available.

---

## Misconfiguration: model_guidance=3.5 vs 3.0 (Major)

### The Problem

The pipeline defaults to `model_guidance=3.5`, but guidance_scale is passed
separately as 3.0 in our engine. These are **two different guidance mechanisms**:

1. `guidance_scale` (our parameter, 3.0) -- classifier-free guidance scale applied
   in the CFG formula: `pred = uncond + guidance_scale * (cond - uncond)`

2. `model_guidance` (pipeline internal, 3.5) -- a guidance embedding fed INTO the
   transformer via `guidance_vec`. This is encoded through a timestep-style sinusoidal
   embedding and injected into the model's conditioning vector.

### Evidence

In the transformer (model_edit.py, lines 157-160):
```python
if self.params.guidance_embed:
    if guidance is None:
        guidance = torch.full((img.shape[0],), 4, ...)
    vec = vec + self.guidance_in(self.timestep_embedding(guidance, 256))
```

The `model_guidance=3.5` value is hardcoded in the pipeline default. This is the
INTERNAL guidance the model was trained with. Changing it could cause subtle quality
shifts.

In the upstream inference.py (line 86):
```python
parser.add_argument("--model_guidance", type=float, default=3.5, ...)
```

So 3.5 is the upstream default. This is likely fine, but worth noting for completeness.

---

## Additional Findings

### _autocast_context Change: NOT a Bug

Our change from `@staticmethod` to instance method is correct:

**Current code (line 437-440):**
```python
def _autocast_context(self, device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=self.dtype)
    return contextlib.nullcontext()
```

This is called on line 649 as `self._autocast_context(device)`. The instance method
approach is correct because:
1. It accesses `self.dtype` for CUDA autocast
2. For MPS, it returns `nullcontext()` which is a no-op -- correct behavior
3. The original `@staticmethod` likely also returned `nullcontext()` for non-CUDA

The MPS code path gets no autocast, which is appropriate since MPS does not support
`torch.autocast` in the same way as CUDA.

### Scheduler Configuration: Correct for MPS

The `RealRestorerFlowMatchScheduler` is device-agnostic. It operates on timestep
values (floats) and performs simple arithmetic. The `step()` method:
```python
prev_sample = sample + (prev_timestep - timestep) * model_output
```
This is a basic Euler step -- no device-specific operations. It works correctly on MPS.

### size_level=512 vs 1024: Quality Impact

The upstream recommends `size_level=1024`:
> "For practical deployment, we recommend using inputs around 1024 x 1024"

Our engine correctly uses `size_level=1024`. Using 512 would:
1. Resize the input to ~512x512 area (smaller)
2. Lose fine details due to downscaling
3. The model was likely trained/tuned for 1024 resolution
4. After restoration, upscaling back introduces interpolation artifacts

### RoPE Fix: Correct

The RoPE computation on CPU with float64 is correct and follows the FLUX proven
pattern:
```python
if pos.device.type == "mps":
    pos = pos.to("cpu")
scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
...
return out.float().to(original_device)
```

MPS does not support float64. Computing RoPE on CPU with float64 and converting
to float32 before returning to MPS is the standard solution.

### VAE Upcast: Correct

The VAE is correctly upcast to float32:
```python
pipe.vae = pipe.vae.to(dtype=torch.float32)
```

GroupNorm in the VAE produces NaN in bfloat16/float16. Float32 VAE is only ~0.6GB
additional memory and prevents decode artifacts.

---

## Root Cause Ranking (by quality impact)

| # | Bug | Impact | Confidence |
|---|-----|--------|------------|
| 1 | Wrong negative_prompt (should be "" for editing) | CRITICAL | 100% -- upstream code is explicit |
| 2 | Wrong version (v1.1 default vs v1.0 model) | CRITICAL | 100% -- HF config says v1.0 |
| 3 | Too few inference steps (4 instead of 28) | CRITICAL* | Depends on how model is invoked |
| 4 | model_guidance not configurable | MINOR | 3.5 is the upstream default |

*Bug #3 may not apply if the engine is always called with default steps=28.

## Recommended Fix Priority

1. **Fix negative_prompt** -- change to empty string for edit mode (1 line fix)
2. **Fix version** -- set `pipe.version = "v1.0"` after loading (1 line fix)
3. **Enforce 28 steps** -- remove or gate the low-step scheduling presets
4. Test with these fixes and compare quality

## What CORRECT Settings Look Like

Based on upstream code, HuggingFace model card, and ComfyUI implementation:

```python
result = pipe(
    image=image,
    prompt="Restore the details and keep the original composition.",
    negative_prompt="",              # EMPTY for restoration
    num_inference_steps=28,          # Full 28 steps
    guidance_scale=3.0,              # CFG scale
    seed=42,
    size_level=1024,                 # 1024x1024 area
)
# Pipeline internally uses:
# version="v1.0"                    # From model config
# model_guidance=3.5                # Internal guidance embedding
# timesteps_truncate=0.93           # Norm processing threshold
# process_norm_power=0.4            # Norm processing power
```

## Sources

- [RealRestorer GitHub](https://github.com/yfyang007/RealRestorer) -- Official code, settings
- [RealRestorer HuggingFace](https://huggingface.co/RealRestorer/RealRestorer) -- Model card, inference examples
- [RealRestorer Paper](https://arxiv.org/html/2603.25502) -- Architecture, training details
- [ComfyUI RealRestorer](https://github.com/StartHua/Comfyui_RealRestorer) -- Third-party implementation confirming settings
- [CFG-Zero*](https://arxiv.org/abs/2503.18886) -- Flow matching CFG improvements
- Upstream `inference.py` -- Empty negative_prompt for edit mode
- Upstream `pipeline_realrestorer.py` -- version/ref_axis logic, CFG implementation
- HuggingFace model cache `transformer/config.json` -- Confirms v1.0, guidance_embeds=true
