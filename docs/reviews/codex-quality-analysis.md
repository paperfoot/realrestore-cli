# RealRestorer MPS Quality Analysis

## Scope

I reviewed these source files in detail:

- `upstream-realrestorer/diffusers/src/diffusers/pipelines/realrestorer/pipeline_realrestorer.py`
- `upstream-realrestorer/diffusers/src/diffusers/models/realrestorer/model_edit.py`
- `upstream-realrestorer/diffusers/src/diffusers/models/transformers/transformer_realrestorer.py`
- `upstream-realrestorer/diffusers/src/diffusers/models/realrestorer/layers.py`
- `upstream-realrestorer/diffusers/src/diffusers/models/realrestorer/connector.py`
- `upstream-realrestorer/diffusers/src/diffusers/models/realrestorer/state_dict_utils.py`
- `upstream-realrestorer/diffusers/src/diffusers/schedulers/scheduling_realrestorer_flow_match.py`
- `upstream-realrestorer/RealRestorer/inference.py`
- `upstream-realrestorer/examples/basic_source.py`
- local wrapper code in `python/realrestore_cli/engine.py` and `python/realrestore_cli/daemon.py`

I could not run a live MPS inference trace in this environment because `torch.backends.mps.is_available()` is false here, so the conclusions below are source-based.

## Bottom Line

The listed MPS patches do not explain a pure "output looks worse than input" regression by themselves.

The strongest source-level quality mismatch is elsewhere:

1. The local CLI always passes a long text-to-image negative prompt for edit mode.
2. Upstream RealRestorer does not do that for edit mode.
3. In this pipeline, the negative prompt is not a raw text embedding shortcut. It goes through the same Qwen prompt-enhancement path as the positive prompt, with the reference image attached in edit mode.

That makes the local negative prompt choice a very plausible quality degrader.

There are two additional quality-sensitive settings that absolutely matter:

1. `version` must match the checkpoint (`v1.0` vs `v1.1`).
2. `size_level` should stay at the authored default `1024` unless you intentionally change both the working resolution and the scheduler regime.

## Exact Correct Inference Configuration

For source-accurate RealRestorer edit inference, the correct configuration is:

```python
prompt = "<task-appropriate restoration prompt>"
negative_prompt = ""
num_inference_steps = 28
guidance_scale = 3.0
size_level = 1024
timesteps_truncate = 0.93
process_norm_power = 0.4
model_guidance = 3.5
version = "auto"
torch_dtype = torch.bfloat16
vae_dtype = torch.float32
```

Notes:

- `negative_prompt=""` is the official upstream edit default in `RealRestorer/inference.py:133-148`.
- Upstream also shows one edit example with a short edit-specific negative prompt: `"oversmoothed, blurry, low quality"` in `examples/basic_source.py:25-32`.
- The public README example uses `torch_dtype bfloat16`, `num_inference_steps 28`, `guidance_scale 3.0`, seed `42` in `README.md:80-89`.
- For source checkpoints, `model_guidance` defaults to `3.5` and `version` defaults to `auto` in `RealRestorer/inference.py:78-87`.
- Upstream source loading keeps the VAE in `float32` while the transformer and text encoder use the main inference dtype in `pipeline_realrestorer.py:128-138`.

If you are validating quality, do not use reduced-step presets. In the local wrapper, only the `HIGH` preset maps back to `28` steps; `BALANCED` and `FAST` reduce steps to `14` and `8` in `python/realrestore_cli/optimizations/scheduling.py:72-93`.

## Answers To The Specific Questions

### 1. Did changing `_autocast_context` from `@staticmethod` to an instance method break anything?

No, not for MPS.

Why:

- The only in-repo call site is `with self._autocast_context(device):` in `pipeline_realrestorer.py:649`.
- `_autocast_context()` only enables autocast for CUDA in `pipeline_realrestorer.py:437-440`.
- On MPS it always returns `contextlib.nullcontext()`.

So:

- On MPS, this patch is a no-op for inference behavior.
- On CUDA, it changes the autocast dtype from hardcoded `bfloat16` to `self.dtype`.
- The only breakage risk is external code calling `RealRestorerPipeline._autocast_context(device)` as an unbound static helper. I found no such call site in this repo.

### 2. Could replacing hardcoded `bfloat16` with `self.dtype` cause quality issues?

Not in the current main restore path, but it is more subtle than it looks.

Important detail:

- `DiffusionPipeline.dtype` returns the dtype of the first `nn.Module` in the pipeline signature order, not necessarily the transformer, in `diffusers/src/diffusers/pipelines/pipeline_utils.py:577-590`.
- I verified locally that `DiffusionPipeline._get_signature_keys(RealRestorerPipeline)` returns `['processor', 'text_encoder', 'transformer', 'vae']`, so `self.dtype` resolves to `text_encoder.dtype`, not transformer dtype.

Why this does not break the current MPS path:

- The local restore path loads the packaged pipeline with `torch_dtype=dtype` in `python/realrestore_cli/engine.py:158-163`.
- For MPS, that same path chooses `torch.bfloat16` in `python/realrestore_cli/engine.py:63-78`.
- It then upcasts only the VAE to `float32` in `python/realrestore_cli/engine.py:171-182`.

That means in the active restore path:

- `text_encoder.dtype == bfloat16`
- `transformer.dtype == bfloat16`
- `vae.dtype == float32`
- `self.dtype == text_encoder.dtype == bfloat16`

So the patched lines in `pipeline_realrestorer.py:585` and `pipeline_realrestorer.py:590-600` still behave like the original hardcoded `bfloat16`.

What this means practically:

- In the current restore path, this patch is not the likely source of degraded quality.
- But it is now a future footgun: if you ever move the text encoder to `float32` while keeping the transformer in `bfloat16`, `self.dtype` will silently change latent noise and reference latents away from the transformer dtype.

If mixed text-encoder / transformer dtypes are introduced later, the denoiser-facing dtype should come from `self.transformer.dtype`, not `self.dtype`.

### 3. Does CPU `float64` RoPE followed by `.float().to(device)` lose quality?

No meaningful quality regression is visible from the source.

Why:

- The patched `rope()` computes the trig terms on CPU in `float64` and then returns `out.float().to(original_device)` in `layers.py:268-281`.
- The original code already ended with `out.float()`. The patch changes where the `float64` computation happens, not the final RoPE tensor dtype.
- `apply_rope()` immediately casts `q` and `k` to `float32` before combining them with the RoPE tensor in `layers.py:292-305`.

So the effective precision path is still:

1. compute the rotation terms in high precision
2. use `float32` for the actual RoPE application
3. cast the rotated tensors back to the model dtype

This patch is a compatibility fix, not a likely quality regression.

### 4. What are the correct settings for `size_level`, `num_inference_steps`, `guidance_scale`, and `model_guidance`?

The authored settings are:

- `size_level = 1024`
- `num_inference_steps = 28`
- `guidance_scale = 3.0`
- `model_guidance = 3.5`

Evidence:

- `__call__()` defaults: `pipeline_realrestorer.py:539-555`
- upstream CLI defaults: `RealRestorer/inference.py:63-87`
- README example: `README.md:80-89`

Why `size_level=1024` matters more than it looks:

- Edit mode resizes the input to an internal working resolution based on `size_level` in `pipeline_realrestorer.py:357-369` and `pipeline_realrestorer.py:581-585`.
- The output is then resized back to the original size in `pipeline_realrestorer.py:676-678`.
- `size_level` also changes `image_seq_len`, which changes the flow-match timestep schedule in `pipeline_realrestorer.py:642-647` and `scheduling_realrestorer_flow_match.py:28-41`.

For the authored default:

- `size_level=1024` corresponds to a packed latent sequence length around `4096`, which matches the scheduler's `max_image_seq_len=4096` regime in `scheduling_realrestorer_flow_match.py:47-55`.

Implication:

- Lowering `size_level` is not just a resolution reduction. It also changes the timestep schedule.
- Raising it above the authored regime likewise changes the schedule, not just detail level.

### 5. Do `process_norm_power` and `timesteps_truncate` matter?

Yes. They are part of the authored denoising behavior.

Defaults:

- `timesteps_truncate = 0.93`
- `process_norm_power = 0.4`

Evidence:

- `pipeline_realrestorer.py:552-554`
- used in both `_denoise_edit()` and `_denoise_t2i()` in `pipeline_realrestorer.py:477-486` and `pipeline_realrestorer.py:526-535`

What they do:

- For early, high-noise steps where `t > 0.93`, the CFG difference is norm-normalized before being applied.
- `process_diff_norm()` in `pipeline_realrestorer.py:348-355` dampens large guidance deltas by dividing by `diff_norm ** 0.4` when `diff_norm > 1`.
- Once `t <= 0.93`, the pipeline switches back to plain CFG.

With the authored `28`-step, `size_level=1024` schedule, this normalization applies to the earliest several steps. It is not decorative tuning.

The local wrapper does not override these values, so the current code is already using the correct authored defaults.

### 6. Does the model need a specific `negative_prompt`?

For edit mode, no long default negative prompt is required, and the current local default is probably wrong for quality.

Upstream behavior:

- In official CLI inference, if `image is not None`, `negative_prompt` defaults to `""` in `RealRestorer/inference.py:133-136`.
- The only upstream edit example that does use a negative prompt uses a short one: `"oversmoothed, blurry, low quality"` in `examples/basic_source.py:25-32`.

Why the local wrapper is risky:

- Local restore code always passes the long text-to-image negative prompt in `python/realrestore_cli/engine.py:268-289`.
- The daemon path does the same in `python/realrestore_cli/daemon.py:146-154`.
- `_encode_prompt()` always builds `prompt_batch = [prompt, negative_prompt]` in `pipeline_realrestorer.py:371-395`.
- `_get_qwenvl_embeds()` runs both strings through the Qwen chat-template prompt-enhancement path, with the reference image attached in edit mode, in `pipeline_realrestorer.py:249-310`.
- The prefix explicitly asks Qwen to generate an "Enhanced prompt" for image generation in `pipeline_realrestorer.py:21-28`.

That means the local long negative prompt is not acting like a trivial empty unconditional branch. It becomes a semantically rich, Qwen-processed negative branch for CFG.

Recommendation:

- For edit mode, default to `negative_prompt=""`.
- If you want an edit-specific negative prompt, use something short like upstream's `"oversmoothed, blurry, low quality"`.
- Do not use the long text-to-image negative prompt as the edit-mode default.

This is the most plausible source-level explanation for "quality is worse than the input" after the NaN issue was fixed.

### 7. Does the `v1.0` vs `v1.1` version check affect quality?

Yes, absolutely.

The version changes two real inference behaviors:

1. Connector scaling:
   - `v1.0` has `connector.scale_factor` in `connector.py:365-377`
   - `v1.1` does not
2. Reference-image positional axis:
   - edit mode uses `ref_axis = 0.0` for `v1.0` and `1.0` for `v1.1` in `pipeline_realrestorer.py:628-639`

Version detection:

- Source checkpoints are auto-detected as `v1.0` if `connector.scale_factor` exists, else `v1.1`, in `state_dict_utils.py:93-101` and `pipeline_realrestorer.py:91-94`.

Recommendation:

- For source checkpoints, keep `version="auto"`.
- Do not force `v1.0` or `v1.1` unless you know the checkpoint layout.

Using the wrong version can degrade quality even if everything else is numerically stable.

## What Is Most Likely Wrong In The Current Local Behavior

In order of likelihood from the source:

1. **Wrong default negative prompt for edit mode**
   - Upstream edit default is empty.
   - Local wrapper always supplies the long text-to-image negative prompt.
   - Because negative prompts are Qwen-enhanced here, this mismatch is stronger than in a normal tokenizer-only CFG pipeline.

2. **Expecting native-resolution improvement while always working at `size_level=1024`**
   - The pipeline resizes the image to a `1024`-target working resolution and then resizes back.
   - For large inputs, this can look softer even when the model is behaving as designed.

3. **Wrong checkpoint version**
   - Wrong `v1.0` / `v1.1` changes both connector scaling and positional IDs.

4. **Any future path that actually uses `float16` on MPS**
   - The active restore path currently chooses `bfloat16` in `engine.py`.
   - But `python/realrestore_cli/optimizations/mps_backend.py:90-97` still claims the optimal MPS dtype is `float16`.
   - That helper is not used by the main restore path today, but if it starts being used, it is a real quality risk for a model authored around `bfloat16`.

## Recommended Validation Configuration

If you want a clean quality baseline on the current code, compare MPS output using exactly this configuration:

```python
pipe(
    image=image,
    prompt="Restore the details and keep the original composition.",
    negative_prompt="",
    num_inference_steps=28,
    guidance_scale=3.0,
    size_level=1024,
    seed=42,
)
```

And for source-loading:

```python
RealRestorerPipeline.from_realrestorer_sources(
    realrestorer_load=...,
    model_path=...,
    device="mps",
    dtype=torch.bfloat16,
    version="auto",
    model_guidance=3.5,
)
```

with the VAE kept in `float32`, matching `pipeline_realrestorer.py:128-138`.

## Final Conclusion

From the source alone:

- `_autocast_context` did not break MPS quality.
- CPU `float64` RoPE did not create a meaningful quality regression.
- `bfloat16 -> self.dtype` did not change the active MPS restore path because `self.dtype` still resolves to `bfloat16` there.

The most likely config-level reason the current MPS output looks worse than the input is the local edit-mode negative prompt policy, followed by version mismatch or resolution/schedule mismatch from non-baseline settings.
