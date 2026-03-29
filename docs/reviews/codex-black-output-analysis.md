# RealRestorer Black Output on Apple Silicon MPS (`float16`)

Date: 2026-03-29

## Verdict

This is not primarily caused by the three line-level patches themselves.

The black image is the symptom of non-finite values (`NaN`/`Inf`) reaching the final image tensor on the MPS `float16` path. The strongest root cause is that the CLI currently loads the entire packaged pipeline with `torch_dtype=torch.float16`, which downcasts the VAE and Qwen text encoder as well as the denoiser. On MPS, that removes the mixed-precision boundary that upstream already uses in `from_realrestorer_sources()`.

The exact fix is:

- keep `transformer` on MPS in `float16`
- keep `vae` on MPS in `float32`
- keep `text_encoder` in `float32` (safest on CPU or staged, second-best on MPS `float32`)
- do not use a single global `torch_dtype=torch.float16` for the whole pipeline on MPS

## Direct answers to the requested checks

### 1. `pipeline_realrestorer.py`: `__call__`, VAE encode/decode, denoise loop

Relevant code:

- `upstream-realrestorer/diffusers/src/diffusers/pipelines/realrestorer/pipeline_realrestorer.py:147-149`
- `upstream-realrestorer/diffusers/src/diffusers/pipelines/realrestorer/pipeline_realrestorer.py:397-430`
- `upstream-realrestorer/diffusers/src/diffusers/pipelines/realrestorer/pipeline_realrestorer.py:585-600`
- `upstream-realrestorer/diffusers/src/diffusers/pipelines/realrestorer/pipeline_realrestorer.py:649-678`

Observations:

- `_tensor_to_pil()` converts the decoded tensor to `uint8` with `.astype(np.uint8)`.
- `_encode_vae_image()` casts the input image to `self.vae.dtype` before `vae.encode(...)`.
- `_decode_vae_latents()` casts latents to `self.vae.dtype` before `vae.decode(...)`.
- The denoise loop runs with `latents`, `ref_latents`, and `noise` in `self.dtype`.

Important detail:

- `self.dtype` is the pipeline dtype from `DiffusionPipeline.dtype`, which returns the dtype of the first module in the pipeline, i.e. the `transformer`, not the VAE.
- See `upstream-realrestorer/diffusers/src/diffusers/pipelines/pipeline_utils.py:578-589`.

That means the `bfloat16 -> self.dtype` patch is actually correct for denoiser latents, as long as component dtypes are set correctly.

### 2. `autoencoder.py`

Relevant code:

- `upstream-realrestorer/diffusers/src/diffusers/models/realrestorer/autoencoder.py:236-240`
- `upstream-realrestorer/diffusers/src/diffusers/models/realrestorer/autoencoder.py:288-294`

Observations:

- The VAE uses GroupNorm, Conv2d, attention, and `exp(0.5 * logvar)` in the latent sampler.
- This is exactly the kind of module that is sensitive to half precision on MPS.
- Encode/decode do not upcast internally; they run in whatever dtype the VAE weights are loaded in.

### 3. Are there `float16` + MPS NaN risks here?

Yes.

This code path is numerically fragile on MPS in `float16` because:

- the VAE contains GroupNorm + convolution-heavy blocks
- the transformer contains many `LayerNorm` calls (`layers.py:394-405`, `397-408`, `499`, `550`)
- the scheduler does not create NaNs by itself; it only propagates non-finite model output (`scheduling_realrestorer_flow_match.py:92-109`)

So if a module emits `NaN` once, the scheduler and later decode will preserve and spread it.

### 4. Does `RuntimeWarning: invalid value encountered in cast` mean latents contain `NaN`/`Inf`?

It proves the tensor passed into `_tensor_to_pil()` contains non-finite values.

Because the warning is emitted at:

- `pipeline_realrestorer.py:149`

and the only tensor being converted there is `decoded` after:

- `decoded = self._decode_vae_latents(...)`
- `decoded = decoded.clamp(-1, 1).mul(0.5).add(0.5)`

the final decoded image tensor is non-finite at that point.

Strictly speaking, this warning does not prove whether the first non-finite values appeared:

- in `ref_latents_tensor` from VAE encode
- in the denoise loop
- or in VAE decode

But it does prove the output image tensor is `NaN`/`Inf`, and `np.uint8` casts those to zero, which explains the black image.

### 5. Did the applied patches break this?

#### Patch: `torch.bfloat16 -> self.dtype` at pipeline lines 586 and 600

Not the root bug.

This patch is necessary for MPS because hardcoded `bfloat16` is wrong there. Also, because `self.dtype` resolves to the transformer's dtype, it is the right dtype for denoiser latents/noise in a mixed-precision layout.

What broke is the current loading strategy, not these two lines.

#### Patch: `_autocast_context()` changed to use `self.dtype`

Not the root bug.

On MPS this function returns `nullcontext()` anyway:

- `pipeline_realrestorer.py:437-440`

So this patch only affects CUDA.

#### Patch: `rope()` `float64 -> float32`

Not the root bug.

This is a compatibility fix:

- `upstream-realrestorer/diffusers/src/diffusers/models/realrestorer/layers.py:268-277`

MPS does not support `float64` there. This change is required and is not a plausible explanation for an all-black output.

### 6. Is this a patch regression or a `float16` + MPS numerical issue?

It is a `float16` + MPS numerical issue caused by loading the packaged pipeline with one global dtype.

The critical mismatch is:

- upstream source loader explicitly keeps `vae` in `float32`
  - `pipeline_realrestorer.py:130`
- local CLI packaged loader uses `from_pretrained(..., torch_dtype=dtype)` with `dtype=torch.float16` on MPS
  - `python/realrestore_cli/engine.py:159-164`
- diffusers loader then applies that dtype to each submodel during pipeline load
  - `upstream-realrestorer/diffusers/src/diffusers/pipelines/pipeline_loading_utils.py:604-605`

So the CLI path removes upstream's intended VAE precision boundary.

### 7. Does VAE decode produce valid output or `NaN`?

At the point of final image conversion, the decoded image tensor is not valid.

That is guaranteed by the cast warning at `_tensor_to_pil()`.

What we cannot prove from source alone is whether:

- `latents` entering decode are already non-finite, or
- `vae.decode()` is the first place non-finites appear

But because the current MPS packaged path almost certainly runs the VAE in `float16`, VAE encode/decode is the most likely immediate trigger.

## The actual bug in this repo

There is a contradiction in local MPS code:

- `python/realrestore_cli/engine.py:75-79` chooses `torch.float16` for MPS
- `python/realrestore_cli/engine.py:176-185` comments say the MPS strategy should keep sensitive parts out of broken half-precision behavior
- prior repo analysis already recommends:
  - `transformer`: `float16`
  - `vae`: `float32`
  - `text_encoder`: `float32`

The current loader does not enforce that mixed-precision layout.

## Exact fix

Do not load the MPS pipeline with `torch_dtype=torch.float16` globally.

Instead, on MPS:

1. Load the packaged pipeline in `float32`.
2. Move the pipeline to MPS.
3. Explicitly downcast only the transformer to `float16`.
4. Explicitly keep the VAE in `float32`.
5. Keep the text encoder in `float32` as well. Safest is CPU or staged `float32`; second-best is MPS `float32`.

### Recommended code change

In `python/realrestore_cli/engine.py`, replace the MPS load path with component-specific dtypes.

Suggested shape:

```python
if device == "mps":
    pipe = RealRestorerPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        use_safetensors=True,
        low_cpu_mem_usage=True,
    )

    from realrestore_cli.optimizations.mps_backend import configure_mps_environment
    configure_mps_environment()

    pipe.to("mps")

    # Safe mixed precision for Apple Silicon:
    pipe.transformer.to(device="mps", dtype=torch.float16)
    pipe.vae.to(device="mps", dtype=torch.float32)

    # Safest correctness option:
    pipe.text_encoder.to(device="cpu", dtype=torch.float32)
    # Alternative if CPU prompt encoding is too slow:
    # pipe.text_encoder.to(device="mps", dtype=torch.float32)
else:
    pipe = RealRestorerPipeline.from_pretrained(
        model_path,
        torch_dtype=dtype,
        use_safetensors=True,
        low_cpu_mem_usage=True,
    )
```

Why this is the right fix:

- the denoiser still uses `float16` latents/noise because `self.dtype` tracks the transformer dtype
- the VAE no longer runs encode/decode in MPS `float16`
- the Qwen text encoder is no longer forced into MPS `float16`
- the three compatibility patches can remain in place

## Recommended guardrails

Add finite-value checks so this never silently returns a black image again.

Suggested checks in `pipeline_realrestorer.py`:

```python
def _assert_finite(name: str, x: torch.Tensor):
    if not torch.isfinite(x).all():
        raise FloatingPointError(f"{name} contains NaN/Inf")
```

Call it after:

- `ref_latents_tensor = self._encode_vae_image(...)`
- each denoise step update to `latents`
- `decoded = self._decode_vae_latents(...)`

This will distinguish:

- VAE encode failure
- denoiser failure
- VAE decode failure

instead of silently casting `NaN` pixels to black.

## Bottom line

The patches did not break a previously valid MPS path. They only removed obvious dtype incompatibilities.

The real bug is that the packaged MPS loader runs the whole pipeline in `float16`, while upstream's source loader already treats the VAE as `float32`.

Keep:

- `transformer`: `float16`

Force:

- `vae`: `float32`
- `text_encoder`: `float32`

Do that, and the black-output failure mode should disappear without reverting the compatibility patches.
