# Proven Working MPS Fix Pattern for FLUX-Like DiT Models

Date: 2026-03-29

## Bottom line

I did not find an official FLUX-specific MPS fix in diffusers that solves:

- `float16` on MPS -> black output / NaNs
- full transformer `float32` -> MPS mixed-dtype matmul assert

with a single flag.

What I did find is:

1. A community FLUX-on-MPS example that is explicitly documented as working by loading the whole FLUX pipeline in `float32` on MPS.
2. Official diffusers transformer pipelines that solve mixed component dtypes by explicitly casting every tensor that crosses into the transformer to `self.transformer.dtype`.
3. An official diffusers hook, `enable_layerwise_casting(storage_dtype=..., compute_dtype=...)`, that can keep transformer weights stored in `float16` while upcasting module weights to `float32` for compute.

That means the exact fix pattern for RealRestorer on MPS is:

- do not rely on MPS autocast for `float32` compute
- do not expect `FluxPipeline`-style code to recast prompt embeddings for you
- if the transformer computes in `float32`, cast all transformer inputs to `float32`
- if you need `float16` storage, combine those input casts with `enable_layerwise_casting(storage_dtype=torch.float16, compute_dtype=torch.float32)`

## Direct answers to the checklist

### 1. Search GitHub for how FLUX.1 runs on MPS

The clearest end-to-end FLUX MPS snippet I found is in the `damian0815/compel` README. It shows FLUX on MPS loaded in full `float32` and explicitly notes that lower precision on MPS causes NaNs:

```python
device = "mps"
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.float32,
).to(device)  # bfloat16 causes NaN on MPS
```

This is proven working code, but it is not memory-feasible for a 39 GB transformer.

I did not find an official Hugging Face FLUX example that keeps FLUX weights in `float16` while computing only attention in `float32` on MPS.

### 2. How diffusers handle MPS in their examples/docs

The official diffusers MPS guidance is generic:

- move the pipeline to `"mps"`
- do a warmup pass if needed
- enable attention slicing on MPS

I did not find an official `examples/` script with a FLUX-specific MPS dtype workaround.

### 3. What the diffusers MPS doc says

The current MPS doc recommends:

- `.to("mps")`
- `pipe.enable_attention_slicing()`

It does not document:

- `upcast_attention` for FLUX
- a Flux-specific mixed-dtype workaround
- `torch.autocast("mps", dtype=torch.float32)`

### 4. Is there an `upcast_attention` flag in diffusers?

Yes, but not for FLUX.

Diffusers does have `upcast_attention` in older / UNet-style attention paths. That is real code in diffusers. But `FluxTransformer2DModel` does not expose an `upcast_attention` constructor/config flag, so there is no one-line `upcast_attention=True` fix for `FluxPipeline` or FLUX-like DiT pipelines.

For this problem, `upcast_attention` is not the answer.

### 5. What `FluxPipeline` currently does

Current `FluxPipeline` source still behaves like this:

- `encode_prompt()` casts `prompt_embeds` to `self.text_encoder.dtype`
- it creates `text_ids` using the prompt/text-encoder dtype
- later it passes `pooled_prompt_embeds` and `prompt_embeds` directly into `self.transformer(...)`
- it does not recast those tensors to `self.transformer.dtype` before the transformer call

That is exactly why:

- `transformer=float32`
- `text_encoder outputs=float16`

causes MPS matmul dtype asserts. The pipeline currently preserves the text-encoder dtype boundary.

### 6. Is there a working `torch.autocast` pattern on MPS for fp16 weights + fp32 compute?

No proven one.

I verified locally with `torch 2.9.1`:

- `torch.autocast(device_type="mps", dtype=torch.float16)` works
- `torch.autocast(device_type="mps", dtype=torch.bfloat16)` works
- `torch.autocast(device_type="mps", dtype=torch.float32)` emits a warning and disables autocast

So `torch.autocast("mps", dtype=torch.float32)` is not the solution.

### 7. Does diffusers have an official `upcast_vae` parameter?

There is an `upcast_vae()` method on some SDXL pipelines, but it is deprecated.

Official replacement:

```python
pipe.vae.to(torch.float32)
```

`FluxPipeline` does not expose a special `upcast_vae` parameter.

## The exact source-backed pattern that works

The official diffusers pattern for transformer pipelines is not "upcast attention only". It is explicit boundary casting to the denoiser / transformer dtype.

Examples from current diffusers source:

- SDXL pipelines cast `prompt_embeds` to `self.unet.dtype`
- Bria casts `prompt_embeds` to `self.transformer.dtype`
- Hunyuan Image casts prompt embeddings to `self.transformer.dtype`
- Sana casts both `latent_model_input` and `prompt_embeds` to `transformer_dtype` immediately before the transformer call, then converts `noise_pred` back to `float()`

That is the pattern RealRestorer needs on MPS.

## Why this matters for RealRestorer specifically

The current RealRestorer pipeline in this repo mirrors the Flux problem:

- `_encode_prompt(...)` requests `dtype=self.text_encoder.dtype`
- `txt_ids` are created with `dtype=prompt_embeds.dtype`
- `img_ids` are created with `dtype=prompt_embeds.dtype`
- edit-mode reference latents and sampled noise follow prompt / pipeline dtype paths
- the transformer is called directly with those tensors

Relevant local lines:

- `upstream-realrestorer/diffusers/src/diffusers/pipelines/realrestorer/pipeline_realrestorer.py:393`
- `upstream-realrestorer/diffusers/src/diffusers/pipelines/realrestorer/pipeline_realrestorer.py:613`
- `upstream-realrestorer/diffusers/src/diffusers/pipelines/realrestorer/pipeline_realrestorer.py:623`
- `upstream-realrestorer/diffusers/src/diffusers/pipelines/realrestorer/pipeline_realrestorer.py:634`
- `upstream-realrestorer/diffusers/src/diffusers/pipelines/realrestorer/pipeline_realrestorer.py:466`
- `upstream-realrestorer/diffusers/src/diffusers/pipelines/realrestorer/pipeline_realrestorer.py:516`

So if you upcast only the transformer on MPS, RealRestorer will still feed it tensors in the prompt/text dtype unless you patch that boundary.

## Exact patch pattern for RealRestorer

Use `transformer_dtype`, not `prompt_embeds.dtype`, for every tensor that enters the transformer.

### 1. After prompt encoding, recast prompt embeddings and build ids in transformer dtype

```python
transformer_dtype = torch.float32 if device.type == "mps" else self.transformer.dtype

prompt_embeds = prompt_embeds.to(device=device, dtype=transformer_dtype)

txt_ids = torch.zeros(
    prompt_embeds.shape[0],
    prompt_embeds.shape[1],
    3,
    dtype=transformer_dtype,
    device=device,
)

packed_h = math.ceil(height / 16)
packed_w = math.ceil(width / 16)
img_ids = self._prepare_img_ids(
    batch_size=prompt_embeds.shape[0],
    packed_height=packed_h,
    packed_width=packed_w,
    dtype=transformer_dtype,
    device=device,
    axis0=0.0,
)
```

### 2. Make sampled latents and edit latents match the transformer dtype

```python
if task_type == "edit":
    ref_latents = self._pack_latents(
        ref_latents_tensor.to(device=device, dtype=transformer_dtype)
    )

noise = randn_tensor(
    (
        1,
        self.latent_channels,
        height // self.vae_scale_factor,
        width // self.vae_scale_factor,
    ),
    generator=generator,
    device=device,
    dtype=transformer_dtype,
)
latents = self._pack_latents(noise)
```

### 3. Inside `_denoise_t2i()` / `_denoise_edit()`, cast the full transformer input path

```python
transformer_dtype = torch.float32 if latents.device.type == "mps" else self.transformer.dtype

latent_model_input = latents.to(dtype=transformer_dtype)
if guidance_scale != -1:
    latent_model_input = latent_model_input.repeat(2, 1, 1)

model_input = latent_model_input
if ref_latents is not None:
    ref_model_input = ref_latents.repeat(latent_model_input.shape[0], 1, 1).to(dtype=transformer_dtype)
    model_input = torch.cat([latent_model_input, ref_model_input], dim=1)

t_vec = torch.full(
    (model_input.shape[0],),
    float(t),
    dtype=transformer_dtype,
    device=model_input.device,
)
guidance_vec = torch.full(
    (model_input.shape[0],),
    self.model_guidance,
    dtype=transformer_dtype,
    device=model_input.device,
)

pred = self.transformer(
    hidden_states=model_input,
    encoder_hidden_states=prompt_embeds,   # already cast above
    prompt_embeds_mask=prompt_mask,
    timestep=t_vec,
    img_ids=img_ids,
    txt_ids=txt_ids,
    guidance=guidance_vec,
    return_dict=False,
)[0].float()
```

That is the important fix. Once the transformer is `float32`, all incoming transformer tensors must also be `float32`.

## If you need fp16 storage but fp32 compute

The official diffusers mechanism is:

```python
pipe.transformer.enable_layerwise_casting(
    storage_dtype=torch.float16,
    compute_dtype=torch.float32,
)
```

This is the only official diffusers feature I found that actually matches "keep weights stored in fp16 but compute in fp32".

Important caveat:

- `enable_layerwise_casting()` only changes module weights around forward.
- It does not automatically recast `prompt_embeds`, `txt_ids`, `img_ids`, latents, or guidance vectors.
- You still need the explicit boundary casts shown above.

## What not to do

- Do not rely on `upcast_attention=True` for FLUX / RealRestorer DiT. That flag is not exposed there.
- Do not use `torch.autocast("mps", dtype=torch.float32)`. It disables autocast on current torch.
- Do not upcast only the transformer weights while leaving prompt embeddings / ids / latents in `float16`.
- Do not assume the current `FluxPipeline` source will recast condition tensors to transformer dtype for you.

## Recommended MPS extras

### Attention slicing

Keep this on for MPS:

```python
pipe.enable_attention_slicing()
```

This is the only official MPS-specific diffusers mitigation in the docs. It reduces large attention allocations and is still worth using even after the dtype fix.

### VAE upcast

If decode still produces non-finite or black outputs, upcast the VAE explicitly:

```python
pipe.vae.to(torch.float32)
```

That is the official replacement for deprecated `upcast_vae()`.

## Final recommendation

For RealRestorer on MPS, the safest order is:

1. If memory allows: run the transformer in full `float32` and cast every transformer input tensor to `float32`.
2. If memory does not allow: keep transformer storage in `float16`, enable `enable_layerwise_casting(storage_dtype=torch.float16, compute_dtype=torch.float32)`, and still cast every transformer input tensor to `float32`.
3. Keep attention slicing enabled on MPS.
4. Upcast the VAE to `float32` if decode stability is still a problem.

## Source links

- Compel FLUX-on-MPS example: https://github.com/damian0815/compel/blob/main/README.md
- Diffusers MPS docs: https://huggingface.co/docs/diffusers/en/optimization/mps
- FluxPipeline source: https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux/pipeline_flux.py
- Flux transformer source: https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_flux.py
- Sana pipeline source: https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/sana/pipeline_sana.py
- Hunyuan Image pipeline source: https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/hunyuan_image/pipeline_hunyuanimage.py
- Bria pipeline source: https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/bria/pipeline_bria.py
- `enable_layerwise_casting()` source: https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/modeling_utils.py
- Layerwise casting hook source: https://github.com/huggingface/diffusers/blob/main/src/diffusers/hooks/layerwise_casting.py
- FLUX on MPS issue (earlier rotary dtype fix): https://github.com/huggingface/diffusers/issues/9047
- PyTorch MPS softmax NaN issue: https://github.com/pytorch/pytorch/issues/96602
