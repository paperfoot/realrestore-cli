---
name: realrestore
description: Restore degraded images (blur, noise, haze, rain, compression, low-light, moire, lens flare, reflections) and remove invisible AI watermarks. Optimized for Apple Silicon with MPS/MLX backends.
---

# realrestore — Image Restoration CLI

Agent-friendly CLI for high-quality image restoration using diffusion models, optimized for Apple Silicon.

## Quick Reference

```bash
# Restore an image (auto-detect degradation type)
realrestore restore input.png -o output.png

# Specific task
realrestore restore blurry.jpg --task deblur -o sharp.jpg

# Fast mode (fewer steps, lower quality)
realrestore restore input.png --steps 10 --quantize int8

# Remove AI watermarks
realrestore watermark-remove image.png -o clean.png

# Benchmark performance
realrestore benchmark --iterations 5

# Full capabilities
realrestore agent-info
```

## Tasks

`auto` | `deblur` | `denoise` | `dehaze` | `derain` | `low_light` | `compression` | `moire` | `lens_flare` | `reflection`

## Backends

`auto` (recommended) | `mps` (Metal) | `mlx` (Apple MLX) | `cpu`

## Exit Codes

- `0` Success
- `1` Transient error (retry)
- `2` Configuration error (run `realrestore setup`)
- `3` Bad input (check arguments)
