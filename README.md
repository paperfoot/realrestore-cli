<div align="center">

# RealRestore CLI

**Restore degraded photos on Apple Silicon. Fast, local, agent-friendly.**

<br />

[![Star this repo](https://img.shields.io/github/stars/paperfoot/realrestore-cli?style=for-the-badge&logo=github&label=%E2%AD%90%20Star%20this%20repo&color=yellow)](https://github.com/paperfoot/realrestore-cli/stargazers)
&nbsp;&nbsp;
[![Follow @longevityboris](https://img.shields.io/badge/Follow_%40longevityboris-000000?style=for-the-badge&logo=x&logoColor=white)](https://x.com/longevityboris)

<br />

[![Rust](https://img.shields.io/badge/Rust-000000?style=for-the-badge&logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-000000?style=for-the-badge&logo=apple&logoColor=white)](https://developer.apple.com/metal/)
[![License: MIT](https://img.shields.io/badge/MIT-green?style=for-the-badge)](LICENSE)

---

A command-line tool for AI image restoration that runs entirely on your Mac. Handles 9 degradation types -- blur, noise, haze, rain, low light, compression artifacts, moire, lens flare, and reflections. Uses diffusion models through Metal (MPS) and MLX backends with float16 and quantization for speed and memory efficiency. Also removes invisible AI watermarks.

[Install](#install) | [How It Works](#how-it-works) | [Features](#features) | [Contributing](#contributing)

</div>

## Why This Exists

Most image restoration tools require cloud GPUs or NVIDIA hardware. If you have an M-series Mac, you have a capable ML accelerator sitting right there. RealRestore CLI wraps the [RealRestorer](https://github.com/yfyang007/RealRestorer) diffusion model with Apple Silicon optimizations so you can restore photos locally -- no uploads, no API keys, no cloud costs.

It also speaks JSON, has semantic exit codes, and self-describes its capabilities via `agent-info`. AI coding agents (Claude, Codex, Gemini) can discover and use it without documentation.

## Before vs After

```
Input: blurry, noisy, or degraded photo
                    |
          realrestore restore
                    |
Output: sharp, clean, restored photo
```

RealRestore uses a FLUX-based diffusion pipeline to reconstruct detail that traditional sharpening filters cannot recover. The model generates plausible high-frequency content based on the degradation type, rather than simply amplifying existing pixels.

## Install

### Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4)
- Rust toolchain (`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`)
- Python 3.10+

### From Cargo

```bash
cargo install realrestore-cli
```

### From Source

```bash
git clone https://github.com/paperfoot/realrestore-cli
cd realrestore-cli
cargo build --release
```

### Setup (Downloads Models)

```bash
realrestore setup
```

This creates a Python virtual environment, installs dependencies, and downloads the RealRestorer model weights from Hugging Face. First run takes a few minutes depending on your connection.

## Quick Start

```bash
# Auto-detect degradation and restore
realrestore restore photo.jpg -o restored.jpg

# Target a specific degradation
realrestore restore blurry.png --task deblur --backend mps

# Fast mode (fewer steps + quantization)
realrestore restore photo.png --steps 10 --quantize int8 -o fast.png

# High quality mode
realrestore restore photo.png --quality high -o best.png

# Remove AI watermarks (StegaStamp, Tree-Ring, spectral)
realrestore watermark-remove ai_image.png -o clean.png

# Benchmark your hardware
realrestore benchmark --iterations 5
```

## How It Works

RealRestore CLI has two layers:

1. **Rust CLI** -- Handles argument parsing, output formatting (colored terminal or JSON), exit codes, and process management. Starts in milliseconds.
2. **Python inference backend** -- Runs the actual diffusion model via PyTorch and the `diffusers` library. Supports MPS (Metal), MLX, and CPU backends.

The CLI spawns the Python backend as a subprocess, passes parameters via command-line arguments, and captures structured JSON output. When piped, all output is machine-readable JSON. In a terminal, you get colored human-friendly output.

### Architecture

```
realrestore restore photo.jpg --task deblur --backend mps
        |
  [Rust CLI: parse args, validate input]
        |
  [Python engine: load model, run inference]
        |
    MPS (Metal) / MLX / CPU
        |
  [Output: restored image + metadata JSON]
```

## Features

### 9 Restoration Tasks

| Task | What It Fixes |
|------|--------------|
| `auto` | Auto-detect the degradation type |
| `deblur` | Motion blur, camera shake |
| `denoise` | Sensor noise, high ISO grain |
| `dehaze` | Fog, haze, atmospheric blur |
| `derain` | Rain streaks and water droplets |
| `low_light` | Underexposed, dark photos |
| `compression` | JPEG artifacts, banding |
| `moire` | Screen capture interference patterns |
| `lens_flare` | Lens flare and light artifacts |
| `reflection` | Glass reflections, unwanted reflections |

### Apple Silicon Optimization

- **MPS backend**: Metal Performance Shaders for GPU inference
- **MLX backend**: Apple's ML framework for optimized tensor ops
- **float16 inference**: Half the memory, faster computation
- **Attention slicing**: Process large images within memory limits
- **VAE slicing**: Decode high-resolution outputs without OOM
- **int8/int4 quantization**: Trade minimal quality for major speed gains

### AI Watermark Removal

Detects and removes invisible watermarks embedded by AI image generators:
- **StegaStamp**: Steganographic watermarks
- **Tree-Ring**: Diffusion-based watermarks
- **Spectral**: Frequency-domain watermarks

### Agent-Friendly Design

Built for AI coding agents following the [agent-cli-framework](https://github.com/paperfoot/agent-cli-framework) patterns:

```bash
# Self-describing capabilities
realrestore agent-info

# JSON output when piped
realrestore restore photo.jpg | jq .

# Semantic exit codes
# 0 = success, 1 = transient (retry), 2 = config error, 3 = bad input
```

### Tiling for High-Resolution Images

Process images larger than GPU memory allows:

```bash
realrestore restore large_photo.jpg --tile --tile-size 512 --tile-overlap 64
```

## Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `--task` | `auto` | Restoration task (see table above) |
| `--backend` | `auto` | Inference backend: `mps`, `mlx`, `cpu` |
| `--quantize` | `none` | Quantization: `int8`, `int4`, `none` |
| `--steps` | `28` | Inference steps (more = higher quality) |
| `--seed` | `42` | Random seed for reproducibility |
| `--quality` | - | Preset: `fast`, `balanced`, `high` |
| `--tile` | `false` | Enable tiling for large images |
| `--tile-size` | `512` | Tile size in pixels |
| `--tile-overlap` | `64` | Tile overlap in pixels |
| `--json` | `false` | Force JSON output in terminal |

## Contributing

Contributions welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions and guidelines.

Areas where help is especially useful:
- CUDA/ROCm backend support
- MLX performance improvements
- New restoration tasks
- Benchmarks on different Apple Silicon chips

## License

[MIT](LICENSE)

---
<div align="center">

Built by [Boris Djordjevic](https://github.com/longevityboris) at [199 Biotechnologies](https://github.com/199-biotechnologies) | [Paperfoot AI](https://paperfoot.ai)

<br />

**If this is useful to you:**

[![Star this repo](https://img.shields.io/github/stars/paperfoot/realrestore-cli?style=for-the-badge&logo=github&label=%E2%AD%90%20Star%20this%20repo&color=yellow)](https://github.com/paperfoot/realrestore-cli/stargazers)
&nbsp;&nbsp;
[![Follow @longevityboris](https://img.shields.io/badge/Follow_%40longevityboris-000000?style=for-the-badge&logo=x&logoColor=white)](https://x.com/longevityboris)

</div>
