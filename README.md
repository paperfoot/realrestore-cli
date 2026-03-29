# realrestore-cli

Agent-friendly image restoration CLI optimized for Apple Silicon. Wraps [RealRestorer](https://github.com/yfyang007/RealRestorer) with MPS/MLX/ANE optimizations.

## Features

- **9 restoration tasks**: deblur, denoise, dehaze, derain, low-light, compression artifacts, moire, lens flare, reflection removal
- **Apple Silicon optimized**: MPS backend, float16, attention/VAE slicing, memory management
- **Multiple backends**: MPS (Metal), MLX, CPU with automatic selection
- **Quantization**: int8/int4 for reduced memory and faster inference
- **AI watermark removal**: Detect and remove invisible watermarks (StegaStamp, Tree-Ring, spectral)
- **Agent-friendly**: JSON output, semantic exit codes, self-describing (`agent-info`)
- **Benchmark suite**: Automated performance comparison across backends

## Install

```bash
# From cargo
cargo install realrestore-cli

# Or build from source
git clone https://github.com/199-biotechnologies/realrestore-cli
cd realrestore-cli
cargo build --release

# Set up Python environment + download models
./target/release/realrestore setup
```

## Usage

```bash
# Restore an image (auto-detect degradation)
realrestore restore photo.jpg -o restored.jpg

# Specific task with options
realrestore restore blurry.png --task deblur --backend mps --steps 20

# Remove AI watermarks
realrestore watermark-remove ai_image.png -o clean.png

# Benchmark
realrestore benchmark --iterations 5

# Agent capabilities
realrestore agent-info
```

## Architecture

Rust CLI (fast startup, structured output) + Python inference backend (PyTorch/diffusers).

Follows the [agent-cli-framework](https://github.com/199-biotechnologies/agent-cli-framework) patterns:
- `agent-info` command for capability discovery
- JSON envelopes when piped, colored output in terminal
- Semantic exit codes (0=success, 1=transient, 2=config, 3=input, 4=rate-limit)
- Skill self-install for Claude/Codex/Gemini

## License

MIT
