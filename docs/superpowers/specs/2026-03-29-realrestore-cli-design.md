# RealRestore CLI Design Spec

## Overview

Agent-friendly Rust CLI wrapping RealRestorer (image restoration via diffusion models), optimized for Apple Silicon (M4 Max, 64GB). Follows the 199 Biotechnologies agent-cli-framework patterns.

## Architecture

```
realrestore-cli (Rust binary)
  ├── agent-info          → JSON capability manifest
  ├── restore <image>     → Run image restoration
  ├── benchmark           → Performance benchmarks (JSON output)
  ├── watermark-remove    → Strip invisible AI watermarks
  ├── degrade             → Apply synthetic degradations (testing)
  ├── skill install       → Install to ~/.claude/skills/ etc
  └── update              → Self-update from GitHub Releases

Python inference backend (invoked as subprocess)
  ├── engine.py           → Core inference orchestration
  ├── optimizations/
  │   ├── mps_backend.py  → Apple Metal Performance Shaders
  │   ├── quantize.py     → 4-bit/8-bit quantization
  │   ├── mlx_backend.py  → MLX framework acceleration
  │   ├── ane.py          → Apple Neural Engine offload
  │   ├── memory.py       → Attention/VAE slicing, tiling
  │   └── compile.py      → torch.compile for MPS
  ├── watermark/
  │   ├── detector.py     → Detect invisible watermarks
  │   └── remover.py      → Remove/neutralize watermarks
  └── benchmarks/
      └── runner.py       → Automated benchmark suite
```

## Key Design Decisions

1. **Rust CLI + Python backend**: Rust for fast startup, structured output, agent UX. Python for ML inference (PyTorch/diffusers ecosystem).
2. **Apple Silicon first**: MPS backend, float16 (not bfloat16), unified memory, optional ANE/MLX.
3. **Progressive optimization**: Start with baseline MPS, then layer quantization, MLX, ANE, compile optimizations.
4. **Bundled Python**: CLI embeds/manages a Python virtualenv with all dependencies.

## CLI Interface

### `realrestore restore`
```bash
realrestore restore input.png -o output.png \
  --task deblur \
  --quality high \
  --backend mps \       # mps | mlx | ane | auto
  --quantize int8 \     # none | int8 | int4
  --steps 28 \
  --seed 42
```

### `realrestore watermark-remove`
```bash
realrestore watermark-remove input.png -o clean.png \
  --method spectral   # spectral | diffusion | ensemble
```

### `realrestore benchmark`
```bash
realrestore benchmark --iterations 5 --backends mps,mlx
```

### Output format
- Terminal: colored progress + result path
- Piped: JSON envelope `{"version":"1","status":"success","data":{...}}`
- Errors: JSON on stderr with `code`, `message`, `suggestion`

## Optimization Strategy (Priority Order)

1. **MPS baseline** — Get working on Apple Silicon with float16
2. **Memory optimization** — Attention slicing, VAE tiling, CPU offload scheduling
3. **Quantization** — bitsandbytes/GPTQ int8/int4 for model components
4. **torch.compile** — MPS backend compilation
5. **MLX conversion** — Convert model components to MLX for faster inference
6. **ANE offload** — CoreML conversion for ANE-compatible operations
7. **Reduced steps** — Distillation or scheduler optimization (fewer steps, same quality)
8. **Tiling** — Process large images in tiles with overlap blending

## Watermark Removal (Research Required)

Target invisible watermarks:
- **Spectral analysis** — Frequency domain detection/removal
- **Diffusion-based** — Re-encode through restoration pipeline (natural byproduct)
- **StegaStamp/Tree-Ring** — Known AI watermark schemes
- **C2PA metadata** — Strip metadata-based provenance

## Benchmark Suite

Metrics per run:
- Wall time (seconds)
- Peak memory (MB)
- PSNR / SSIM vs reference
- Tokens consumed (for agent cost tracking)

## Success Criteria

- Restore a 1024x1024 image in <60s on M4 Max
- Peak memory <48GB (leave headroom on 64GB)
- CLI startup <100ms
- All 9 restoration tasks functional
- Watermark removal functional for at least 2 schemes
- Automated benchmarks with JSON output
