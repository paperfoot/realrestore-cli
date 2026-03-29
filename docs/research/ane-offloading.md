# Apple Neural Engine (ANE) Offloading for Diffusion Models

## Research Summary

This document covers ANE capabilities, CoreML conversion workflows, operation compatibility, optimization patterns, hybrid pipeline architectures, and practical recommendations for offloading parts of the RealRestorer inference pipeline to the Apple Neural Engine.

---

## 1. M4 Max ANE Hardware Specifications

| Spec | Value |
|------|-------|
| Cores | 16-core Neural Engine |
| Advertised TOPS | 38 TOPS (INT8) |
| Measured FP16 throughput | ~19 TFLOPS (via direct _ANEClient API) |
| On-chip SRAM | ~32 MB |
| Queue depth | 127 concurrent evaluation requests |
| Power efficiency | ~6.6 TFLOPS/W at peak load |
| Idle power | 0 mW (hard power gating) |
| Dispatch overhead | ~0.095 ms per dispatch |
| First compilation latency | 20-40 ms (cached hits effectively free) |
| DVFS | Independent dynamic voltage/frequency scaling |
| Improvement over M1 Max | >3x faster |
| Improvement over M3 | >2x |

### Critical Performance Findings (Reverse Engineering)

- **INT8 does NOT provide 2x speedup over FP16** — the ANE dequantizes INT8 to FP16 before compute. INT8 saves memory bandwidth only, not compute cycles.
- **SRAM cliff**: Working sets >32 MB spill to DRAM with ~30% throughput degradation (e.g., 4096x4096 matmul drops from 5.7 to 4.0 TFLOPS).
- **1x1 convolutions are ~3x faster than equivalent matmul** — ANE's compute primitive is convolution.
- **CoreML imposes 2-4x overhead** on small operations vs. direct ANE API access.
- **Deep graphs (16-64 chained ops)** utilize hardware far more efficiently than single operations (single ops waste ~70% capacity).

Sources: [Inside the M4 ANE Part 1](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine), [Part 2: Benchmarks](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-615)

---

## 2. ANE-Compatible vs. Incompatible Operations

### Operations That Run on ANE

| Operation | Notes |
|-----------|-------|
| Conv2d | Primary compute primitive, preferred over Linear |
| 1x1 Convolution | ~3x faster than matmul; use for all linear transforms |
| Batched matmul via einsum | Use formula `bchq,bkhc->bkhq` to avoid transposes |
| Elementwise ops | Add, multiply, etc. (non-broadcastable variants) |
| Upsampling | Scaling factor <= 2 only |
| Deconvolution | Supported |
| Activation functions | ReLU, tanh-approximated GELU (NOT standard GELU) |
| Pooling | Kernel size <= 13, stride <= 2 |

### Operations That CANNOT Run on ANE (Fallback to CPU/GPU)

| Operation | Workaround |
|-----------|-----------|
| nn.Linear | Replace with Conv2d + state_dict hooks |
| Custom layers | No public ANE API — always falls back |
| RNN/LSTM/GRU | Not supported |
| Gather operations | Not supported |
| Dilated convolutions | Not supported |
| Broadcastable layers | Replace `AddBroadcastableLayer` with `AddLayer`, etc. |
| ND layers (ConcatND, SplitND) | Decompose into separate programs |
| Concat operation | Causes immediate compilation failure on ANE |
| Standard GELU | Replace with tanh-approximated GELU |
| Pooling (kernel >13, stride >2) | Split or restructure |
| Upsampling (factor >2) | Chain multiple 2x upsample ops |
| Broadcasting (CxHxW * Cx1x1) | Restructure tensor operations |
| Multiple reshape/transpose | Minimize; use einsum patterns instead |
| Convolutions >32K channels | Requires CPU fallback |

### Fallback Behavior

- Core ML can split execution: ANE for supported ops, then switch to GPU/CPU for unsupported.
- **ANE <-> CPU switching** is relatively cheap and can alternate.
- **ANE <-> GPU switching** is expensive; Core ML prefers to do it once rather than alternate.
- Every switch incurs data format conversion overhead despite unified memory.
- If only 1-2 layers are unsupported, consider editing the mlmodel to replace them with ANE-compatible alternatives.

Sources: [hollance/neural-engine unsupported layers](https://github.com/hollance/neural-engine/blob/master/docs/unsupported-layers.md), [ANE vs GPU](https://github.com/hollance/neural-engine/blob/master/docs/ane-vs-gpu.md)

---

## 3. ANE Optimization Patterns

### Tensor Format Requirements

ANE uses **NCDHW + Interleave** format internally:
```
PyTorch (B, S, C) → ANE (B, C, 1, S)
```

- **Critical constraint**: The last axis of an ANE buffer is NOT packed — must be contiguous and aligned to 64 bytes.
- Misalignment causes **32x memory overhead in FP16** or **64x in INT8**.
- For 1024x1024 matrix: represent as `[1, 1024, 1, 1024]` in 4D.

### Multi-Input/Output Programs

- All output buffers must have **identical byte sizes**, ordered alphabetically by MIL variable name.
- All input IOSurfaces must have the **same allocation size**, also ordered alphabetically.
- Data transfer uses **IOSurfaces** (same as GPU textures) — theoretically enables zero-copy ANE<->GPU transfer if sharing the same IOSurfaceRef.

### Four Core Optimization Principles (Apple Research)

1. **Data Format Migration**: Convert `(B, S, C)` to `(B, C, 1, S)` using Conv2d replacements with state_dict hooks for checkpoint compatibility.

2. **Tensor Chunking**: Split query/key/value tensors into explicit single-head attention functions for improved L2 cache residency. Chunk query sequences into blocks of 512 to avoid large intermediate tensors.

3. **Memory Copy Minimization**: Avoid reshapes entirely. Use only one transpose on the key tensor right before the Q*K matmul. Use einsum formula `bchq,bkhc->bkhq` for direct hardware mapping.

4. **Bandwidth Management**: When configurations become bandwidth-bound (latency plateaus across sequence lengths 32-128), increase batch size or reduce parameters via quantization.

### Attention Implementations

| Variant | Target | Description |
|---------|--------|-------------|
| `SPLIT_EINSUM` | ANE (default) | Apple's ANE-optimized attention; splits computation for ANE constraints |
| `SPLIT_EINSUM_V2` | ANE (mobile) | 10-30% improvement over V1 for mobile devices |
| `ORIGINAL` | CPU+GPU | Standard attention; use when NOT targeting ANE |
| `scaled_dot_product_attention_sliced_q` | ANE | New coremltools graph pass; 34% faster, 45% less memory on ANE (Depth-Anything benchmark) |

### Performance Results from Apple's Transformer Work

- DistilBERT on ANE achieved **10x faster** throughput and **14x less memory** consumption.
- Latency: 3.47ms at 0.454W on iPhone 13 (A15 chip).
- Similar principles generalize to Stable Diffusion despite being 19x larger.

Sources: [Deploying Transformers on ANE](https://machinelearning.apple.com/research/neural-engine-transformers), [Stable Diffusion CoreML](https://machinelearning.apple.com/research/stable-diffusion-coreml-apple-silicon)

---

## 4. CoreML Conversion for Diffusion Model Components

### Conversion Pipeline

```python
import coremltools as ct

# Basic conversion from PyTorch traced model
mlmodel = ct.convert(
    traced_model,
    inputs=[ct.TensorType(shape=input_shape)],
    compute_units=ct.ComputeUnit.ALL,           # CPU + GPU + ANE
    # compute_units=ct.ComputeUnit.CPU_AND_NE,  # Force ANE path
    convert_to="mlprogram",                      # Required for latest optimizations
    minimum_deployment_target=ct.target.macOS15,
)
mlmodel.save("model.mlpackage")
```

### Compute Unit Options

| Option | Engines Used | When to Use |
|--------|-------------|-------------|
| `ALL` | CPU + GPU + ANE | Default; CoreML decides optimal split |
| `CPU_AND_NE` | CPU + ANE | Force ANE usage; best for ANE-optimized models |
| `CPU_AND_GPU` | CPU + GPU | When model has many ANE-incompatible ops |
| `CPU_ONLY` | CPU only | Debugging/baseline |

**Warning**: On macOS 26.3, `compute_units=ALL` may route computation to GPU rather than ANE. Use `CPU_AND_NE` to force ANE execution.

### Stable Diffusion Component Split

Apple's ml-stable-diffusion converts the four neural network components separately:

| Component | Size | ANE Suitability | Notes |
|-----------|------|-----------------|-------|
| Text Encoder (CLIP) | ~340M params | Good | Transformer architecture, ANE-friendly with split_einsum |
| UNet | ~860M params | Good (with chunking) | Must split into 2 chunks (<1GB each) for mobile ANE; use `--chunk-unet` |
| VAE Encoder | ~34M params | Moderate | Contains upsampling; some ops may fall back |
| VAE Decoder | ~49M params | Moderate | Contains upsampling and convolutions |

### Conversion Command (ml-stable-diffusion)

```bash
python -m python_coreml_stable_diffusion.torch2coreml \
    --model-version "stabilityai/stable-diffusion-2-1" \
    --output-dir ./coreml_models \
    --attention-implementation SPLIT_EINSUM \  # ANE-optimized
    --chunk-unet \                             # Split for mobile ANE
    --convert-text-encoder \
    --convert-vae-encoder \
    --convert-vae-decoder \
    --convert-unet
```

### Model Format

- Use `mlprogram` format (NOT legacy `neuralnetwork`) for latest ANE optimizations.
- Weights are float16 by default.
- SD3's MMDiT currently requires fp32, so it only supports `CPU_AND_GPU` (NOT ANE).

Sources: [coremltools docs](https://apple.github.io/coremltools/docs-guides/source/convert-pytorch.html), [ml-stable-diffusion](https://github.com/apple/ml-stable-diffusion), [HuggingFace CoreML guide](https://huggingface.co/docs/diffusers/en/optimization/coreml)

---

## 5. Apple's ml-stable-diffusion: Architecture & Split Strategy

### How Apple Splits the Pipeline

The Stable Diffusion pipeline comprises 4 neural networks (~1.275 billion parameters total). Apple converts each component independently to CoreML, allowing Core ML to route operations to the optimal compute unit per component.

### Split Einsum Technique

The key ANE optimization is `SPLIT_EINSUM`:
1. Replace `nn.Linear` layers with `Conv2d` equivalents
2. Reshape tensors from `(B, S, C)` to `(B, C, 1, S)` for ANE's native format
3. Split multi-head attention into explicit single-head operations
4. Use einsum `bchq,bkhc->bkhq` to avoid intermediate transpose/reshape
5. Chunk query sequences into blocks of 512 to prevent large intermediate tensors

### UNet Chunking

For mobile/ANE deployment:
- UNet is split into 2 approximately equal chunks, each <1GB
- Required for ANE deployment on iOS/iPadOS when weights are NOT quantized to 6-bit or less
- On macOS, chunking is optional but can improve memory behavior

### Performance Characteristics

- Swift inference is slightly faster than Python (models already in compiled `.mlmodelc` format)
- Python inference requires uncompiled `.mlpackage` format
- ANE execution benefits from batch compilation at startup

Sources: [ml-stable-diffusion GitHub](https://github.com/apple/ml-stable-diffusion), [HuggingFace blog](https://huggingface.co/blog/diffusers-coreml)

---

## 6. Swift/Python CoreML Inference Patterns

### Python Inference

```python
import coremltools as ct

# Load model
model = ct.models.MLModel("model.mlpackage")

# Run inference
prediction = model.predict({"input": input_data})
```

For diffusion pipelines, use the `python_coreml_stable_diffusion` package:
```python
from python_coreml_stable_diffusion.pipeline import get_coreml_pipe

pipe = get_coreml_pipe(
    pytorch_pipe=pytorch_pipe,
    mlpackages_dir="./coreml_models",
    model_version="stabilityai/stable-diffusion-2-1",
    compute_unit="ALL",  # or "CPU_AND_NE" for ANE
)
image = pipe(prompt="...", num_inference_steps=50)
```

### Swift Inference

```swift
import CoreML
import StableDiffusion

let config = MLModelConfiguration()
config.computeUnits = .all  // or .cpuAndNeuralEngine

let pipeline = try StableDiffusionPipeline(
    resourcesAt: modelURL,
    controlNet: [],
    configuration: config,
    disableSafety: false,
    reduceMemory: false
)

let images = try pipeline.generateImages(
    configuration: pipelineConfig,
    progressHandler: { progress in
        return true  // continue
    }
)
```

### Key Differences

| Aspect | Python | Swift |
|--------|--------|-------|
| Model format | `.mlpackage` (uncompiled) | `.mlmodelc` (compiled, faster) |
| Startup time | Slower (compilation at load) | Faster (pre-compiled) |
| ANE access | Via CoreML framework | Via CoreML framework |
| Flexibility | Higher (easier prototyping) | Lower (but better performance) |
| Integration | Good for conversion/testing | Best for production deployment |

---

## 7. ANE + MPS Hybrid Pipelines

### Architecture Pattern: Disaggregated Inference

The most promising hybrid approach splits work between ANE and GPU based on each accelerator's strengths:

```
[Input] → [Preprocessing (CPU)] → [UNet/Backbone (ANE)] → [Attention/Complex (MPS/GPU)] → [VAE Decode (GPU)] → [Output]
```

### Hybrid ANE-MLX Approach (AtomGradient)

Demonstrated on LLMs but applicable pattern for diffusion:

| Stage | Engine | Rationale |
|-------|--------|-----------|
| Prefill (parallel) | CoreML/ANE | ANE excels at parallel batch processing |
| Sequential decode | MLX/GPU | GPU better for sequential token generation |
| KV-Cache bridge | CPU | Format conversion between frameworks |

**Key finding**: On macOS 26.3, `compute_units=ALL` routes to GPU, not ANE. Genuine ANE batch dispatch via private API achieved 268 tok/s (11.3x speedup over sequential).

### Practical Hybrid Strategy for Diffusion Models

For a restoration pipeline like RealRestorer:

| Component | Recommended Engine | Reasoning |
|-----------|--------------------|-----------|
| Text Encoder | ANE | Transformer; well-suited for split_einsum |
| UNet (denoising backbone) | ANE + GPU hybrid | Conv2d layers on ANE; complex attention on GPU |
| ControlNet / IP-Adapter | GPU (MPS) | Complex conditioning; many unsupported ops |
| VAE Encoder | GPU (MPS) | Upsampling factors, broadcasting |
| VAE Decoder | GPU (MPS) | Same constraints as encoder |
| Image preprocessing | CPU | Simple transforms |
| Post-processing | CPU/GPU | Resize, color correction |

### Challenges

1. **KV-cache format incompatibility** between CoreML and MLX — requires custom bridge
2. **Compilation overhead**: CoreML model compilation can take 2-97 minutes per sequence length variant (one-time, cached)
3. **No unified framework**: Must manage ANE (CoreML) and GPU (MPS/MLX) as separate inference engines
4. **IOSurface overhead**: ~2.3ms per dispatch for ANE, which can negate benefits for small operations

Sources: [hybrid-ane-mlx-bench](https://github.com/AtomGradient/hybrid-ane-mlx-bench), [Anemll](https://github.com/Anemll/Anemll)

---

## 8. Existing Projects & Ecosystem

### Apple ml-stable-diffusion
- Official Apple project for Stable Diffusion on CoreML
- Supports SD 1.x, 2.x; SD3 partially (MMDiT requires fp32/GPU only)
- [GitHub](https://github.com/apple/ml-stable-diffusion)

### DiffusionKit (Argmax)
- Python + Swift packages for diffusion on Apple Silicon
- Supports CoreML and MLX backends
- SD3 support upstreamed from ml-stable-diffusion
- [GitHub](https://github.com/argmaxinc/DiffusionKit)

### Anemll
- Direct ANE access framework for LLM inference
- Supports Llama, Qwen, Gemma, DeepSeek architectures
- ANE Profiler tool (no Xcode required)
- Chunked model conversion, LUT4/LUT6 quantization
- [GitHub](https://github.com/Anemll/Anemll)

### Orion (Research)
- First system for ANE training + inference via reverse-engineered APIs
- Catalogs 20 ANE constraints (14 previously undocumented)
- Delta compilation technique: 8.5x speedup over full recompile
- GPT-2 124M at 170 tok/s decode on ANE (M4 Max)
- [Paper](https://arxiv.org/html/2603.06728v1)

### MochiDiffusion
- Native macOS app running Stable Diffusion via CoreML
- User-friendly GUI with ANE support
- [GitHub](https://github.com/MochiDiffusion/MochiDiffusion)

### hollance/neural-engine
- Community documentation of ANE capabilities and limitations
- Definitive reference for supported/unsupported operations
- [GitHub](https://github.com/hollance/neural-engine)

---

## 9. Orion's ANE Constraint Catalog (20 Constraints)

From the Orion research paper, which catalogs the most comprehensive set of known ANE constraints:

| # | Constraint | Impact |
|---|-----------|--------|
| 1 | Weights baked at compile time | Cannot update weights without recompilation |
| 2 | Concat operation rejected by compiler | Must decompose into separate programs |
| 3 | Causal masks silently ignored in SDPA | Incorrect attention computation without workaround |
| 4 | 1x1 convolutions ~3x faster than matmul | Always prefer Conv2d over Linear |
| 5 | ~119 compilations per process limit | Must manage compilation budget |
| 6 | 32 MB SRAM capacity | Working sets must fit or face 30% degradation |
| 7 | IOSurface-backed I/O required | All tensor transfer via shared memory surfaces |
| 8 | [1, C, 1, S] tensor layout mandatory | No flexible layouts |
| 9 | Dispatch overhead ~0.095ms | Small ops dominated by overhead |
| 10 | Multi-output buffers must have identical sizes | Constrains model architecture |
| 11 | Alphabetical ordering of I/O variables | Non-obvious requirement |
| 12 | Convolutions >32K channels rejected | Requires model splitting |
| 13 | Standard GELU must use tanh approximation | Silent correctness issue |
| 14 | FP16 overflow cascades possible | Requires careful scaling |
| 15-20 | Additional undocumented constraints | See Orion paper for details |

Source: [Orion: Characterizing and Programming Apple's Neural Engine](https://arxiv.org/html/2603.06728v1)

---

## 10. Recommendations for RealRestorer Pipeline

### When ANE Offloading Makes Sense

**Good candidates for ANE**:
- Convolutional backbone layers (the core of most restoration networks)
- Transformer blocks with ANE-optimized attention (split_einsum)
- Lightweight preprocessing networks
- Running multiple inference streams in parallel (ANE queue depth of 127)

**Poor candidates for ANE**:
- VAE decoder (upsampling, broadcasting issues)
- Complex attention mechanisms not adapted for ANE
- Operations requiring dynamic shapes
- Any operation using concat, gather, dilated convolutions
- Models >8B parameters (practical limit)

### Recommended Architecture

```
┌─────────────────────────────────────────────────┐
│              RealRestorer Pipeline               │
├─────────────────────────────────────────────────┤
│                                                  │
│  [Input Image] ──→ [Preprocess (CPU)]            │
│                         │                        │
│                         ▼                        │
│  [Text Encoder] ──→ ANE (split_einsum)           │
│                         │                        │
│                         ▼                        │
│  [UNet Conv Layers] ──→ ANE (Conv2d optimized)   │
│  [UNet Attention]  ──→ MPS/GPU (complex ops)     │
│                         │                        │
│                         ▼                        │
│  [ControlNet]     ──→ MPS/GPU                    │
│                         │                        │
│                         ▼                        │
│  [VAE Decode]     ──→ MPS/GPU                    │
│                         │                        │
│                         ▼                        │
│  [Post-process]   ──→ CPU                        │
│                         │                        │
│                         ▼                        │
│                   [Output Image]                 │
│                                                  │
└─────────────────────────────────────────────────┘
```

### Conversion Workflow

1. **Export PyTorch components** individually (text encoder, UNet, VAE)
2. **Replace nn.Linear with Conv2d** in all ANE-targeted components
3. **Reshape tensors** from `(B, S, C)` to `(B, C, 1, S)`
4. **Replace standard GELU** with tanh-approximated GELU
5. **Convert with coremltools** targeting `CPU_AND_NE` for ANE components
6. **Use SPLIT_EINSUM** attention for transformer blocks
7. **Chunk large models** (<1GB per chunk for mobile, optional on macOS)
8. **Profile with Xcode** Core ML Performance Report or Anemll's ANE Profiler
9. **Benchmark** ANE vs GPU paths to validate actual speedup

### Key Trade-offs

| Factor | ANE | GPU (MPS) |
|--------|-----|-----------|
| Throughput | Lower (19 TFLOPS FP16) | Higher (~depends on model) |
| Power consumption | ~2W | ~20W |
| Memory usage | ~500MB (8B model) | ~8GB (8B model) |
| Startup overhead | High (compilation) | Low |
| Operation support | Limited | Full |
| Framework maturity | Beta/experimental | Stable (PyTorch MPS, MLX) |
| Dynamic shapes | Poor | Good |
| Production readiness | Low | High |

### Practical Bottom Line

**For RealRestorer specifically**:
- ANE offloading is most beneficial for **power-constrained** or **memory-constrained** scenarios
- The primary production path should remain **MPS/GPU** for throughput and stability
- ANE can serve as a **secondary compute path** for specific components (text encoding, lightweight convolution stages)
- The **biggest win** is running text encoder on ANE while UNet runs on GPU — parallel execution on separate compute units
- **Do not** attempt to run the entire pipeline on ANE — the operation restrictions and framework immaturity make this impractical
- Consider ANE offloading as a **Phase 2 optimization** after MPS and MLX paths are stable

### Quantization for ANE

- **A17 Pro / M4+**: INT8 activation + weight quantization leverages optimized ANE compute paths for latency improvement
- **Older chips**: INT8 saves bandwidth only (dequantized to FP16 for compute)
- **LUT4/LUT6 palettization**: Supported by coremltools and Anemll for aggressive model compression
- **Mixed-Bit Palettization (MBP)**: Per-layer bit-width optimization maintaining signal strength — available via `coremltools.optimize.coreml`

---

## References

- [Deploying Transformers on the Apple Neural Engine](https://machinelearning.apple.com/research/neural-engine-transformers)
- [Stable Diffusion with Core ML on Apple Silicon](https://machinelearning.apple.com/research/stable-diffusion-coreml-apple-silicon)
- [ml-stable-diffusion GitHub](https://github.com/apple/ml-stable-diffusion)
- [coremltools GitHub](https://github.com/apple/coremltools)
- [Inside the M4 ANE, Part 1](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine)
- [Inside the M4 ANE, Part 2: Benchmarks](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-615)
- [Orion: Characterizing and Programming Apple's Neural Engine](https://arxiv.org/html/2603.06728v1)
- [hollance/neural-engine](https://github.com/hollance/neural-engine)
- [DiffusionKit (Argmax)](https://github.com/argmaxinc/DiffusionKit)
- [Anemll](https://github.com/Anemll/Anemll)
- [Hybrid ANE-MLX Bench](https://github.com/AtomGradient/hybrid-ane-mlx-bench)
- [HuggingFace CoreML Diffusers Guide](https://huggingface.co/docs/diffusers/en/optimization/coreml)
- [HuggingFace Stable Diffusion CoreML Blog](https://huggingface.co/blog/diffusers-coreml)
- [Apple Neural Engine LLM Inference Guide](https://insiderllm.com/guides/apple-neural-engine-llm-inference/)
- [coremltools Conversion Guide](https://apple.github.io/coremltools/docs-guides/source/convert-pytorch.html)
