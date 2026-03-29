# X/Twitter Research: ML Optimization for Diffusion Models on Apple Silicon

**Date compiled:** 2026-03-29
**Scope:** Extensive search across X/Twitter and supporting sources for optimization techniques relevant to running diffusion models on Apple Silicon.

---

## Table of Contents

1. [TurboQuant](#1-turboquant)
2. [Apple Silicon + Diffusion Optimization](#2-apple-silicon--diffusion-optimization)
3. [MPS + PyTorch Optimization](#3-mps--pytorch-optimization)
4. [MLX + Diffusion Models](#4-mlx--diffusion-models)
5. [M4 Max / M5 ML Performance Benchmarks](#5-m4-max--m5-ml-performance-benchmarks)
6. [Unified Memory ML Optimization](#6-unified-memory-ml-optimization)
7. [Quantization on Apple Silicon](#7-quantization-on-apple-silicon)
8. [torch.mps Improvements & Tips](#8-torchmps-improvements--tips)
9. [CoreML + Diffusion](#9-coreml--diffusion)
10. [Metal + Machine Learning](#10-metal--machine-learning)
11. [Key Researcher Posts](#11-key-researcher-posts)
12. [Image Restoration Optimization](#12-image-restoration-optimization)
13. [RealRestorer Mentions](#13-realrestorer-mentions)
14. [ANE / Neural Engine ML Workloads](#14-ane--neural-engine-ml-workloads)
15. [Bonus: Adjacent Discoveries](#15-bonus-adjacent-discoveries)

---

## 1. TurboQuant

### Overview
TurboQuant is a compression algorithm published at ICLR 2026 by Google Research. While primarily targeting LLM KV cache compression, the underlying vector quantization technique has broad implications for any model running in memory-constrained environments like Apple Silicon.

### Key Technical Details
- **Technique:** Randomly rotates input vectors to induce a concentrated Beta distribution on coordinates, then applies optimal scalar quantizers per coordinate.
- **Performance:** 3-bit key quantization with zero accuracy loss; 4-bit achieves up to 8x performance increase over 32-bit on H100.
- **Memory reduction:** 6x reduction in KV cache memory.
- **Training-free:** No fine-tuning required -- works as a drop-in quantization layer.

### Relevance to realrestore-cli
TurboQuant's data-oblivious, training-free approach could potentially be adapted for compressing attention caches in diffusion model U-Net/transformer architectures on Apple Silicon, where memory bandwidth is the primary bottleneck.

### Sources
- [ICLR 2026 Paper (OpenReview)](https://openreview.net/pdf/6593f484501e295cdbe7efcbc46d7f20fc7e741f.pdf)
- [arXiv: TurboQuant](https://arxiv.org/abs/2504.19874)
- [PyTorch implementation (tonbistudio)](https://github.com/tonbistudio/turboquant-pytorch) -- from-scratch implementation, 5x compression at 3-bit with 99.5% attention fidelity
- [llama.cpp Discussion #20969](https://github.com/ggml-org/llama.cpp/discussions/20969)
- [VentureBeat coverage](https://venturebeat.com/infrastructure/googles-new-turboquant-algorithm-speeds-up-ai-memory-8x-cutting-costs-by-50)
- [MarkTechPost coverage](https://www.marktechpost.com/2026/03/25/google-introduces-turboquant-a-new-compression-algorithm-that-reduces-llm-key-value-cache-memory-by-6x-and-delivers-up-to-8x-speedup-all-with-zero-accuracy-loss/)

---

## 2. Apple Silicon + Diffusion Optimization

### X/Twitter Findings

**@zhijianliu_ (Zhijian Liu) -- March 2026**
> "ParoQuant just got a big upgrade. Supports the new Qwen3.5 models. Now runs on MLX (fast local inference on Apple Silicon). Preserves reasoning quality with 4-bit quantization. We also built an agent demo running locally on my 4-year-old M2 Max."
- **Source:** [x.com/zhijianliu_/status/2030395176380956834](https://x.com/zhijianliu_/status/2030395176380956834)
- **Key insight:** 4-bit quantization with pairwise rotation preserves quality on Apple Silicon via MLX. Demonstrates that even older M2 Max hardware can run quantized models effectively.

**@adrgrondin (Adrien Grondin) -- March 2026**
> "The new Qwen 3.5 by @Alibaba_Qwen running on-device on iPhone 17 Pro. The 2B 6-bit model here is running with MLX optimized for Apple Silicon."
- **Source:** [x.com/adrgrondin/status/2028568689709084919](https://x.com/adrgrondin/status/2028568689709084919)
- **Key insight:** MLX is now running quantized models on mobile Apple Silicon (iPhone), showing the framework's maturity.

**@ronaldmannak (Ronald Mannak)**
> "Nvidia renamed their 'Apple Silicon killer' to DGX Spark. Same memory bandwidth as the M4 memory, same price range ($3,000), but more powerful GPU and twice the memory of a Mac mini."
- **Source:** [x.com/ronaldmannak/status/1902081078464250191](https://x.com/ronaldmannak/status/1902081078464250191)
- **Key insight:** Competitive pressure from NVIDIA DGX Spark validates Apple Silicon's position in local ML inference. Memory bandwidth parity is notable.

---

## 3. MPS + PyTorch Optimization

### Critical Tips from Community

**Float32 over Float16 for MPS:**
Apple's GPU architecture does not have the same float16 optimization paths as NVIDIA's Tensor Cores. Precision conversion overhead can actually slow things down. Use float32 on MPS unless specifically testing float16 performance.

**Memory Management Best Practices:**
```python
# Disable watermark ratio for full memory utilization
os.environ.pop('PYTORCH_MPS_HIGH_WATERMARK_RATIO', None)
# Force garbage collection after each generation
torch.mps.empty_cache()
```

**Attention Optimization:**
Use chunked attention with dynamic chunk sizing based on sequence length. Ensure uniform chunk sizes by redistributing leftover elements. Only apply padding when necessary.

**Known Limitations (as of PyTorch 2.11):**
- `torch.compile` support for MPS is still in early stages; complex fusions fall back to CPU or run as unfused generic Metal kernels.
- No native FlashAttention support -- MPS relies on Apple's SDPA implementation.
- "Silent NaN" issue -- MPS is less strict about floating-point exceptions than CUDA; models can train producing garbage values without detection.
- Distributed training is not supported on MPS.
- MPS memory leaks persist even in latest nightly builds.

**MPS on macOS 26 (Tahoe) Issue:**
- **Source:** [GitHub pytorch/pytorch #167679](https://github.com/pytorch/pytorch/issues/167679) -- MPS built but not available on macOS 26 (Tahoe), PyTorch 2.9.1 / 2.10 nightly. Active compatibility issue.

### Sources
- [Apple Developer: Accelerated PyTorch training on Mac](https://developer.apple.com/metal/pytorch/)
- [State of PyTorch Hardware Acceleration 2025](https://tunguz.github.io/PyTorch_Hardware_2025/)
- [Medium: MPS-Accelerated Image Generation Deep Dive](https://medium.com/@michael.hannecke/unleashing-apple-silicons-hidden-ai-superpower-a-technical-deep-dive-into-mps-accelerated-image-9573ba90570a)
- [Medium: Optimizing PyTorch MPS Attention](https://medium.com/@rakshekaraj/optimizing-pytorch-mps-attention-memory-efficient-large-sequence-processing-without-accuracy-5239f565f07b)
- [Edward Yang's blog: State of torch.compile August 2025](https://blog.ezyang.com/2025/08/state-of-torch-compile-august-2025/)

---

## 4. MLX + Diffusion Models

### X/Twitter Findings

**@awnihannun (Awni Hannun, Apple MLX team) -- July 2025**
> "The latest MLX has a CUDA back-end! pip install 'mlx[cuda]'. With the same codebase you can develop locally, run your model on Apple silicon, or in the cloud on Nvidia GPUs."
- **Source:** [x.com/awnihannun/status/1948878861795819662](https://x.com/awnihannun/status/1948878861795819662)
- **Key insight:** MLX now supports CUDA, meaning a single codebase can target both Apple Silicon and NVIDIA GPUs. This is huge for projects like realrestore-cli that need to support multiple backends.

**@awnihannun (Awni Hannun) -- August 2025**
> "Cool exploration of distributed inference on heterogeneous hardware. Including Apple silicon with MLX."
- **Source:** [x.com/awnihannun/status/1953459420622377446](https://x.com/awnihannun/status/1953459420622377446)
- **Key insight:** Heterogeneous distributed inference (mixing different Mac models) is becoming viable.

**@awnihannun (Awni Hannun) -- December 2025**
> "Distributed inference in MLX on Apple silicon will be much faster in Tahoe 26.2"
- **Source:** [x.com/awnihannun/status/1999596403472105975](https://x.com/awnihannun/status/1999596403472105975)
- **Key insight:** macOS Tahoe 26.2 brings significant distributed inference performance improvements for MLX.

**@argmaxinc (Argmax) -- June 2024**
> "DiffusionKit now supports Stable Diffusion 3 Medium -- MLX Python and Core ML Swift Inference work great for on-device inference on Mac!"
- **Source:** [x.com/argmaxinc/status/1800923256603759102](https://x.com/argmaxinc/status/1800923256603759102)
- **Repos:** [github.com/argmaxinc/DiffusionKit](https://github.com/argmaxinc/DiffusionKit), [HuggingFace: argmaxinc/mlx-stable-diffusion-3.5-large](https://huggingface.co/argmaxinc/mlx-stable-diffusion-3.5-large)
- **Key insight:** DiffusionKit provides production-quality on-device diffusion with both MLX (Python) and CoreML (Swift) paths.

### MLX Diffusion Performance (from benchmarks)
- **FLUX-dev-4bit (12B params) on M5:** 3.8x faster than M4 for image generation.
- **GPU Neural Accelerators (M5):** Up to 4x speedup vs M4 baseline for TTFT.
- **MLX v0.31.1** released March 12, 2026 -- latest stable.

### ComfyUI-MLX Extension
- **Performance:** 70% faster model loading, 35% faster generation with loaded models, 30% less memory.
- **Source:** [github.com/thoddnn/ComfyUI-MLX](https://github.com/thoddnn/ComfyUI-MLX)
- **Key insight:** Bypasses PyTorch entirely for diffusion on Apple Silicon. Uses MLX + DiffusionKit backend.

### Sources
- [Apple ML Research: Exploring LLMs with MLX and M5 Neural Accelerators](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)
- [WWDC 2025: Get started with MLX for Apple silicon](https://developer.apple.com/videos/play/wwdc2025/315/)
- [arXiv: Benchmarking On-Device ML on Apple Silicon with MLX](https://arxiv.org/html/2510.18921v1)
- [MLX GitHub](https://github.com/ml-explore/mlx)

---

## 5. M4 Max / M5 ML Performance Benchmarks

### M4 Max (2025)
- **Geekbench 6:** Single-core ~4,000, multi-core ~26,000
- **GPU:** 40-core, up to 128GB unified memory
- **Memory bandwidth:** ~546 GB/s (M4 Max)
- **ML advantage:** Unified memory eliminates CPU-GPU transfer overhead

### M5 (October 2025) and M5 Pro/Max (March 2026)
- **Unified memory bandwidth:** 153 GB/s (M5 base) -- 30% increase over M4, 2x over M1
- **GPU Neural Accelerators:** Dedicated matrix-multiplication units in each GPU core
- **LLM performance:** Up to 6.9x faster prompt processing vs M1 Pro (M5 Max)
- **Image generation:** Up to 8x faster than M1 Max, 3.8x faster than M4 Max (M5 Max)
- **Neural Engine:** 16 cores, hardware support for int8 quantized inference

### Sources
- [Apple Newsroom: M5 Max announcement (March 2026)](https://www.apple.com/newsroom/2026/03/apple-debuts-m5-pro-and-m5-max-to-supercharge-the-most-demanding-pro-workflows/)
- [Apple Newsroom: M5 announcement (October 2025)](https://www.apple.com/newsroom/2025/10/apple-unleashes-m5-the-next-big-leap-in-ai-performance-for-apple-silicon/)
- [Apple M4 Max vs NVIDIA comparison (2026)](https://en.gamegpu.com/News/zhelezo/Apple-M4-Max-vs.-Nvidia:-Where-does-Mac-really-perform-in-2026)
- [Jason Taylor: M4 Max benchmarks](https://jasontaylor.blog/2025/01/13/m4-macbook-pro-max-benchmarks/)

---

## 6. Unified Memory ML Optimization

### Architecture Advantage
Apple Silicon's unified memory architecture eliminates the GPU memory bottleneck. CPU, GPU, and Neural Engine share a single high-bandwidth memory pool, so the GPU accesses model weights without PCIe bus transfer overhead.

### Key Facts
- **M5 bandwidth:** 153 GB/s unified memory bandwidth
- **No explicit copies:** Unlike CUDA where tensors must be copied to GPU VRAM, MPS/MLX operate directly on unified memory
- **Power efficiency:** 3x better performance per watt compared to discrete GPU systems

### Practical Implication for Diffusion Models
Large diffusion models (2-12B parameters) that would exceed discrete GPU VRAM fit naturally in unified memory. A 128GB M4 Max can hold the full FLUX-dev model unquantized, whereas most consumer GPUs max out at 24GB VRAM.

### Sources
- [Apple: Unified Memory Architecture](https://applemagazine.com/apple-unified-memory/)
- [SitePoint: Local LLMs on Apple Silicon 2026](https://www.sitepoint.com/local-llms-apple-silicon-mac-2026/)
- [arXiv: Profiling Apple Silicon Performance for ML Training](https://arxiv.org/pdf/2501.14925)

---

## 7. Quantization on Apple Silicon

### ParoQuant (ICLR 2026)

**@zhijianliu_ (Zhijian Liu) -- March 2026**
> "ParoQuant just got a big upgrade. Now runs on MLX (fast local inference on Apple Silicon). Preserves reasoning quality with 4-bit quantization."
- **Source:** [x.com/zhijianliu_/status/2030395176380956834](https://x.com/zhijianliu_/status/2030395176380956834)
- **Repo:** [github.com/z-lab/paroquant](https://github.com/z-lab/paroquant)
- **Technique:** Learned pairwise rotations to suppress weight outliers; INT4 with near-AWQ speed
- **Key insight:** State-of-the-art 4-bit quantization now has first-class Apple Silicon / MLX support

### Apple's Mixed-Bit Palettization (MBP)
- Selects bit-widths among ANE-supported options (1, 2, 4, 6, 8 bits) per layer
- Achieves average 2.81-bit quantization while maintaining signal quality
- SDXL compressed from 4.8 GB to 1.4 GB (71% reduction) at 4.5 effective bits per parameter
- **Source:** [HuggingFace: SDXL CoreML quantization](https://huggingface.co/blog/stable-diffusion-xl-coreml)

### Apple Intelligence Foundation Models (2025)
- 2-bit quantization-aware training with learnable scaling and EMA smoothing
- Embedding tables quantized to 4-bit, KV caches to 8-bit
- **Source:** [Apple ML Research: Foundation Models Tech Report 2025](https://machinelearning.apple.com/research/apple-foundation-models-tech-report-2025)

### ANE Quantization
- On A17 Pro / M4+, quantizing both activations and weights to int8 leverages optimized Neural Engine compute
- CoreML tools support palettization, linear quantization, and pruning
- **Source:** [Apple: Quantization Algorithms Guide](https://apple.github.io/coremltools/docs-guides/source/opt-quantization-algos.html)

### BFloat16 Status
- `MPSDataType.bFloat16` is supported in Metal Performance Shaders
- MLX supports both bfloat16 and float16 as first step to reduce memory by 50%
- Some compatibility issues remain: InvokeAI reports bfloat16 config option not being used ([Issue #5799](https://github.com/invoke-ai/InvokeAI/issues/5799))

---

## 8. torch.mps Improvements & Tips

### Actionable Optimization Tips

1. **Use float32:** Apple GPU lacks NVIDIA-style float16 Tensor Core acceleration. Conversion overhead can negate gains.

2. **Memory management:**
   ```python
   os.environ.pop('PYTORCH_MPS_HIGH_WATERMARK_RATIO', None)
   torch.mps.empty_cache()  # After each generation
   ```

3. **CPU fallback for unsupported ops:**
   ```bash
   export PYTORCH_ENABLE_MPS_FALLBACK=1
   ```

4. **Chunked attention:** Dynamic chunk sizing based on sequence length with uniform redistribution of remainders.

5. **Avoid torch.compile on MPS:** Complex fusions fall back to CPU or run as unfused generic Metal kernels. Stick to eager mode.

6. **Watch for Silent NaNs:** MPS is less strict about floating-point exceptions. Add NaN checks in training loops.

### Known Issues (Current)
- Memory leak: Usage steadily increases during training (persists in nightly builds)
- macOS 26 Tahoe compatibility issue with PyTorch 2.9.1/2.10 nightly ([#167679](https://github.com/pytorch/pytorch/issues/167679))
- No distributed training support

### Sources
- [PyTorch docs: torch.mps](https://docs.pytorch.org/docs/stable/mps.html)
- [PyTorch Lightning: MPS training](https://lightning.ai/docs/pytorch/stable/accelerators/mps_basic.html)

---

## 9. CoreML + Diffusion

### X/Twitter Findings

**@pcuenq (Pedro Cuenca, HuggingFace) -- 2023 (foundational)**
> "Stable Diffusion XL running on Mac using Core ML and advanced quantization techniques! Open-sourced today: SDXL support in Apple's conversion & inference package. New mixed-bit quantization for Core ML."
- **Source:** [x.com/pcuenq/status/1684726893537632256](https://x.com/pcuenq/status/1684726893537632256)
- **Key insight:** Established the mixed-bit palettization pipeline for CoreML diffusion that remains the gold standard.

**@vkweb3 (VKWeb) -- February 2025**
> "Apple's new Core ML updates now let devs run larger LLMs on-device with better efficiency."
- **Source:** [x.com/vkweb3/status/1951462446876729805](https://x.com/vkweb3/status/1951462446876729805)

**@argmaxinc (Argmax)**
> "DiffusionKit now supports Stable Diffusion 3 Medium -- MLX Python and Core ML Swift Inference work great for on-device inference on Mac!"
- **Source:** [x.com/argmaxinc/status/1800923256603759102](https://x.com/argmaxinc/status/1800923256603759102)
- **Repos:** [github.com/argmaxinc/DiffusionKit](https://github.com/argmaxinc/DiffusionKit)

### CoreML Diffusion Architecture
- CoreML models can leverage ALL compute engines: CPU, GPU, and Neural Engine simultaneously
- Apple's ml-stable-diffusion package: [github.com/apple/ml-stable-diffusion](https://github.com/apple/ml-stable-diffusion)
- Supports palettized quantization (6-bit default, mixed-bit down to ~2.81-bit average)

---

## 10. Metal + Machine Learning

### Metal 4 (WWDC 2025) -- Major Advancement

**@A_SynapseMedia (Artificial Synapse Media) -- June 2025**
> "Apple Unveils Metal 4 with AI-Powered Graphics for iOS, macOS, and Vision Pro at WWDC 2025"
- **Source:** [x.com/A_SynapseMedia/status/1933626701012967813](https://x.com/A_SynapseMedia/status/1933626701012967813)

### Metal 4 Key Features for ML
1. **MTLTensor:** Native multi-dimensional tensor resource type -- first-class ML data type in Metal
2. **MTL4MachineLearningCommandEncoder:** Runs entire ML networks on GPU timeline alongside draws/dispatches
3. **Shader ML:** Run ML networks directly within fragment shaders, no device memory round-trip
4. **Metal Performance Primitives:** Shader primitives that natively operate on tensors
5. **TensorOps framework:** Supports M5 GPU Neural Accelerators

### Practical Impact
- ML inference can now be interleaved with rendering without memory copies
- Neural upscaling, asset compression, animation blending within the render pipeline
- MLX leverages Metal 4 TensorOps for Neural Accelerator support

### Sources
- [WWDC 2025: Discover Metal 4](https://developer.apple.com/videos/play/wwdc2025/205/)
- [WWDC 2025: Combine Metal 4 ML and graphics](https://developer.apple.com/videos/play/wwdc2025/262/)
- [Apple Developer: Machine learning passes](https://developer.apple.com/documentation/metal/machine-learning-passes)

---

## 11. Key Researcher Posts

### @awnihannun (Awni Hannun -- Apple MLX team)

| Date | Post | Key Insight |
|------|------|-------------|
| Jul 2025 | [MLX CUDA backend](https://x.com/awnihannun/status/1948878861795819662) | MLX now dual-targets Apple Silicon + NVIDIA CUDA |
| Aug 2025 | [Distributed heterogeneous inference](https://x.com/awnihannun/status/1953459420622377446) | Mixed hardware clustering with MLX |
| Dec 2025 | [Tahoe 26.2 speedup](https://x.com/awnihannun/status/1999596403472105975) | macOS 26.2 brings major distributed inference gains |

### @_apaszke (Adam Paszke -- PyTorch co-creator)
No recent public posts found specifically about Apple Silicon / MPS optimization in 2025-2026. Adam Paszke has been less active on X regarding MPS-specific topics; his focus appears to be on JAX/XLA and general compiler work.

### @reach_vb (Vaibhav Srivastav -- HuggingFace)
No direct posts found matching Apple Silicon optimization in 2025-2026 search windows. HuggingFace's MPS optimization work is documented in their Diffusers library documentation rather than on X.

### @alexocheema (Alex Cheema -- Exo Labs) -- 2026
> "Running GLM-4.7-Flash on 4 x M4 Pro Mac Minis using @exolabs. Uses tensor parallelism with RDMA over Thunderbolt & MLX backend (h/t @awnihannun). Runs at 100 tok/sec."
- **Source:** [x.com/alexocheema/status/2013694573910937980](https://x.com/alexocheema/status/2013694573910937980)
- **Key insight:** 100 tok/sec on a 4-node Apple Silicon cluster with tensor parallelism. Aiming for 200 tok/sec.

---

## 12. Image Restoration Optimization

### HYPIR (SIGGRAPH 2025) -- 8K Restoration in 1.7 Seconds

**@ChinaScience -- January 2025**
> "China has unveiled an AI image restoration tool called HYPIR that uses advanced diffusion models to enhance low-quality images to 8K resolution with rich details. With powerful cloud computing resources, the tool can restore an image in as fast as 1.7 seconds."
- **Source:** [x.com/ChinaScience/status/1950723085004312978](https://x.com/ChinaScience/status/1950723085004312978)
- **Repo:** [github.com/XPixelGroup/HYPIR](https://github.com/XPixelGroup/HYPIR)
- **Key technique:** Leverages pretrained diffusion model priors but discards iterative sampling; fine-tunes a GAN for single forward pass. Text-guided restoration with adjustable texture richness.

**@sz_mediagroup (Shenzhen Channel) -- January 2025**
> "Turn old photos into 8K in just 1.7 seconds! Shenzhen Institute of Advanced Technology, Chinese Academy of Sciences unveils HYPIR."
- **Source:** [x.com/sz_mediagroup/status/1951500709582574058](https://x.com/sz_mediagroup/status/1951500709582574058)

### Restormer (CVPR 2022 -- Still Referenced)

**@KhanSalmanH (Salman Khan)**
> "We introduce Restormer, an efficient transformer model to deal with high-res images. SOTA results on 16 datasets for image denoising, deraining, motion/defocus deblurring."
- **Source:** [x.com/KhanSalmanH/status/1502254293784698883](https://x.com/KhanSalmanH/status/1502254293784698883)

### Z-Image-Turbo (November 2025)
- 6B parameter model matching quality of FLUX (32B)
- 3x faster than FLUX on same hardware
- 1024x1024 in ~160 seconds on Apple Silicon (ComfyUI)
- 1024x1536: 6-7 seconds vs 18-20 seconds with FLUX
- **Source:** [Medium: Z-Image-Turbo + ComfyUI on Apple Silicon](https://medium.com/@tchpnk/z-image-turbo-comfyui-on-apple-silicon-2026-0aa78d05132d)

---

## 13. RealRestorer Mentions

### X/Twitter Findings

**@HuggingPapers (DailyPapers) -- March 2026**
> "RealRestorer: A unified framework for generalizable real-world image restoration using large-scale image editing models. Handles 9 degradation types -- from blur and noise to rain and reflection -- while preserving natural visual details. Ranks first among open-source methods."
- **Source:** [x.com/HuggingPapers/status/2037504984934240309](https://x.com/HuggingPapers/status/2037504984934240309)
- **Key insight:** RealRestorer is now recognized as #1 among open-source restoration methods. 9 degradation types with a unified framework.

**@aimodelsfyi (cool ai and ml papers) -- March 2026**
> "RealRestorer trains an open-source restorer on a broad real-world degradation set and introduces a benchmark that measures both cleanup and content consistency, aiming for reliable generalization. Probably useful for teams deploying vision in the wild (driving,...)."
- **Source:** [x.com/aimodelsfyi/status/2037728568231461210](https://x.com/aimodelsfyi/status/2037728568231461210)
- **Key insight:** Community recognition of RealRestorer's benchmark for both cleanup quality AND content consistency -- a dual metric approach.

---

## 14. ANE / Neural Engine ML Workloads

### Orion: ANE Reverse Engineering Breakthrough (March 2026)

**@AmbsdOP (Vali Neagu) -- March 2026**
> "YES! Someone reverse-engineered Apple's Neural Engine and trained a neural network on it. Apple never allowed this. ANE is inference-only. No public API, no docs. They cracked it open anyway. M4 ANE = 6.6 TFLOPS/W vs 0.08 for an A100 (80x more efficient)."
- **Source:** [x.com/AmbsdOP/status/2028457255968874940](https://x.com/AmbsdOP/status/2028457255968874940)

**@jbrukh (Jake Brukhman) -- March 2026**
> "Breakthrough reverse engineering of Apple Neural Engine shows that Apple silicon trains models very power-efficiently. The evidence that AI inference and training are coming to your desk is piling up."
- **Source:** [x.com/jbrukh/status/2028639759917052028](https://x.com/jbrukh/status/2028639759917052028)

**@LiorOnAI (Lior Alexander) -- March 2026**
> "Someone just bypassed Apple's Neural Engine to train models. The Neural Engine inside every M-series Mac was designed for inference. Run models, don't train them. No public API, no documentation, and certainly no backpropagation. A researcher reverse-engineered the private..."
- **Source:** [x.com/LiorOnAI/status/2028560569952031145](https://x.com/LiorOnAI/status/2028560569952031145)

**@ronaldmannak (Ronald Mannak) -- March 2026**
> "The Apple Neural Engine in the M4 just got reverse-engineered. Some interesting tidbits: the ANE is ridiculously efficient. CoreML adds 2-4x overhead for small operations. So the big CoreAI update rumored for WWDC..."
- **Source:** [x.com/ronaldmannak/status/2028560995875168292](https://x.com/ronaldmannak/status/2028560995875168292)
- **Key insight:** CoreML adds 2-4x overhead for small operations. Direct ANE access could be significantly faster.

### Orion Project Details
- **Paper:** [arXiv:2603.06728 -- Orion: Characterizing and Programming Apple's Neural Engine](https://arxiv.org/abs/2603.06728)
- **Repo:** [github.com/mechramc/Orion](https://github.com/mechramc/Orion)
- **Foundational repo:** [github.com/maderix/ANE](https://github.com/maderix/ANE)
- **Performance on M4 Max:** 170+ tokens/s for GPT-2 124M inference; 110M transformer trained on TinyStories in 22 minutes (1,000 steps, zero NaN)
- **Key finding:** Apple's 38 TOPS ANE spec is misleading -- INT8 is dequantized to fp16 before computation, yielding ~19 TFLOPS actual throughput
- **32 MB SRAM cliff:** 30% throughput drop when model exceeds ANE's 32 MB SRAM
- **Recompilation optimization:** Weight patching reduces recompilation from 4,200 ms to 494 ms (8.5x faster), yielding 3.8x training speedup

### ANE Practical Limitations
- Designed primarily for inference, not training
- No public API or documentation from Apple
- Bypassing requires using private `_ANEClient` and `_ANECompiler` APIs (may break in future macOS updates)
- Best as component of hybrid AI stack (ANE + GPU + CPU)

---

## 15. Bonus: Adjacent Discoveries

### Exo: Distributed Apple Silicon Clusters

**@exolabs (EXO Labs) -- 2026**
> "Clustering with EXO gets a speed-up with multiple Macs using RDMA tensor parallelism."
- **Source:** [x.com/exolabs/status/2002117326637051974](https://x.com/exolabs/status/2002117326637051974)
- **Repo:** [github.com/exo-explore/exo](https://github.com/exo-explore/exo)
- **Performance:** 1.8x speedup on 2 devices, 3.2x on 4 devices
- **Key tech:** RDMA over Thunderbolt 5 for low-latency tensor parallelism, auto peer discovery
- **Relevance:** Could enable distributed diffusion model inference across multiple Macs

### MLX-Optiq: Quality-Aware Quantization
- PyPI package for optimized quantization targeting MLX
- **Source:** [pypi.org/project/mlx-optiq](https://pypi.org/project/mlx-optiq/)

### FLUX-MLX-ComfyUI
- FLUX nodes for ComfyUI using MLX backend specifically for Apple Silicon
- **Source:** [github.com/CamilleHbp/Flux-MLX-ComfyUI](https://github.com/CamilleHbp/Flux-MLX-ComfyUI)

### Apple's Profiling Paper (January 2025)
- Academic profiling of Apple Silicon for ML training
- **Source:** [arXiv:2501.14925](https://arxiv.org/pdf/2501.14925)

### LLM Quantization Benchmarks for Apple Silicon
- Comprehensive Q4 vs Q6 vs Q8 benchmarks
- **Source:** [SiliconScore: Quantization Guide](https://siliconscore.com/guides/quantization-guide/)

---

## Summary: Top Actionable Insights for realrestore-cli

### Highest Priority

1. **Investigate MLX as alternative backend:** MLX now has CUDA support too, enabling a single codebase for Apple Silicon and NVIDIA. DiffusionKit proves diffusion models run well on MLX. This could replace the PyTorch/MPS path entirely.

2. **Use float32 on MPS, not float16:** Apple Silicon lacks Tensor Core-style float16 acceleration. The conversion overhead can hurt performance. This aligns with our existing MPS dtype fix.

3. **Memory management is critical:** Call `torch.mps.empty_cache()` after each generation. Disable watermark ratio. MPS has known memory leaks.

4. **ANE is 80x more power-efficient than A100:** The Orion project proves ANE training is possible. CoreML adds 2-4x overhead for small ops. Direct ANE access via private APIs is dramatically faster but risky for production.

### Medium Priority

5. **Mixed-Bit Palettization for model compression:** Apple's technique reduces SDXL from 4.8 GB to 1.4 GB. Could be applied to restoration model weights.

6. **ParoQuant 4-bit with MLX:** State-of-the-art quantization now has MLX support. Could enable running larger restoration models on lower-memory Macs.

7. **Metal 4 TensorOps (M5+):** MLX leverages these for up to 4x speedup on M5. Worth targeting as users upgrade hardware.

8. **HYPIR's single-pass restoration:** Discards iterative diffusion sampling in favor of GAN fine-tuning on diffusion priors. 1.7 seconds for 8K restoration. Worth studying their architecture.

### Lower Priority / Future

9. **Exo distributed inference:** Multi-Mac clustering for large models, 3.2x speedup on 4 devices. Relevant if restoration models grow beyond single-machine capacity.

10. **TurboQuant for attention cache:** 3-bit KV cache quantization with zero accuracy loss. Could help if attention memory is a bottleneck.

11. **torch.compile on MPS:** Still immature as of 2026. Do not rely on it.

12. **Z-Image-Turbo architecture:** 6B params matching 32B FLUX quality at 3x speed. The architectural efficiency tricks may be applicable to restoration model design.

---

*Research compiled from 25+ web searches across X/Twitter, academic papers, GitHub repositories, and developer documentation. Last updated: 2026-03-29.*
