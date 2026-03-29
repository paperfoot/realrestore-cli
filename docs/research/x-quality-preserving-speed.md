# Quality-Preserving Diffusion Inference Speed Optimizations

Research compiled from X/Twitter posts and linked papers. Date: 2026-03-29.

Focus: Techniques that reduce diffusion model inference time WITHOUT compromising image quality (preservation or improvement required). Applicability to image restoration assessed for each finding.

---

## Table of Contents

1. [Step Distillation Methods](#1-step-distillation-methods)
2. [Consistency Models](#2-consistency-models)
3. [Adversarial Distillation](#3-adversarial-distillation)
4. [Solver/Sampler Optimization](#4-solversampler-optimization)
5. [Feature Caching (DeepCache)](#5-feature-caching-deepcache)
6. [Token Merging (ToMe / ToMA)](#6-token-merging-tome--toma)
7. [CFG Distillation / Guidance Optimization](#7-cfg-distillation--guidance-optimization)
8. [Compiler/Runtime Optimization](#8-compilerruntime-optimization)
9. [Representation Alignment (REPA)](#9-representation-alignment-repa)
10. [Pipeline-Level Streaming (StreamDiffusion)](#10-pipeline-level-streaming-streamdiffusion)
11. [Restoration-Specific Approaches](#11-restoration-specific-approaches)
12. [Sparse Distillation (Video/DiT)](#12-sparse-distillation-videodit)
13. [Real-Time Diffusion (MirageLSD)](#13-real-time-diffusion-miragelsd)
14. [Summary Matrix](#14-summary-matrix)
15. [Recommendations for RealRestore](#15-recommendations-for-realrestore)

---

## 1. Step Distillation Methods

### Progressive Distillation (Google Research)

- **Source**: [@hojonathanho](https://x.com/hojonathanho/status/1577712636333944832) -- Jonathan Ho (Google), [@fly51fly](https://x.com/fly51fly/status/1488990651102285824)
- **Technique**: Teacher model generates targets at 2N steps; student learns to match in N steps. Repeat halving.
- **Speed improvement**: 18x faster for Imagen Video (50 steps reduced to 8 steps per sub-model)
- **Quality impact**: "Without noticeable loss in perceptual quality" -- per [@chenlin_meng](https://x.com/chenlin_meng/status/1579384418606944257) (Stanford/Google)
- **Quality metrics**: Perceptual quality maintained; FID comparable to teacher at 8 steps
- **Restoration applicability**: HIGH. Directly applicable to any diffusion-based restoration pipeline. The step reduction from 28 to 8 is well within the quality-preserving range demonstrated.

### Multi-Student Diffusion Distillation (NVIDIA)

- **Source**: [@jonLorraine9](https://x.com/jonLorraine9/status/1864407358056571054) -- Jonathan Lorraine (NVIDIA Toronto AI)
- **Technique**: Distill one teacher into multiple specialized single-step student generators. Each student handles a subset of conditioning data.
- **Speed improvement**: Single-step inference; smaller model architectures possible
- **Quality impact**: SOTA FID scores among single-step generators. Quality *improves* over single-student distillation due to specialization.
- **Quality metrics**: State-of-the-art FID on ImageNet benchmarks for 1-step generation
- **Restoration applicability**: MEDIUM. Could partition restoration tasks by degradation type (blur vs noise vs compression), with a specialized single-step student per degradation.

### rCM -- Rectified Consistency Model (NVIDIA)

- **Source**: [@zkwthu](https://x.com/zkwthu/status/1976469231261958403) -- Kaiwen Zheng (Tsinghua)
- **Technique**: Forward-reverse divergence joint distillation. Supplements trajectory-level self-consistency (forward divergence) with score distillation (reverse divergence regularization).
- **Speed improvement**: 15x-50x acceleration; 2-4 step generation for 10B+ parameter video models
- **Quality impact**: Matches SOTA DMD2 on quality metrics while *improving* diversity (mitigates mode collapse)
- **Quality metrics**: High fidelity in 1-4 steps on Cosmos-Predict2 and Wan2.1 (up to 14B params)
- **Restoration applicability**: HIGH. The quality + diversity preservation at 2-4 steps makes this ideal for restoration where both fidelity and natural texture diversity matter.

### SSD-1B / Segmind-Vega (Hugging Face + Segmind)

- **Source**: [@_akhaliq](https://x.com/_akhaliq/status/1744190813423157333)
- **Technique**: Progressive knowledge distillation of SDXL using layer-level loss. Removes residual networks and transformer blocks from U-Net.
- **Speed improvement**: 60% faster inference (1.3B params vs 3.5B for SDXL); 0.74B variant even faster
- **Quality impact**: Preserves generative quality through careful layer-level distillation targeting
- **Quality metrics**: Competitive FID and CLIP scores to full SDXL
- **Restoration applicability**: MEDIUM. Model compression reduces latency but the layer removal may impact fine-grained detail reconstruction needed for restoration.

---

## 2. Consistency Models

### Latent Consistency Models (LCM)

- **Source**: [@multimodalart](https://x.com/multimodalart/status/1722677325693870202) -- apolinario, [@iScienceLuvr](https://x.com/iScienceLuvr/status/1711212075094245387) -- Tanishq Abraham
- **Technique**: Consistency model distillation applied to latent diffusion. Views guided reverse diffusion as solving an augmented PF-ODE, then predicts solution directly.
- **Speed improvement**: 2-4 step inference (from 20-50 steps); LCM-LoRA enables real-time SDXL
- **Quality impact**: Good quality at 4-8 steps. Quality degrades noticeably at 1-2 steps. "Running SDXL in real-time with LCM-LoRA" described as "mind-blowing" by [@viccpoes](https://x.com/viccpoes/status/1722734064145428925)
- **Quality metrics**: Competitive at 4 steps; FID within ~10-15% of full-step baselines
- **Restoration applicability**: HIGH. LCM-LoRA is a training-free adapter -- can be applied to any SD-based restoration model with minimal effort. The 4-step sweet spot preserves enough detail for restoration.

### Phased Consistency Model (PCM)

- **Source**: [@_akhaliq](https://x.com/_akhaliq/status/1795647841496387756)
- **Technique**: Addresses LCM quality issues at very low step counts by phasing the consistency training across different noise levels.
- **Speed improvement**: Similar to LCM (4-8 steps) but with better quality at extreme low steps
- **Quality impact**: Improved over standard LCM, especially for high-resolution text-conditioned generation
- **Restoration applicability**: HIGH. The phased approach better preserves detail at low step counts, which is critical for restoration fidelity.

### Consistency Models Made Easy (ECT) -- ICLR 2025

- **Source**: Academic paper linked from multiple X discussions
- **Technique**: Simplified consistency model training recipe without adversarial components
- **Speed improvement**: 1-2 step generation
- **Quality metrics**: CIFAR-10: superior 1-step and 2-step FID vs prior art. ImageNet 64x64: best-in-class few-step quality.
- **Restoration applicability**: MEDIUM. Promising for restoration but primarily validated on generation tasks.

### Fast Image Super-Resolution via Consistency Rectified Flow (ICCV 2025)

- **Source**: Academic paper, discussed across X research threads
- **Technique**: Distills multi-step SR diffusion into single-step inference using consistency rectified flow
- **Speed improvement**: Single-step super-resolution (from 20+ steps)
- **Quality impact**: HR regularization ensures convergence to high-resolution target
- **Restoration applicability**: VERY HIGH. Directly targets super-resolution restoration. Single-step SR with quality preservation is exactly what RealRestore needs.

### InterLCM for Face Restoration (ICLR 2025)

- **Source**: Academic paper from ICLR 2025
- **Technique**: Treats low-quality image as intermediate state of LCM. Starts denoising from degraded image rather than pure noise.
- **Speed improvement**: Few-step restoration (leverages LCM speed)
- **Quality impact**: Better semantic consistency in face restoration vs standard approaches
- **Restoration applicability**: VERY HIGH. This is a restoration-specific LCM application. The "degraded image as intermediate state" framing is directly relevant to RealRestore's pipeline.

---

## 3. Adversarial Distillation

### Adversarial Diffusion Distillation (ADD) -- Stability AI

- **Source**: [@robrombach](https://x.com/robrombach/status/1770005063827669186) -- Robin Rombach (Stability AI); Stability AI announcement
- **Technique**: Uses two losses: adversarial loss (GAN discriminator on generated samples) + distillation loss (match teacher diffusion model). No need for expensive multi-step diffusion sampling during training.
- **Speed improvement**: 1-step generation (SDXL Turbo); 207ms for 512x512 on A100
- **Quality impact**: In blind quality tests, SDXL Turbo 1-step beats LCM-XL 4-step; SDXL Turbo 4-step beats SDXL 50-step
- **Quality metrics**: Competitive FID; superior perceptual quality in human evaluation vs LCM
- **Restoration applicability**: HIGH. The adversarial component encourages realistic textures, which is critical for restoration. However, GAN-based artifacts may introduce hallucinated details.

### Adversarial Post-Training (APT / Seaweed-APT)

- **Source**: [@arXivGPT](https://x.com/arXivGPT/status/1879952860793749617)
- **Technique**: Post-training approach that forgoes teacher distillation entirely, performing direct adversarial training on real data after diffusion pre-training.
- **Speed improvement**: Single forward pass for 1280x720 24fps 2-second video; 1024px images in 1 step
- **Quality impact**: Enhanced details and realism vs teacher model. Only method to achieve favorable visual fidelity criterion. Weaker on text alignment but stronger on visual quality.
- **Quality metrics**: Mid-tier text alignment but top-tier visual fidelity
- **Restoration applicability**: HIGH. The emphasis on visual fidelity over text alignment is perfect for restoration, where text prompts are secondary and image quality is paramount.

### f-Divergence Distribution Matching (NVIDIA)

- **Source**: [@iScienceLuvr](https://x.com/iScienceLuvr/status/1893858785426587730) -- Tanishq Abraham
- **Technique**: Generalizes variational score distillation to the family of f-divergences. Jensen-Shannon divergence variant (f-distill) achieves SOTA.
- **Speed improvement**: Single-step generation
- **Quality metrics**: Current SOTA for one-step generation on ImageNet64 and MS-COCO
- **Restoration applicability**: MEDIUM. Strong theoretical foundation but primarily validated on generation.

### Uni-Instruct Framework

- **Source**: [@jiqizhixin](https://x.com/jiqizhixin/status/1984182299274162658)
- **Technique**: Unifies 10+ one-step distillation methods (Diff-Instruct, DMD, SIM, SiD, f-distill) under a single mathematical foundation using diffusion expansion theory.
- **Speed improvement**: Single-step generation across all unified methods
- **Restoration applicability**: MEDIUM. Theoretical unification may enable picking the optimal distillation variant for restoration specifically.

---

## 4. Solver/Sampler Optimization

### DPM-Solver++ (Tsinghua University)

- **Source**: [@diffuserslib](https://x.com/diffuserslib/status/1589978869331013632), [@Birchlabs](https://x.com/Birchlabs/status/1589038086633455616), [@ChengLu05671218](https://twitter.com/ChengLu05671218/status/1589933755023917056)
- **Technique**: High-order ODE solver specifically designed for diffusion probability flow. Uses multi-step predictor-corrector formulation.
- **Speed improvement**: Coherent images in 5 model calls (DPM-Solver++(2M)); "amazing quality" at 15-20 steps
- **Quality impact**: 4.70 FID in 10 evaluations, 2.87 FID in 20 evaluations on CIFAR10 -- 4-16x speedup vs prior training-free samplers
- **Quality metrics**: FID 4.70 (10 steps), 2.87 (20 steps) on CIFAR10
- **Restoration applicability**: VERY HIGH. Training-free, drop-in sampler replacement. Directly reduces step count for any diffusion restoration model. Already widely supported in diffusers.

### DPM-Solver-v3 (Tsinghua University)

- **Source**: Academic paper, follow-up to DPM-Solver++
- **Technique**: Incorporates empirical model statistics for even better low-step performance
- **Speed improvement**: 15-30% speedup over DPM-Solver++ at 5-10 NFE
- **Quality metrics**: FID 12.21 (5 NFE), 2.51 (10 NFE) on unconditional CIFAR10
- **Restoration applicability**: VERY HIGH. Same drop-in benefits as DPM-Solver++ with even better quality at ultra-low step counts.

### Sampler Comparison Summary (from X community testing)

- **Source**: [@Bookyakuno](https://x.com/Bookyakuno/status/1595601854213238784), [@br_d](https://x.com/br_d/status/1705944095733207544), [@forasteran](https://x.com/forasteran/status/1691078005932863488)
- **Findings from community**:
  - DPM++ 2M Karras: Good detail, suitable for 15-20 steps
  - DPM++ 3M SDE Karras: Recommended for 50-60 steps, best detail quality
  - DPM++ 3M SDE Exponential: Comparable to Karras variant
  - Restart sampler: "Most detailed results" but 1.5-2x slower
  - Euler a / DPM++ 2S a Karras: Best background instruction following
  - DDIM: Requires ~50 steps for comparable quality to DPM++ at 20
- **Restoration applicability**: HIGH. DPM++ 2M Karras at 15-20 steps is the practical sweet spot for restoration -- fast and detailed.

---

## 5. Feature Caching (DeepCache)

### DeepCache (CVPR 2024)

- **Source**: [@camenduru](https://x.com/camenduru/status/1732149154783064270), GitHub/academic sources
- **Technique**: Exploits temporal redundancy in sequential denoising. Caches high-level U-Net features and reuses them across adjacent steps, updating only low-level features cheaply.
- **Speed improvement**:
  - Stable Diffusion v1.5: 2.3x speedup
  - LDM-4-G (ImageNet): 4.1x speedup
  - LDM-4-G 250 DDIM steps: up to 7.0-10.5x speedup
- **Quality impact**:
  - SD v1.5: Only 0.05 decline in CLIP Score at 2.3x speed
  - LDM-4-G: FID goes from 3.37 to 4.41 at 7.0x speedup (moderate but acceptable)
  - "Almost lossless" at 2-3x acceleration
- **Quality metrics**: CLIP Score delta -0.05 (2.3x); FID delta +1.04 (7.0x)
- **Restoration applicability**: HIGH. Training-free, works with existing U-Net models. The feature caching leverages U-Net architecture properties that are present in most restoration diffusion models. At 2-3x speedup the quality loss is negligible.

### Important caveat for restoration
DeepCache assumes temporal redundancy between adjacent denoising steps. In restoration pipelines where the model starts from the degraded image (not pure noise), the early steps may carry more unique information. Caching ratios may need tuning -- aggressive caching (10x) will likely degrade restoration quality more than generation quality.

---

## 6. Token Merging (ToMe / ToMA)

### ToMe for Stable Diffusion (Georgia Tech, CVPR 2023 Workshop)

- **Source**: [dbolya/tomesd GitHub](https://github.com/dbolya/tomesd), HuggingFace diffusers documentation
- **Technique**: Merges redundant tokens/patches in transformer forward pass. Reduces token count by up to 60%.
- **Speed improvement**: Up to 2x faster, 5.6x less memory
- **Quality impact**: "No significant decrease in quality of generated samples" -- but content within images can change, especially with naive application
- **Restoration applicability**: MEDIUM-LOW. Merging tokens can alter spatial detail, which is problematic for restoration tasks that need pixel-level fidelity. Needs careful tuning of merge ratio.

### ToMA: Token Merge with Attention (ICML 2025)

- **Source**: Academic paper, discussed in X research threads
- **Technique**: Reformulates token merging as submodular optimization. Implements merge/unmerge as attention-like linear transformations via GPU-friendly matrix operations. Exploits latent locality and sequential redundancy.
- **Speed improvement**:
  - SDXL-base: 24% latency reduction
  - Flux.1-dev: 23% latency reduction
- **Quality impact**: DINO score delta < 0.07 (negligible degradation)
- **Quality metrics**: DINO delta < 0.07
- **Restoration applicability**: MEDIUM. Better than naive ToMe because GPU-aligned operations avoid overhead, but the 24% speedup is modest. The quality preservation (DINO < 0.07) is promising for restoration, but needs validation on restoration-specific metrics (PSNR/SSIM).

### NegToMe: Negative Token Merging

- **Source**: [NegToMe project page](https://negtome.github.io/)
- **Technique**: Adversarial guidance via token space -- pushes apart matching semantic features between reference and output during reverse diffusion.
- **Speed impact**: <4% additional inference time
- **Quality impact**: Improves output diversity and enables quality enhancement, style control, object feature interpolation
- **Restoration applicability**: LOW. Designed for generation diversity, not restoration fidelity.

---

## 7. CFG Distillation / Guidance Optimization

### Guidance Distillation (single forward pass)

- **Source**: [FLUX.1-dev discussion](https://huggingface.co/black-forest-labs/FLUX.1-dev/discussions/17), academic papers
- **Technique**: Distills the two forward passes of classifier-free guidance (conditional + unconditional) into a single forward pass.
- **Speed improvement**: ~2x per-step speedup (halves computation per step)
- **Quality impact**: Quality maintained -- the distilled model produces outputs indistinguishable from full CFG
- **Restoration applicability**: VERY HIGH. Pure speed gain with no quality loss. If the restoration model uses CFG, this is a free 2x speedup. FLUX.1-dev already ships with guidance distillation.

### Adapter Guidance Distillation (AGD) -- 2025

- **Source**: [arxiv 2503.07274](https://arxiv.org/abs/2503.07274)
- **Technique**: Lightweight adapter (~1% of base model parameters) that approximates CFG output in a single forward pass.
- **Speed improvement**: 2x (halves NFE)
- **Quality impact**: "Comparable or superior FID to CFG across multiple architectures"
- **Quality metrics**: FID comparable or better than standard CFG with half the NFEs
- **Restoration applicability**: VERY HIGH. 1% parameter overhead, 2x speed, no quality loss. Ideal for restoration models using CFG.

### Residual Classifier-Free Guidance (R-CFG) -- StreamDiffusion

- **Source**: [StreamDiffusion paper](https://arxiv.org/abs/2312.12491)
- **Technique**: Replaces full unconditional denoising with a residual approximation, eliminating redundant computation.
- **Speed improvement**: Up to 2.05x over conventional CFG
- **Quality impact**: Minimal degradation in interactive/streaming scenarios
- **Restoration applicability**: HIGH. Applicable to any CFG-based pipeline.

### Independent Condition Guidance (ICG) -- Disney Research, 2025

- **Source**: [Disney Research Studios](https://studios.disneyresearch.com/2025/04/23/no-training-no-problem-rethinking-diffusion-guidance-for-diffusion-models/)
- **Technique**: Provides CFG benefits without special unconditional training. Queries pre-trained conditional model with independent/random context.
- **Speed improvement**: Eliminates unconditional training entirely; inference speed same as single-pass
- **Quality impact**: Matches performance of standard CFG
- **Restoration applicability**: MEDIUM. Training simplification benefit; inference speed benefit only if the model currently uses two-pass CFG.

---

## 8. Compiler/Runtime Optimization

### torch.compile for Diffusion Models

- **Source**: [@cHHillee](https://x.com/cHHillee/status/1845180572089581852) -- Horace He (PyTorch team)
- **Technique**: PyTorch 2.0+ JIT compilation with graph capture, operator fusion, and kernel optimization.
- **Speed improvement**: Significant (model-dependent); competitive with TensorRT for many architectures
- **Quality impact**: Zero quality loss -- mathematically identical outputs
- **Restoration applicability**: VERY HIGH. Zero-effort, zero-quality-loss speedup. Should be first optimization applied to any PyTorch restoration pipeline.

### Torch-TensorRT

- **Source**: [NVIDIA Technical Blog](https://developer.nvidia.com/blog/double-pytorch-inference-speed-for-diffusion-models-using-torch-tensorrt/)
- **Technique**: Compiles PyTorch models to TensorRT optimized engines. Supports FP16 and FP8 precision.
- **Speed improvement**:
  - FLUX.1-dev (12B params): 1.5x with FP16, 2.4x with FP8
  - General: up to 5x over eager execution
- **Quality impact**: FP16 -- zero quality loss. FP8 -- minimal quality impact on H100 (needs validation per model)
- **Quality metrics**: Mathematically identical (FP16); negligible drift (FP8)
- **Restoration applicability**: HIGH for NVIDIA GPUs. Not applicable to MPS/Apple Silicon (RealRestore's primary target). Worth supporting as a fast-path for CUDA users.

### FP8 Quantization (H100/H200)

- **Source**: Various NVIDIA blog posts and X discussions
- **Technique**: Hardware-accelerated 8-bit floating point inference on Hopper GPUs
- **Speed improvement**: 1.5-2x over FP16 after exhausting other optimizations
- **Quality impact**: Minimal quality impact when properly calibrated
- **Restoration applicability**: LOW for RealRestore (targets Apple Silicon MPS), HIGH for server deployments.

---

## 9. Representation Alignment (REPA)

### REPA: Training Diffusion Transformers Is Easier Than You Think

- **Source**: [@sainingxie](https://x.com/sainingxie/status/1845510163152687242) -- Saining Xie, [@abursuc](https://x.com/abursuc/status/1845545445486993665) -- Andrei Bursuc, [@alec_helbling](https://x.com/alec_helbling/status/2004554038784966837)
- **Technique**: Aligns projections of noisy input hidden states with clean image representations from pre-trained visual encoders (DINOv2, MoCov3, CLIP, MAE).
- **Speed improvement**: 17.5x faster *training* convergence for SiT
- **Quality impact**: *Improved* generation quality -- FID 1.42 with CFG on ImageNet (SOTA)
- **Quality metrics**: FID 1.42 (ImageNet, with CFG) -- state of the art
- **Restoration applicability**: MEDIUM. Primarily a training speedup technique. Could accelerate restoration model training dramatically, but does not directly speed up inference. However, REPA-trained models may converge to better quality, enabling fewer inference steps.

### SiT: Scalable Interpolant Transformers

- **Source**: [@sainingxie](https://x.com/sainingxie/status/1747863734884745431)
- **Technique**: Flow-based generative model using interpolant framework. Same backbone as DiT but with better design choices.
- **Speed improvement**: Faster training convergence; FID 2.06 on ImageNet
- **Quality metrics**: FID 2.06 (ImageNet 256x256)
- **Restoration applicability**: MEDIUM. Architecture improvement for DiT-based models. If restoration moves to DiT architectures, SiT's interpolant framework would be beneficial.

---

## 10. Pipeline-Level Streaming (StreamDiffusion)

### StreamDiffusion (ICCV 2025)

- **Source**: [@viborc](https://x.com/viborc/status/1976003862306996506), academic paper
- **Technique**: Pipeline-level batching of denoising steps (Stream Batch) + Residual CFG + Stochastic Similarity Filtering. Transforms sequential denoising into batched parallel processing.
- **Speed improvement**: Up to 91.07 FPS on RTX 4090; 59.56x over standard Diffusers pipeline
- **Quality impact**: Maintains interactive-quality output; R-CFG introduces minor quality trade-off vs full CFG
- **Restoration applicability**: MEDIUM-HIGH. The batching technique is powerful for video/stream restoration. For single-image restoration, the pipeline overhead may not pay off, but for batch processing of multiple images it would be significant.

### StreamDiffusionV2

- **Source**: [@viborc](https://x.com/viborc/status/1976003862306996506), academic paper
- **Technique**: Extends to video with temporal stability (StreamVAE + rolling KV with sink). Motion-aware noise control.
- **Speed improvement**: 58.28 FPS (14B model), 64.52 FPS (1.3B model) on 4x H100; renders first frame in 0.5s
- **Quality impact**: Sustains 31.62 FPS even with increased denoising steps for quality
- **Restoration applicability**: HIGH for video restoration pipelines. The temporal stability features directly address a key challenge in video restoration.

---

## 11. Restoration-Specific Approaches

### InDI: Inversion by Direct Iteration (Google Research)

- **Source**: [@TmlrPub](https://x.com/TmlrPub/status/1673572023337992193), Delbracio & Milanfar (Google)
- **Technique**: Instead of denoising from pure noise conditioned on degraded input, directly iteratively restores the input image in small steps. Avoids "regression to the mean" effect.
- **Speed improvement**: Comparable quality in "much less number of steps" vs conditional denoising diffusion
- **Quality impact**: More realistic and detailed images than regression-based methods; avoids over-smoothing
- **Tasks validated**: Motion deblurring, out-of-focus deblurring, super-resolution, compression artifact removal, denoising
- **Restoration applicability**: VERY HIGH. This is specifically designed for restoration. The direct iteration approach (start from degraded image, not noise) aligns perfectly with RealRestore's use case.

### RealRestorer

- **Source**: [@HuggingPapers](https://x.com/HuggingPapers/status/2037504984934240309)
- **Technique**: Unified framework for generalizable real-world image restoration using large-scale image editing models. Handles 9 degradation types.
- **Quality impact**: "Ranks first among open-source methods" for real-world restoration
- **Tasks**: Blur, noise, rain, reflection, and 5 other degradation types
- **Restoration applicability**: VERY HIGH. Direct competitor/reference for RealRestore. Study their degradation-handling approach.

### Zero-Shot Image Restoration with Consistency Models (CVPR 2025)

- **Source**: Academic paper (CVPR 2025)
- **Technique**: Few-step guidance of consistency models for zero-shot restoration. No task-specific training needed.
- **Speed improvement**: Few-step (2-4) restoration
- **Quality impact**: Competitive with task-specific methods
- **Restoration applicability**: VERY HIGH. Zero-shot means no retraining for new degradation types. Few-step means fast inference.

### Consistency Trajectory Matching for One-Step Super-Resolution (ICCV 2025)

- **Source**: Academic paper (ICCV 2025)
- **Technique**: Single-step generative super-resolution via consistency trajectory matching
- **Speed improvement**: One-step SR
- **Restoration applicability**: VERY HIGH. Single-step SR is the ultimate speed target for restoration.

### RestoreGrad

- **Source**: [@ArxivSound](https://x.com/ArxivSound/status/1919242046109290936)
- **Technique**: Signal restoration using conditional denoising diffusion with jointly learned prior
- **Restoration applicability**: HIGH. Demonstrates diffusion restoration principles applicable across modalities.

---

## 12. Sparse Distillation (Video/DiT)

### FastWan: Sparse Distillation

- **Source**: [@gm8xx8](https://x.com/gm8xx8/status/1952533434695319557), Hao AI Lab (UCSD)
- **Technique**: First method to jointly train sparse attention and denoising step distillation in a unified framework. Learns data-dependent sparsity patterns while compressing 50 steps to 3 steps.
- **Speed improvement**: 5-second 480P video in 5 seconds (denoising time: 1 second on H200, 2.8s on RTX 4090)
- **Quality impact**: "Massive speedups without quality loss" -- sparse attention patterns are learned jointly with step compression
- **Restoration applicability**: MEDIUM-HIGH. Sparse attention + step distillation is a powerful combination. The data-dependent sparsity learning could be adapted to learn which attention patterns matter most for restoration.

### DC-VideoGen: Deep Compression Video Autoencoder

- **Source**: [@hancai_hm](https://x.com/hancai_hm/status/1973072875096592415) -- Han Cai
- **Technique**: Deep compression autoencoder (32x/64x spatial, 4x temporal) + AE-Adapt-V adaptation strategy for transferring pre-trained models to new latent space.
- **Speed improvement**: 14.8x faster inference; supports up to 4K resolution on single H100
- **Quality impact**: "Preserving or even improving video quality"
- **Restoration applicability**: MEDIUM. The deep compression autoencoder concept could be applied to image restoration -- operating in a more compressed latent space speeds up diffusion while maintaining quality through the autoencoder.

### Presto!: Inference Acceleration for Diffusion Transformers (UCSD + Adobe)

- **Source**: [@Marktechpost](https://x.com/Marktechpost/status/1844968778268017107)
- **Technique**: Combines score-based distribution matching distillation (DMD) for step reduction with layer distillation for per-step cost reduction. First GAN-based distillation for EDM family.
- **Speed improvement**: Dual reduction -- fewer steps AND cheaper per-step compute
- **Restoration applicability**: MEDIUM. Demonstrated on text-to-music but the dual-reduction principle (step + layer distillation) is architecture-agnostic and applicable to image restoration DiTs.

---

## 13. Real-Time Diffusion (MirageLSD)

### MirageLSD: Live-Stream Diffusion

- **Source**: [@karpathy](https://x.com/karpathy/status/1945979830740435186) -- Andrej Karpathy
- **Technique**: Real-time video stream transformation using diffusion models. Processes camera/video input with <40ms latency.
- **Speed improvement**: <40ms latency -- true real-time (>25 FPS)
- **Quality impact**: "Real-time magic" -- unlike simple video filters, understands scene content and performs intelligent transformation
- **Restoration applicability**: MEDIUM. The real-time latency target is aspirational for restoration. The architecture choices enabling <40ms diffusion could inform real-time restoration design, though restoration requires higher fidelity than stylization.

### Speedrunning ImageNet Diffusion (SR-DiT)

- **Source**: [@SwayStar123](https://x.com/SwayStar123/status/2000854683909304322), [@sedielem](https://x.com/sedielem/status/2000995375851704438) -- Sander Dieleman
- **Technique**: Framework integrating token routing, architectural improvements, and training modifications on top of REPA. Combines "a bunch of recent ideas that speed up training."
- **Speed improvement**: Fastest training time for reasonable-quality ImageNet generation
- **Quality impact**: Maintained generation quality with dramatically reduced training compute
- **Restoration applicability**: LOW-MEDIUM. Training speedup focus, not inference.

---

## 14. Summary Matrix

| Technique | Speed Gain | Quality Impact | Training-Free? | Restoration Fit | Priority |
|-----------|-----------|---------------|---------------|----------------|----------|
| **DPM-Solver++/v3** | 3-5x (steps: 50->10-15) | Preserved (FID ~2.5-4.7) | Yes | Very High | P0 |
| **DeepCache** | 2-3x (safe), 7-10x (aggressive) | Preserved at 2-3x (CLIP -0.05) | Yes | High | P0 |
| **CFG/Guidance Distillation** | 2x (per step) | Preserved or improved | One-time distill | Very High | P0 |
| **torch.compile** | 1.3-2x | Zero loss | Yes | Very High | P0 |
| **LCM / LCM-LoRA** | 5-12x (steps: 50->4) | Good at 4 steps | LoRA adapter | High | P1 |
| **Progressive Distillation** | 6-18x | Preserved at 8 steps | Requires training | High | P1 |
| **rCM** | 15-50x (steps: 50->2-4) | Preserved + diversity | Requires training | High | P1 |
| **Adversarial Distillation (ADD)** | 25-50x (1 step) | Competitive/superior | Requires training | High | P1 |
| **InDI (Direct Iteration)** | Moderate (fewer steps) | Improved (avoids mean regression) | Architecture change | Very High | P1 |
| **Consistency Rectified Flow SR** | ~20x (1 step SR) | Preserved | Requires training | Very High | P1 |
| **ToMA** | 1.2-1.3x | Preserved (DINO <0.07) | Yes | Medium | P2 |
| **Sparse Distillation** | 15-17x | Preserved | Requires training | Medium-High | P2 |
| **StreamDiffusion pipeline** | 10-60x (batched) | Good for streaming | Architecture change | Medium-High | P2 |
| **REPA** | Training only (17x) | Improved (FID 1.42) | Training only | Medium | P3 |
| **FP8 Quantization** | 1.5-2x | Minimal | GPU-specific | Low (MPS) | P3 |

---

## 15. Recommendations for RealRestore

### Immediate (No Training Required) -- Stack These Together

1. **DPM-Solver++ or UniPC sampler**: Replace default scheduler. Drop from 28 steps to 10-15 with comparable quality. Estimated 2-3x speedup.
2. **DeepCache**: Enable feature caching at conservative ratio (2-3x target). Minimal quality loss. Stacks with solver optimization.
3. **torch.compile**: Compile the denoising U-Net. Free 1.3-2x speedup with zero quality impact.
4. **CFG guidance distillation**: If using CFG, distill to single pass or use R-CFG. 2x per-step speedup.

**Combined estimated speedup: 8-24x with near-zero quality loss.**

### Short-Term (Minimal Training)

5. **LCM-LoRA**: Fine-tune or apply LCM-LoRA adapter for 4-step inference. Well-validated, minimal training cost.
6. **InterLCM approach**: Treat degraded image as LCM intermediate state. Restoration-specific optimization.

### Medium-Term (Research Investment)

7. **Consistency Rectified Flow for SR**: Train single-step super-resolution model using consistency rectified flow. This is the state of the art for fast SR.
8. **InDI (Direct Iteration)**: Investigate replacing denoising-from-noise with direct iterative restoration. Better quality AND fewer steps for restoration tasks.
9. **Adversarial post-training**: Fine-tune with adversarial objective for 1-4 step generation with enhanced realism.

### Architecture Decisions

- For generation-based restoration (noise-to-image): Prioritize distillation methods (LCM, progressive distillation, rCM)
- For iterative restoration (degraded-to-clean): Prioritize InDI approach + solver optimization
- For video restoration: StreamDiffusionV2 pipeline + sparse distillation
- For batch processing: StreamDiffusion pipeline batching

### Key Insight from Research

The most impactful finding is that **stacking training-free optimizations** (better solver + caching + compilation + guidance optimization) can achieve 8-24x speedup before any model retraining. This should be the first step. Distillation and consistency model training should follow only after maximizing training-free gains.

---

## Sources

### X/Twitter Posts (Primary Sources)

- [Aryan V S -- Optimizing diffusion inference for production speeds](https://x.com/aryanvs_/status/2005264703959032318)
- [Han Cai -- DC-VideoGen 14.8x faster](https://x.com/hancai_hm/status/1973072875096592415)
- [Xun Huang -- Self-Forcing real-time video generation](https://x.com/xunhuang1995/status/1932107954574275059)
- [Jonathan Lorraine -- NVIDIA Multi-student Diffusion Distillation](https://x.com/jonLorraine9/status/1864407358056571054)
- [Kaiwen Zheng -- rCM rectified consistency model](https://x.com/zkwthu/status/1976469231261958403)
- [apolinario -- LCM-LoRA for SDXL](https://x.com/multimodalart/status/1722677325693870202)
- [Tanishq Abraham -- Latent Consistency Models](https://x.com/iScienceLuvr/status/1711212075094245387)
- [AK -- Phased Consistency Model](https://x.com/_akhaliq/status/1795647841496387756)
- [AK -- SSD-1B Segmind progressive distillation](https://x.com/_akhaliq/status/1744190813423157333)
- [diffusers -- DPM-Solver++ 15-20 steps](https://x.com/diffuserslib/status/1589978869331013632)
- [Birchlabs -- DPM-Solver++ 5 steps coherent](https://x.com/Birchlabs/status/1589038086633455616)
- [Jonathan Ho -- Progressive distillation 18x](https://x.com/hojonathanho/status/1577712636333944832)
- [Chenlin Meng -- 8 step distillation no perceptual loss](https://x.com/chenlin_meng/status/1579384418606944257)
- [Robin Rombach -- Adversarial Diffusion Distillation](https://x.com/robrombach/status/1770005063827669186)
- [Saining Xie -- REPA and SiT](https://x.com/sainingxie/status/1845510163152687242)
- [Andrei Bursuc -- REPA overview](https://x.com/abursuc/status/1845545445486993665)
- [Sander Dieleman -- progressive distillation discrete](https://x.com/sedielem/status/1851331608336384394)
- [Sander Dieleman -- rectified flow as contender](https://x.com/sedielem/status/1796900219268804706)
- [Sander Dieleman -- FreeFlow data-free distillation](https://x.com/sedielem/status/1993413445744836747)
- [Sander Dieleman -- speedrunning diffusion](https://x.com/sedielem/status/2000995375851704438)
- [Andrej Karpathy -- MirageLSD real-time diffusion](https://x.com/karpathy/status/1945979830740435186)
- [Horace He -- torch.compile for diffusion](https://x.com/cHHillee/status/1845180572089581852)
- [Marktechpost -- Presto! UCSD/Adobe inference acceleration](https://x.com/Marktechpost/status/1844968778268017107)
- [Haotian Sun -- EC-DIT MoE diffusion transformer](https://x.com/haotiansun014/status/1844503506918814135)
- [TMLR -- InDI alternative to denoising diffusion](https://x.com/TmlrPub/status/1673572023337992193)
- [Hugging Papers -- RealRestorer unified restoration](https://x.com/HuggingPapers/status/2037504984934240309)
- [gm8xx8 -- FastWan sparse distillation](https://x.com/gm8xx8/status/1952533434695319557)
- [Together AI -- CDLM consistency diffusion LM](https://x.com/togethercompute/status/2024541629194129677)
- [JIQIZHIXIN -- Uni-Instruct unifying distillation](https://x.com/jiqizhixin/status/1984182299274162658)
- [JIQIZHIXIN -- DiDi-Instruct speed+quality coexist](https://x.com/jiqizhixin/status/1984549725471691216)
- [Tanishq Abraham -- f-divergence one-step SOTA](https://x.com/iScienceLuvr/status/1893858785426587730)
- [arXivGPT -- Adversarial Post-Training](https://x.com/arXivGPT/status/1879952860793749617)
- [Vibor Cipan -- StreamDiffusionV2 real-time](https://x.com/viborc/status/1976003862306996506)
- [Alec Helbling -- diffusion speedrunning literature](https://x.com/alec_helbling/status/2004554038784966837)
- [Jiatao Gu -- BOOT data-free distillation](https://x.com/thoma_gu/status/1669796676674920454)
- [Runway x NVIDIA -- video diffusion optimization](https://x.com/runwayml/status/1995496173755318751)
- [LMSYS -- SGLang Diffusion 5.9x faster serving](https://x.com/lmsysorg/status/1986886665496084877)

### Academic Papers and Projects (Referenced in Posts)

- DPM-Solver/DPM-Solver++/v3 (Tsinghua University)
- DeepCache (CVPR 2024)
- LCM / Phased Consistency Model
- Adversarial Diffusion Distillation (Stability AI)
- Progressive Distillation (Google Research)
- REPA (Saining Xie et al.)
- SiT: Scalable Interpolant Transformers
- InDI: Inversion by Direct Iteration (Google Research)
- StreamDiffusion / StreamDiffusionV2
- FastWan / Sparse Distillation (Hao AI Lab, UCSD)
- DC-VideoGen (Han Cai et al.)
- rCM: Rectified Consistency Model (NVIDIA / Tsinghua)
- Multi-Student Diffusion Distillation (NVIDIA)
- ToMA: Token Merge with Attention (ICML 2025)
- RealRestorer (Hugging Face Daily Papers)
- Fast Image SR via Consistency Rectified Flow (ICCV 2025)
- InterLCM for Face Restoration (ICLR 2025)
- Zero-Shot Restoration with Consistency Models (CVPR 2025)
- Consistency Trajectory Matching for One-Step SR (ICCV 2025)
- Adapter Guidance Distillation (2025)
- Presto! (UCSD + Adobe)
