# Innovative Diffusion Inference Speedup Techniques (2025-2026)

> **Research date**: 2026-03-29
> **Purpose**: Identify cutting-edge techniques for faster RealRestorer inference beyond standard optimizations (scheduler swaps, quantization, attention slicing)
> **Target hardware**: Apple Silicon M-series (MPS backend), secondary CUDA support

---

## Table of Contents

1. [Distillation for Fewer Steps](#1-distillation-for-fewer-steps)
2. [Token Merging / Token Pruning](#2-token-merging--token-pruning)
3. [DeepCache & Feature Caching](#3-deepcache--feature-caching)
4. [Parallel Denoising](#4-parallel-denoising)
5. [Speculative Decoding for Diffusion](#5-speculative-decoding-for-diffusion)
6. [Wavelet Diffusion](#6-wavelet-diffusion)
7. [Dynamic Resolution Denoising](#7-dynamic-resolution-denoising)
8. [Progressive Distillation](#8-progressive-distillation)
9. [Classifier-Free Guidance Distillation](#9-classifier-free-guidance-distillation)
10. [Flash Diffusion](#10-flash-diffusion)
11. [PAG (Perturbed Attention Guidance)](#11-pag-perturbed-attention-guidance)
12. [Additional Cutting-Edge Techniques](#12-additional-cutting-edge-techniques)
13. [Comprehensive Surveys & Repos](#13-comprehensive-surveys--repos)
14. [Applicability Matrix for RealRestorer](#14-applicability-matrix-for-realrestorer)
15. [Recommended Implementation Roadmap](#15-recommended-implementation-roadmap)

---

## 1. Distillation for Fewer Steps

### 1a. Latent Consistency Models (LCM)

**What**: LCMs predict the solution of the guided reverse diffusion PF-ODE directly in latent space, enabling 2-4 step inference matching 25-50 step DDIM quality.

**Performance**: 10-100x faster than standard diffusion sampling. 768x768 LCM distilled from SD requires only 32 A100 GPU-hours.

**Key innovation**: Consistency distillation augmented with latent-space ODE solving. TLCM (Training-efficient LCM) further reduces cost.

**Links**:
- Paper: [arXiv:2310.04378](https://arxiv.org/abs/2310.04378)
- Code: [github.com/luosiallen/latent-consistency-model](https://github.com/luosiallen/latent-consistency-model)
- Diffusers integration: [HF docs](https://huggingface.co/docs/diffusers/en/using-diffusers/inference_with_lcm)
- LCM distillation training: [HF training docs](https://huggingface.co/docs/diffusers/training/lcm_distill)

**RealRestorer applicability**: **HIGH** -- LCM-LoRA can be applied to any diffusion pipeline including custom U-Net architectures. The LCMScheduler is already in diffusers. Key challenge: RealRestorer uses a modified SDXL architecture with Qwen2.5-VL text encoder; LCM distillation would need to handle this custom connector. Could train an LCM-LoRA adapter specific to RealRestorer's restoration tasks. Expected speedup: **4-7x** (28 steps -> 4-6 steps).

### 1b. SDXL-Turbo / Adversarial Diffusion Distillation (ADD)

**What**: ADD combines score distillation from a teacher model with adversarial training to enable 1-4 step generation. SDXL-Turbo generates 512x512 images in 207ms on A100.

**Performance**: Single-step real-time generation. 50x step reduction.

**Key innovation**: GAN loss + score distillation = faithful single-step generation without mode collapse.

**Links**:
- Paper: [arXiv:2311.17042](https://arxiv.org/abs/2311.17042)
- Model: [huggingface.co/stabilityai/sdxl-turbo](https://huggingface.co/stabilityai/sdxl-turbo)
- Stability AI blog: [stability.ai/research/adversarial-diffusion-distillation](https://stability.ai/research/adversarial-diffusion-distillation)

**RealRestorer applicability**: **MEDIUM** -- ADD requires full distillation training with a discriminator network. This is not a plug-and-play technique. However, since RealRestorer is based on SDXL's U-Net, the ADD training recipe could theoretically be adapted. Major concern: restoration quality at 1-2 steps may not preserve fine detail fidelity needed for image restoration (vs generation). Research non-commercial license limits deployment.

### 1c. DMD2 (Distribution Matching Distillation v2)

**What**: NeurIPS 2024 Oral. Improves on DMD with two-time-scale update rules, GAN loss integration, and multi-step sampling support. Achieves FID 1.28 on ImageNet-64x64.

**Performance**: 1-4 step generation. FID 8.35 on zero-shot COCO 2014 with SDXL backbone.

**Key innovation**: Eliminates regression loss, no dataset construction needed, addresses training-inference mismatch via simulated generator samples during training.

**Links**:
- Paper: [arXiv:2405.14867](https://arxiv.org/abs/2405.14867)
- Code: [github.com/tianweiy/DMD2](https://github.com/tianweiy/DMD2)
- Models: [huggingface.co/tianweiy/DMD2](https://huggingface.co/tianweiy/DMD2)
- LoRA adapters: available on [CivitAI](https://civitai.com/models/1608870/dmd2-speed-lora-sdxl-pony-illustrious)

**RealRestorer applicability**: **HIGH** -- DMD2 is specifically designed for SDXL-class models. The available DMD2 LoRA adapters for SDXL could potentially be applied directly to RealRestorer's U-Net without full retraining. The multi-step support (vs 1-step-only) is ideal for restoration tasks where more steps = better detail. Expected speedup: **4-7x** with 4-step LoRA.

---

## 2. Token Merging / Token Pruning

### 2a. ToMe for Stable Diffusion (ToMeSD)

**What**: Merges redundant tokens/patches in the transformer forward pass, reducing attention computation. Originally from Bolya et al. (CVPR 2023 Workshop).

**Performance**: 2x speedup reported for SD. More pronounced gains at higher resolutions (1024x1024+).

**Links**:
- Original: [Token Merging for Fast Stable Diffusion (CVPR 2023W)](https://openaccess.thecvf.com/content/CVPR2023W/ECV/papers/Bolya_Token_Merging_for_Fast_Stable_Diffusion_CVPRW_2023_paper.pdf)
- Diffusers integration: [HF ToMe docs](https://huggingface.co/docs/diffusers/optimization/tome)

### 2b. ToMA: Token Merge with Attention (ICML 2025)

**What**: Evolved version of ToMe that uses attention maps to guide merging decisions rather than similarity-based bipartite matching.

**Performance**: 24% total generation time reduction on SDXL-base, 23% on Flux.1-dev, with negligible quality degradation. Outperforms ToMeSD and ToFu.

**Links**:
- Paper: [arXiv:2509.10918](https://arxiv.org/html/2509.10918v2)
- ICML 2025 poster: [icml.cc/virtual/2025/poster/46449](https://icml.cc/virtual/2025/poster/46449)

**Apple Silicon/MPS compatibility**: ToMe operates at the PyTorch tensor level -- it's pure Python + tensor ops. **Fully MPS compatible**. No custom CUDA kernels required. The token merging happens before attention computation, so it reduces the number of tokens flowing through MPS attention operations.

**RealRestorer applicability**: **HIGH** -- Training-free, plug-and-play. Can be applied to any transformer-based diffusion model. RealRestorer's U-Net has transformer blocks where ToMe/ToMA can directly reduce token counts. The `tomesd` library patches the pipeline in a single function call. Expected speedup: **1.2-1.5x** on top of other optimizations. Key caveat: for restoration tasks, aggressive token merging risks losing fine details in highly textured regions. Merge ratio should be conservative (0.3-0.5 vs 0.5-0.75 for generation).

---

## 3. DeepCache & Feature Caching

### 3a. DeepCache (CVPR 2024)

**What**: Training-free method that exploits temporal redundancy across denoising steps. Caches high-level U-Net features and only recomputes low-level features, since high-level features change slowly across adjacent steps.

**Performance**: 2.3x speedup on SD v1.5 with only 0.05 CLIP Score decline. 4.1x speedup on LDM-4-G (ImageNet) with 0.22 FID decrease.

**Key innovation**: Reuses deep encoder features, only recomputes shallow decoder features. Almost free -- no training, no architecture changes.

**Links**:
- Paper: [arXiv:2312.00858](https://arxiv.org/abs/2312.00858)
- Code: [github.com/horseee/DeepCache](https://github.com/horseee/DeepCache)
- Project page: [horseee.github.io/Diffusion_DeepCache](https://horseee.github.io/Diffusion_DeepCache/)
- Supports: SD, SDXL, Stable Video Diffusion, Inpainting, Img2Img

**RealRestorer applicability**: **VERY HIGH** -- Zero training, works on U-Net architectures, already supports SDXL. The temporal redundancy assumption holds for restoration where changes between steps are gradual. Can be combined with reduced steps for compound speedup (e.g., 8 steps + DeepCache = 3-4 effective forward passes). MPS compatible (pure PyTorch). Expected speedup: **1.5-2.3x** with negligible quality loss.

### 3b. ToCa: Token-wise Feature Caching (ICLR 2025)

**What**: Goes beyond DeepCache by recognizing that different tokens have different sensitivities to caching -- some tokens tolerate caching 10x better than others. Uses four scoring functions to select optimal tokens for caching per layer.

**Performance**: 2.36x on OpenSora, 1.93x on PixArt-alpha, nearly lossless.

**Key innovation**: Token-level granularity for caching decisions. No additional computation for scoring.

**Links**:
- Paper: [arXiv:2410.05317](https://arxiv.org/abs/2410.05317)
- Code: [github.com/Shenyi-Z/ToCa](https://github.com/Shenyi-Z/ToCa)
- ICLR 2025: [openreview.net/forum?id=yYZbZGo4ei](https://openreview.net/forum?id=yYZbZGo4ei)

**RealRestorer applicability**: **HIGH** -- ToCa is designed for DiT (Diffusion Transformer) architectures. RealRestorer uses a modified U-Net with transformer blocks, so ToCa's token-wise scoring could be adapted to the transformer layers within the U-Net. More sophisticated than DeepCache but requires more integration work.

### 3c. Block Caching / Learning-to-Cache

**What**: Observes that U-Net/DiT block outputs change slowly across timesteps and reuses stale block outputs, with schedules determined by measured block sensitivity.

**Performance**: Block Caching achieves 1.5-1.8x speedup. Learning-to-Cache (L2C) removes up to 93.68% of computation in cached steps (46.84% overall) with < 0.01 FID drop.

**Links**:
- Block Caching (CVPR 2024): [fwmb.github.io/blockcaching](https://fwmb.github.io/blockcaching/)
- Learning-to-Cache: [arXiv:2406.01733](https://arxiv.org/html/2406.01733v1)
- FastCache (2025): [arXiv:2505.20353](https://arxiv.org/abs/2505.20353)
- ReFrame (ICML 2025): [ubc-aamodt-group.github.io/reframe-layer-caching](https://ubc-aamodt-group.github.io/reframe-layer-caching/)

**RealRestorer applicability**: **HIGH** -- All caching methods are architecture-agnostic at the block level. FastCache (2025) adds spatial-aware token selection, which is ideal for restoration where spatially important tokens (edges, textures) need full computation while smooth regions can be cached.

---

## 4. Parallel Denoising

### 4a. ParaDiGMS (NeurIPS 2023 Spotlight)

**What**: Trades compute for wall-clock speed by denoising multiple steps in parallel using Picard iterations. Guesses future denoising states, runs them concurrently, and iteratively refines until convergence.

**Performance**: 2-4x speedup. 0.2s on 100-step DiffusionPolicy, 14.6s on 1000-step SD v2.1. Compatible with DDIM and DPMSolver.

**Links**:
- Paper: [arXiv:2305.16317](https://arxiv.org/abs/2305.16317)
- Code: [github.com/AndyShih12/paradigms](https://github.com/AndyShih12/paradigms)

**RealRestorer applicability**: **LOW for Apple Silicon** -- Parallel denoising requires multiple model copies running simultaneously. On unified memory Apple Silicon, you can't duplicate a 20GB model. Only viable on multi-GPU CUDA setups. For single-GPU/MPS, this technique provides no benefit.

### 4b. AsyncDiff (NeurIPS 2024)

**What**: Model parallelism -- splits the noise prediction network across multiple GPUs with asynchronous denoising. Exploits high similarity between consecutive step hidden states.

**Performance**: 2.7x speedup (4x GPUs) with negligible quality loss. 4x speedup with 0.38 CLIP Score drop.

**Links**:
- Paper: [NeurIPS 2024 proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/file/ad15848baa3932c0d2deabf0e11d1dcd-Paper-Conference.pdf)
- Project page: [czg1225.github.io/asyncdiff_page](https://czg1225.github.io/asyncdiff_page/)

### 4c. DistriFusion (CVPR 2024 Highlight)

**What**: Patch parallelism -- divides high-res images into sub-patches, each processed on a different GPU with stale activation reuse.

**Performance**: Near-linear scaling with number of GPUs. Integrated into NVIDIA TensorRT-LLM.

**Links**:
- Paper: [arXiv:2402.19481](https://arxiv.org/abs/2402.19481)
- Code: [github.com/mit-han-lab/distrifuser](https://github.com/mit-han-lab/distrifuser)

**RealRestorer applicability**: **LOW** -- All parallel denoising methods require multi-GPU setups. Not applicable to Apple Silicon single-chip. Could be relevant for a future cloud/API deployment mode.

---

## 5. Speculative Decoding for Diffusion

### 5a. Concept

Speculative decoding, borrowed from LLM acceleration, uses a small "draft" model to predict multiple future states in parallel, then a large "verifier" model accepts/rejects predictions in a single forward pass.

### 5b. Speculative Diffusion Decoding (SpecDiff) -- NAACL 2025

**What**: Uses discrete diffusion models to generate draft sequences, enabling parallel drafting and verification.

**Performance**: Up to 7.2x speedup over standard generation, 1.75x over existing speculative decoding.

**Link**: [aclanthology.org/2025.naacl-long.601](https://aclanthology.org/2025.naacl-long.601/)

### 5c. Self Speculative Decoding (SSD) -- 2025

**What**: Uses the diffusion model itself as both drafter and verifier -- no auxiliary model needed.

**Performance**: 3.46x speedup with identical output to stepwise decoding.

**Link**: [arXiv:2510.04147](https://arxiv.org/abs/2510.04147)

### 5d. DiffuSpec -- 2025

**What**: Training-free framework that uses pretrained diffusion LM to produce multi-token drafts in a single forward pass. Compatible with standard autoregressive verifiers.

**Performance**: Up to 3x wall-clock speedup.

**Link**: [arXiv:2510.02358](https://arxiv.org/abs/2510.02358)

**RealRestorer applicability**: **EXPERIMENTAL / RESEARCH** -- Speculative decoding for diffusion is currently focused on diffusion LLMs, not image diffusion models. The concept could theoretically apply: use a smaller/faster U-Net as drafter and RealRestorer as verifier. However, there are no existing implementations for image restoration diffusion. This is a potential novel research direction, not a near-term optimization.

---

## 6. Wavelet Diffusion

### 6a. WaveDiff (CVPR 2023)

**What**: Performs the diffusion process in wavelet domain instead of pixel domain. Decomposes images into low/high frequency wavelet subbands, denoises in wavelet space, then reconstructs via inverse wavelet transform.

**Performance**: 2.5x faster than DDGAN (fastest diffusion method at time of publication). Approaches real-time StyleGAN speeds.

**Key innovation**: Wavelet decomposition separates structure (low-freq) from texture (high-freq), allowing targeted denoising per frequency band.

**Links**:
- Paper: [arXiv:2211.16152](https://arxiv.org/abs/2211.16152)
- Code: [github.com/VinAIResearch/WaveDiff](https://github.com/VinAIResearch/WaveDiff)

### 6b. Fast-cWDM (BraTS 2025 Challenge)

**What**: Conditional wavelet diffusion for 3D MRI synthesis. Reduces denoising steps to 100, uses wavelet-transformed inputs for speed + memory savings.

**Link**: [bioRxiv:2026.02.14.705904](https://www.biorxiv.org/content/10.64898/2026.02.14.705904v1.full)

**RealRestorer applicability**: **MEDIUM-HIGH** -- Image restoration is inherently a frequency-domain problem. Restoring high-frequency details (edges, textures) while preserving low-frequency structure (composition, color) maps perfectly to wavelet decomposition. A wavelet-domain variant of RealRestorer could denoise each frequency band with different step counts (fewer for low-freq, more for high-freq). Major challenge: requires architectural changes to the U-Net to operate on wavelet coefficients instead of RGB latents. This is an innovation opportunity, not a drop-in optimization.

---

## 7. Dynamic Resolution Denoising

### 7a. LSSGen: Latent Space Scaling (2025)

**What**: Progressively upscales latent representation through multiple stages during denoising. Early steps run at lower resolution, later steps at full resolution. No architectural changes or retraining.

**Performance**: 1.5x speedup with 246% improvement in perceptual quality. Works across FLUX, SDXL, SD1.5, Playground, and LCM.

**Key innovation**: ResNet-based latent upsampler + principled schedule-shifting mechanism.

**Links**:
- Paper: [arXiv:2507.16154](https://arxiv.org/html/2507.16154v1)

### 7b. NoiseShift: Resolution-Aware Noise Recalibration (2025)

**What**: Training-free fix for the noise-resolution mismatch problem -- the same noise level removes disproportionately more signal from lower-resolution images.

**Link**: [arXiv:2510.02307](https://arxiv.org/html/2510.02307v1)

### 7c. Latent Space Super-Resolution (CVPR 2025)

**What**: Enables latent upsampling with fewer denoising steps while producing detailed outputs.

**Link**: [CVPR 2025 proceedings](https://openaccess.thecvf.com/content/CVPR2025/papers/Jeong_Latent_Space_Super-Resolution_for_Higher-Resolution_Image_Generation_with_Diffusion_Models_CVPR_2025_paper.pdf)

**RealRestorer applicability**: **HIGH** -- For restoration, early denoising steps establish structure (which doesn't need high resolution), while later steps refine details (which do). Running the first 50% of steps at 0.5x resolution and upscaling for the remaining steps could yield ~1.5-2x speedup with minimal quality loss. LSSGen is training-free and model-agnostic. MPS compatible. This is one of the most practical innovations for our use case.

---

## 8. Progressive Distillation

**What**: Iteratively distills a trained sampler into a model taking half as many steps. Apply repeatedly: 8192 -> 4096 -> 2048 -> ... -> 4 steps.

**Performance**: FID 3.0 on CIFAR-10 with only 4 steps. Full distillation procedure takes less time than original model training.

**Links**:
- Paper: [arXiv:2202.00512](https://arxiv.org/abs/2202.00512)
- ICLR 2022: [openreview.net/forum?id=TIdIXIpzhoI](https://openreview.net/forum?id=TIdIXIpzhoI)
- Blog analysis: [sander.ai/2024/02/28/paradox.html](https://sander.ai/2024/02/28/paradox.html)
- Direct Distillation (2025 improvement): [PMC article](https://pmc.ncbi.nlm.nih.gov/articles/PMC11856141/)

**RealRestorer applicability**: **MEDIUM** -- Progressive distillation is a well-established technique but requires significant training compute for each halving round. For RealRestorer's 28-step default, we'd need: 28->14->7->4 (three rounds). Each round requires training on restoration-specific data. The newer "direct distillation" approaches (2025) skip intermediate rounds and go directly to the target step count, which is more practical.

---

## 9. Classifier-Free Guidance Distillation

### 9a. The Problem

CFG doubles compute: each denoising step runs the U-Net twice (conditional + unconditional), then interpolates. For RealRestorer at 28 steps, this means 56 U-Net forward passes.

### 9b. Adapter Guidance Distillation (AGD) -- 2025

**What**: Keeps the base model frozen, trains lightweight adapters (~2% extra parameters) to approximate CFG in a single forward pass. Effectively halves NFEs.

**Performance**: Comparable or superior FID to CFG with half the neural function evaluations. Enables distillation of 2.6B parameter models on a single 24GB GPU.

**Link**: [arXiv:2503.07274](https://arxiv.org/abs/2503.07274)

### 9c. CFG++ (ICLR 2025)

**What**: Manifold-constrained CFG that improves sample quality and enables compatibility with smaller step counts.

**Link**: [github.com/CFGpp-diffusion/CFGpp](https://github.com/CFGpp-diffusion/CFGpp)

### 9d. Independent Condition Guidance (ICG) -- 2025

**What**: Training-free alternative to CFG. Provides CFG benefits without special training or doubled compute. Can be applied at inference time on any pretrained conditional model.

**Links**:
- Disney Research: [studios.disneyresearch.com/2025/04/23/no-training-no-problem-rethinking-diffusion-guidance-for-diffusion-models/](https://studios.disneyresearch.com/2025/04/23/no-training-no-problem-rethinking-diffusion-guidance-for-diffusion-models/)
- Paper: [openreview.net/forum?id=b3CzCCCILJ](https://openreview.net/forum?id=b3CzCCCILJ)

### 9e. CFG-Free Diffusion -- 2025

**What**: Eliminates CFG entirely during training, building guidance into the model weights.

**Link**: [arXiv:2502.12154](https://arxiv.org/html/2502.12154v1)

**RealRestorer applicability**: **VERY HIGH** -- This is the single highest-impact optimization for RealRestorer. The current pipeline runs CFG at guidance_scale=7.5, meaning 2x U-Net calls per step. Eliminating CFG gives an immediate **2x speedup** with zero quality trade-off if done correctly.

- **AGD (best option)**: Train a tiny adapter LoRA (~2% params) on restoration data to absorb the guidance signal. One-time training cost, permanent 2x speedup. Feasible on a single 24GB GPU.
- **ICG (quick win)**: Training-free, can test immediately. Disney Research backing.
- Combined with step reduction (28->8 steps), total speedup potential: **7x** (3.5x from steps * 2x from CFG elimination).

---

## 10. Flash Diffusion (AAAI 2025 Oral)

**What**: Efficient distillation method that trains a student to predict in a single step what the teacher achieves in multiple steps. Uses adaptive timestep distribution that shifts during training.

**Performance**: 5x speedup (500% faster inference). Reduces training parameters by 2.5-64x vs other distillation methods. Requires only several GPU-hours of training.

**Key innovation**: Adaptive timestep distribution shifts focus during training from noisy to clean predictions. Works across UNet (SD1.5, SDXL) and DiT (Pixart-alpha) architectures.

**Links**:
- Paper: [arXiv:2406.02347](https://arxiv.org/html/2406.02347v1)
- Code: [github.com/gojasper/flash-diffusion](https://github.com/gojasper/flash-diffusion)
- Project page: [gojasper.github.io/flash-diffusion-project](https://gojasper.github.io/flash-diffusion-project/)

**Supported tasks**: text-to-image, inpainting, face-swapping, super-resolution. Can accelerate existing LoRAs in a training-free manner.

**RealRestorer applicability**: **HIGH** -- Flash Diffusion explicitly supports super-resolution and uses SDXL-compatible architectures. The training cost (few GPU-hours) is much lower than progressive distillation. Can accelerate existing LoRAs without retraining them. Key advantage: works with 4-step LCMScheduler inference. Combined with CFG distillation, could achieve **8-10x total speedup**.

---

## 11. PAG (Perturbed Attention Guidance)

**What**: Replaces self-attention maps with identity matrices to generate "degraded structure" samples, then guides denoising away from these degraded versions. Published at ECCV 2024, updated July 2025.

**Performance**: Improves quality in unconditional settings where CFG cannot be used. Particularly effective for ControlNet with empty prompts and image restoration (inpainting, deblurring).

**Key innovation**: Uses the model's own attention mechanism as guidance signal -- no external classifier or doubled compute needed for unconditional guidance.

**Links**:
- Paper: [arXiv:2403.17377](https://arxiv.org/abs/2403.17377)
- Code: [github.com/cvlab-kaist/Perturbed-Attention-Guidance](https://github.com/cvlab-kaist/Perturbed-Attention-Guidance)
- Diffusers integration: [HF PAG docs](https://huggingface.co/docs/diffusers/en/using-diffusers/pag)
- ComfyUI extensions: [github.com/pamparamm/sd-perturbed-attention](https://github.com/pamparamm/sd-perturbed-attention)

**Can PAG replace CFG for single-pass inference?**
- **Partially yes**: PAG provides guidance without the 2x compute cost of CFG. However, PAG still requires modifying attention maps (substituting with identity), which means a second forward pass with modified attention. The computational overhead is less than full CFG but not zero.
- **For restoration tasks specifically**: PAG is explicitly highlighted for image restoration (inpainting, deblurring). Since RealRestorer's tasks (deblur, denoise, dehaze, etc.) align perfectly, PAG could be a better guidance mechanism than CFG for our use case.
- **Combined approach**: PAG + reduced CFG scale could maintain quality with lower total compute. Some implementations use PAG_scale=3.0 + CFG_scale=3.0 instead of CFG_scale=7.5.

**RealRestorer applicability**: **HIGH** -- Native diffusers support, directly relevant to restoration tasks, reduces guidance overhead. Can be tested immediately without training.

---

## 12. Additional Cutting-Edge Techniques

### 12a. T-Stitch: Trajectory Stitching (ICLR 2025, NVIDIA)

**What**: Uses a smaller, faster model for early denoising steps (where global structure is determined) and switches to the full model for later steps (detail refinement). Training-free.

**Performance**: On DiT-XL, 40% of early steps replaced by 10x faster DiT-S without quality drop. Works with SD models.

**Links**:
- Code: [github.com/NVlabs/T-Stitch](https://github.com/NVlabs/T-Stitch)
- Paper: [arXiv:2402.14167](https://arxiv.org/abs/2402.14167)
- ICLR 2025: [openreview.net/forum?id=rnHqwPH4TZ](https://openreview.net/forum?id=rnHqwPH4TZ)

**RealRestorer applicability**: **MEDIUM** -- Would require a smaller "companion" U-Net for early steps. Since RealRestorer has a single 39GB model, we'd need to train or find a compatible smaller model. The concept is sound but the requirement for a pre-existing smaller model limits immediate applicability.

### 12b. Shortcut Models (2025)

**What**: Single network that conditions on both noise level and desired step size, allowing variable-budget inference from 1 to N steps. Self-distillation training.

**Performance**: 128x reduction in sampling time. Higher quality than consistency models and reflow at all step budgets.

**Links**:
- Paper: [arXiv:2410.12557](https://arxiv.org/abs/2410.12557)
- Follow-ups: [arXiv:2512.11831](https://arxiv.org/abs/2512.11831), [arXiv:2502.00688](https://arxiv.org/abs/2502.00688)

**RealRestorer applicability**: **MEDIUM** -- Requires retraining with shortcut conditioning. The flexible step budget at inference time is attractive for a CLI tool where users can trade speed vs quality. Not a drop-in solution.

### 12c. Glance: Phase-Aware LoRA Experts (2025)

**What**: Two lightweight LoRA adapters specializing in early (semantic) and late (refinement) denoising phases. Slow-LoRA stabilizes early steps, Fast-LoRA accelerates late steps.

**Performance**: 5x acceleration. Trained on 1 sample using a single V100 in 1 hour. Generalizes to unseen prompts.

**Link**: [arXiv:2512.02899](https://arxiv.org/abs/2512.02899)

**RealRestorer applicability**: **HIGH** -- Extremely low training cost (1 sample, 1 hour, 1 GPU). The phase-aware approach aligns with restoration: early steps establish structural fidelity, late steps refine textures. We could train Slow-LoRA and Fast-LoRA on a single restoration example. MPS compatible (standard LoRA inference).

### 12d. SageAttention (ICLR 2025 + ICML 2025 + NeurIPS 2025 Spotlight)

**What**: Quantized attention kernel achieving 2-5x speedup over FlashAttention2 without end-to-end metric loss. SageAttention2 uses INT4 quantization.

**Performance**: 2.1x over FlashAttention2, 2.7x over xformers. SageAttention2: 3x over FA2, 4.5x over xformers.

**Links**:
- Code: [github.com/thu-ml/SageAttention](https://github.com/thu-ml/SageAttention)
- Paper: [arXiv:2410.02367](https://arxiv.org/abs/2410.02367)

**RealRestorer applicability**: **LOW for Apple Silicon** -- SageAttention requires CUDA kernels (Hopper, Ampere, RTX). No MPS/Metal support. Only relevant for CUDA deployment. For MPS, we already use the PyTorch native scaled_dot_product_attention which uses Metal-optimized kernels.

### 12e. Consistency Flow Matching for Super-Resolution (ICCV 2025)

**What**: One-step super-resolution via consistency rectified flow. TSD-SR achieves 40x faster than SeeSR with single-step inference.

**Links**:
- TSD-SR (CVPR 2025): [paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Dong_TSD-SR_One-Step_Diffusion_with_Target_Score_Distillation_for_Real-World_Image_CVPR_2025_paper.pdf)
- CTM for SR (ICCV 2025): [paper](https://openaccess.thecvf.com/content/ICCV2025/papers/You_Consistency_Trajectory_Matching_for_One-Step_Generative_Super-Resolution_ICCV_2025_paper.pdf)
- Latent Consistency Flow Matching: [arXiv:2502.03500](https://arxiv.org/html/2502.03500)
- MFSR (MeanFlow SR): [openreview.net/forum?id=qDg8KNq0Fm](https://openreview.net/forum?id=qDg8KNq0Fm)

**RealRestorer applicability**: **RESEARCH** -- These are purpose-built SR models, not drop-in accelerations for RealRestorer. However, the consistency trajectory matching concept could inspire a one-step fine-tuning of RealRestorer for specific restoration tasks where speed matters most.

### 12f. TurboDiffusion (2025)

**What**: Combines SageAttention + Sparse-Linear Attention (SLA) + timestep distillation (rCM) for 100-200x video diffusion speedup.

**Link**: [github.com/thu-ml/TurboDiffusion](https://github.com/thu-ml/TurboDiffusion)

**RealRestorer applicability**: **LOW** -- Video-focused, CUDA-only. Concepts (stacked acceleration layers) are inspiring but components are NVIDIA-specific.

---

## 13. Comprehensive Surveys & Repos

### Surveys

| Survey | Venue | Link |
|--------|-------|------|
| Efficient Diffusion Models: A Survey | TMLR 2025 | [arXiv:2502.06805](https://arxiv.org/abs/2502.06805) |
| Efficient Diffusion Models: Principles to Practices | TPAMI 2025 | [arXiv:2410.11795](https://arxiv.org/abs/2410.11795) |

### Curated GitHub Repos

| Repo | Description | Link |
|------|-------------|------|
| Efficient-Diffusion-Model-Survey | TMLR 2025 survey companion, 200+ papers | [GitHub](https://github.com/AIoT-MLSys-Lab/Efficient-Diffusion-Model-Survey) |
| Awesome-Generation-Acceleration | Curated acceleration resources | [GitHub](https://github.com/xuyang-liu16/Awesome-Generation-Acceleration) |
| PostDiff (ICCV 2025) | Compute-optimal deployment | [GitHub](https://github.com/GATECH-EIC/PostDiff) |
| Awesome Consistency Models | Curated list of consistency model papers | [GitHub](https://github.com/G-U-N/Awesome-Consistency-Models) |
| Awesome Diffusion ICLR 2025 | All diffusion submissions to ICLR 2025 | [GitHub](https://github.com/moatifbutt/awesome-diffusion-iclr-2025) |

---

## 14. Applicability Matrix for RealRestorer

Ranked by **impact * feasibility** for Apple Silicon MPS deployment:

| Rank | Technique | Speedup | Training? | MPS OK? | Difficulty | Compounds? |
|------|-----------|---------|-----------|---------|------------|------------|
| 1 | **CFG Distillation (AGD/ICG)** | 2x | Minimal/None | Yes | Low-Medium | Yes |
| 2 | **DeepCache** | 1.5-2.3x | None | Yes | Low | Yes |
| 3 | **ToMe/ToMA token merging** | 1.2-1.5x | None | Yes | Low | Yes |
| 4 | **LCM-LoRA distillation** | 4-7x | Medium | Yes | Medium | Partial |
| 5 | **DMD2 LoRA** | 4-7x | Medium | Yes | Medium | Partial |
| 6 | **Dynamic resolution (LSSGen)** | 1.5-2x | None | Yes | Medium | Yes |
| 7 | **Flash Diffusion** | 5x | Low (hours) | Yes | Medium | Partial |
| 8 | **Glance LoRA experts** | 5x | Minimal (1h) | Yes | Low-Medium | Partial |
| 9 | **PAG guidance** | 1.3-1.5x | None | Yes | Low | Yes |
| 10 | **Wavelet diffusion** | 2-3x | High | Yes | High | No |
| 11 | **Progressive distillation** | 7x | High | Yes | High | No |
| 12 | **T-Stitch** | 1.5-2x | None | Yes | Medium | Yes |
| 13 | **Shortcut models** | 10-128x | Very High | Yes | Very High | No |
| 14 | **SageAttention** | 2-5x | None | **No** | N/A (CUDA) | Yes |
| 15 | **Parallel denoising** | 2-4x | None | **No** | N/A (multi-GPU) | Yes |

### Compound Speedup Potential (Stackable Techniques)

The following can be stacked without interference:

```
Base:       28 steps @ CFG 7.5 = 56 U-Net calls                    = 1.0x
+ CFG elimination (AGD/ICG): 28 steps, 1 pass each = 28 calls      = 2.0x
+ Step reduction (8 steps via DPM++ SDE Karras): 8 calls            = 7.0x
+ DeepCache (cache every other step): ~5 effective calls             = 11.2x
+ ToMe (20% token reduction): ~4 effective calls equivalent         = 14.0x
+ Dynamic resolution (early steps at 0.5x): ~3.5 effective          = 16.0x
```

**Theoretical maximum compound speedup: ~16x** (56 effective calls -> ~3.5 equivalent full-resolution, full-token, single-pass calls).

Practical estimate with quality preservation: **8-12x**.

---

## 15. Recommended Implementation Roadmap

### Phase 1: Zero-Training Quick Wins (Days)

1. **DeepCache integration** -- `pip install deepcache`, patch pipeline. Expect 1.5-2x. Zero risk.
2. **ToMe/ToMeSD** -- `import tomesd; tomesd.apply_patch(pipe, ratio=0.4)`. Expect 1.2x. Conservative ratio for restoration.
3. **PAG testing** -- Use diffusers PAG pipeline variant. Test quality vs CFG for restoration tasks.
4. **ICG (Independent Condition Guidance)** -- Training-free CFG replacement from Disney Research. Test immediately.

### Phase 2: Minimal Training (Weeks)

5. **AGD (Adapter Guidance Distillation)** -- Train ~2% adapter params to absorb CFG. Single 24GB GPU. 2x speedup permanent.
6. **Glance LoRA experts** -- Train Slow-LoRA + Fast-LoRA on restoration examples. 1 hour on single GPU.
7. **LCM-LoRA or DMD2-LoRA** -- Distill step reduction into LoRA adapter. May use existing SDXL LoRAs as starting point.

### Phase 3: Architecture Innovation (Months)

8. **Dynamic resolution denoising** -- Implement LSSGen-style latent upscaling mid-trajectory.
9. **Flash Diffusion training** -- Full distillation for 4-step inference.
10. **Wavelet-domain restoration** -- Novel architecture operating on wavelet coefficients. Highest innovation potential, highest effort.

### Phase 4: Research Frontiers (Exploratory)

11. **Speculative decoding for image diffusion** -- Novel research direction: small U-Net drafts, RealRestorer verifies.
12. **Shortcut model training** -- Self-distillation for arbitrary step budgets at inference time.
13. **Consistency trajectory matching** -- One-step restoration model.

---

## Key Takeaways

1. **CFG elimination is the lowest-hanging fruit with highest impact** -- 2x speedup from a single change, applicable today via ICG (training-free) or AGD (minimal training).

2. **Caching techniques (DeepCache, ToCa, FastCache) are the most practical** -- Zero training, plug-and-play, MPS compatible, and stack with everything else.

3. **Distillation (LCM/DMD2/Flash) provides the largest single-technique speedups** -- But requires training investment. DMD2 LoRAs for SDXL may work out-of-the-box.

4. **Token merging is free performance** -- Small but compounds with everything. Must use conservative ratios for restoration fidelity.

5. **Parallel denoising and SageAttention are CUDA-only** -- Not applicable to our primary Apple Silicon target. Reserve for cloud deployment.

6. **The theoretical compound speedup ceiling is ~16x** -- From 56 U-Net calls to ~3.5 effective full-compute equivalents. Practical target: 8-12x with quality preservation.

7. **Wavelet diffusion and speculative decoding are the most innovative frontiers** -- No existing implementations for image restoration, but the concepts are sound and could differentiate RealRestorer from competitors.
