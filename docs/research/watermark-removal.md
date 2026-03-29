# Invisible AI Watermark Detection & Removal: Research Findings

## Table of Contents
1. [Watermark Taxonomy](#1-watermark-taxonomy)
2. [StegaStamp](#2-stegastamp)
3. [Tree-Ring Watermarks](#3-tree-ring-watermarks)
4. [Google SynthID](#4-google-synthid)
5. [Meta Stable Signature](#5-meta-stable-signature)
6. [C2PA / Content Credentials](#6-c2pa--content-credentials)
7. [Spectral Analysis & Frequency Domain Techniques](#7-spectral-analysis--frequency-domain-techniques)
8. [DWT-Based Removal](#8-dwt-based-removal)
9. [DCT-Based Removal](#9-dct-based-removal)
10. [Diffusion Pipeline Natural Removal](#10-diffusion-pipeline-natural-removal)
11. [Universal Attack Methods](#11-universal-attack-methods)
12. [Detection Before Removal](#12-detection-before-removal)
13. [Open-Source Tools](#13-open-source-tools)
14. [Implementation Strategy for RealRestore](#14-implementation-strategy-for-realrestore)

---

## 1. Watermark Taxonomy

AI image watermarks fall into three categories:

### Post-Generation (Post-hoc)
Watermarks applied after image generation. The watermark encoder modifies pixel values to embed a message.
- **StegaStamp**: Embeds bitstrings into global image features via a trained encoder network
- **HiDDeN**: Neural network-based data hiding
- **TrustMark**: Advanced invisible watermark with robustness claims
- **VINE**: Watermarking with generative priors

### In-Generation (Latent Space)
Watermarks embedded during the diffusion process itself.
- **Tree-Ring**: Embeds pattern in Fourier transform of initial noise vector
- **Gaussian Shading**: Modifies initial noise distribution
- **PRC**: Provably undetectable watermarks in the latent space

### Model-Level (Decoder Modification)
Watermarks baked into the model weights so all outputs are watermarked.
- **Stable Signature** (Meta): Fine-tunes the VAE decoder to embed watermarks
- **SynthID** (Google): Spread-spectrum phase encoding during generation

### Metadata-Based
Not pixel-level; stored in file metadata.
- **C2PA / Content Credentials**: EXIF, PNG chunks, signed manifests
- **EXIF AI tags**: Model name, prompt, seed, parameters

---

## 2. StegaStamp

### How It Works
StegaStamp trains an encoder-decoder pair end-to-end. The encoder takes an image and a 100-bit message, producing a visually identical output with the message embedded in global image features. The decoder recovers the bitstring. It is designed to survive physical-world perturbations (printing, photography, rotation, illumination changes).

**Embedding domain**: Low and mid-frequency bands (spread across the image).

### Detection
- Requires the trained StegaStamp decoder model
- Decode the image and check bit accuracy against expected message
- Threshold: typically >75% bit accuracy indicates watermark presence

### Removal Approaches

#### VAE Fine-tuning (NeurIPS 2024 Challenge Winner)
The most effective known approach:
1. **Paired data generation**: Create 1,000 pairs using SD 2.1 — each image embedded with a random 100-bit message and its inverse
2. **Fine-tune SDXL VAE**: Minimize MSE between watermarked input and inverse-message target
   - Adam optimizer, lr=1e-5, 10 epochs, batch 16, gradient clipping (max norm 1.0)
   - Completes in ~2 GPU hours on A6000
3. **Test-time optimization**: Refine VAE per-image using combined MSE + LPIPS + SSIM loss
4. **CIELAB color restoration**: Preserve luminance from optimized output, adopt chrominance from original
   - Results: PSNR improvement >6dB, SSIM +0.176 vs VAE-only baseline

#### Deep Image Prior (DIP)
- Untrained CNN optimized to reconstruct the image, naturally separating image content from watermark artifacts
- **Limited effectiveness** against StegaStamp because it embeds in low/mid-frequency bands that DIP picks up early in optimization
- Works well against DwtDctSVD, rivaGAN, SSL watermarks (PSNR ~34-36 dB)

#### Diffusion Regeneration
- Encode to latent space, add noise, denoise
- StegaStamp reduced from 99.8% to 7.4% detection accuracy
- Guided diffusion (with decoder feedback) reduces to 0.0%

#### UnMarker (IEEE S&P 2025)
- Spectral disruption reduces StegaStamp detection to ~43%
- No detector feedback needed, fully black-box

---

## 3. Tree-Ring Watermarks

### How It Works
Embeds a circular pattern in the Fourier transform of the initial noise vector used for diffusion generation. Detection inverts the diffusion process (DDIM inversion) to recover the noise, then checks for the pattern in Fourier space.

**Key vulnerability**: Operates in the frequency domain of the latent noise, specifically as phase patterns at specific frequency coordinates.

### Detection
- Requires access to the same diffusion model for DDIM inversion
- Compute 2D FFT of recovered noise, check Fourier magnitude at center coordinates
- Compare against threshold (AUC typically >0.99 for clean detection)

### Removal Approaches

#### Spatial Translation (Simple, Effective)
From NeurIPS 2024 challenge winner:
- Apply a 7-pixel horizontal shift to disrupt phase-based watermarks
- Restore leftmost columns from original to hide boundary artifacts
- **No model training required** — purely geometric
- Reduces detection AUC from 0.993 to near chance

#### VAE Surrogate Attack (USENIX Security 2025)
From "A Crack in the Bark":
- Only requires the publicly available VAE used to train the diffusion model
- Reduces Tree-Ring detector AUC from 0.993 to 0.153
- Maintains high image quality
- Exploits the common practice of sharing/reusing VAEs across diffusion models

#### Steganalysis-Based Attack
- Identifies content-agnostic ripple pattern in Tree-Ring images
- Subtracts the pattern to evade detection with minimal quality loss

#### Diffusion Regeneration
- Standard img2img with ~30% noise addition erases Tree-Ring watermarks
- No knowledge of the watermark needed

---

## 4. Google SynthID

### How It Works
SynthID uses spread-spectrum phase encoding in the frequency domain during image generation. It embeds watermark information at specific frequency bins with consistent phase relationships. The watermark is woven into the generation process across specific frequency coordinates.

**Identified carrier frequencies**: (+-14, +-14), (+-126, +-14), (+-98, +-14), (+-128, +-128), (+-210, +-14), (+-238, +-14).

### Detection
- Google's proprietary detector checks for phase coherence at known frequency coordinates
- Correlation threshold ~0.179 against a baseline codebook
- Available via Google AI Studio / Vertex AI

### Removal Approaches

#### Frequency-Targeted Subtraction (V3 Approach)
1. Compute FFT of each color channel separately
2. Identify bins with high magnitude in the codebook AND phase consistency
3. Subtract watermark magnitude (capped at 30% of image magnitude at that frequency)
4. Perform vector subtraction along watermark's phase direction
5. Apply inverse FFT to reconstruct
- Reduces confidence below detection threshold while maintaining visual quality

#### Diffusion Regeneration
- Re-process watermarked image through a diffusion model pipeline using the original as structural guide
- Reported ~79% success rate (disputed by Google)
- ComfyUI with custom nodes is the common implementation path
- Requires GPU with 16GB+ VRAM

#### UnMarker
- Achieves 79% attack success rate against SynthID
- Uses spectral disruption without detector feedback

#### Commercial Tools
- AISEO SynthID Remover, ChromaStudio, RemoveMySynthID.com
- Most degrade detection confidence rather than fully eliminating the watermark
- Effectiveness varies by specific implementation

---

## 5. Meta Stable Signature

### How It Works
Fine-tunes the VAE decoder of a diffusion model so that ALL generated images inherently contain the watermark. The watermark is baked into the model weights rather than applied post-hoc. Detection uses a trained watermark extractor on generated images.

### Detection
- Watermark extractor network compares decoded bits against expected message
- Works on any image generated by the watermarked model

### Removal Approaches

#### Model Fine-tuning Attack
- Fine-tune the watermarked diffusion model on new data to remove the watermark from weights
- Generated images from fine-tuned model are non-watermarked
- Maintains visual quality of generated images
- Paper: "Stable Signature is Unstable" (ICLR 2025)

#### Diffusion Regeneration
- Detection rate drops to 0.000 TPR@0.1%FPR
- FID: 23.57 (minimal quality degradation)
- Two iterations with timestep 20-100 per diffusion step balance low detection with high quality

#### UnMarker
- Spectral disruption effective against Stable Signature
- Black-box, no detector feedback needed

---

## 6. C2PA / Content Credentials

### How It Works
C2PA is a metadata-based provenance standard. Content Credentials are stored as signed manifests in file metadata (EXIF, XMP, PNG chunks). Since C2PA 2.1, "durable credentials" use invisible watermarks as a soft binding to allow credential recovery even after metadata stripping.

### Detection
- Read EXIF/XMP metadata for C2PA manifest
- Check PNG text chunks for `c2pa` keys
- Use C2PA verification libraries (c2pa-rs, c2pa-node)
- For durable credentials: check for Digimarc watermark pointing to credential store

### Removal
C2PA metadata is trivially stripped:
```python
from PIL import Image
img = Image.open("input.png")
# Save without metadata
img.save("output.png")  # PNG strips metadata by default in many configurations
```

For comprehensive metadata removal:
- Strip EXIF: `piexif.remove("image.jpg")`
- Strip PNG chunks: re-encode pixel data to new PNG
- Strip XMP: remove XMP packet from file
- C2PA durable binding (watermark-based): requires watermark removal techniques from other sections

### Tools
- `noai-watermark --remove-ai` — strips AI metadata including C2PA manifests
- `exiftool -all= image.png` — removes all metadata
- Python `Pillow` + `piexif` for programmatic stripping

---

## 7. Spectral Analysis & Frequency Domain Techniques

### Core Principle
All invisible watermarks must modify the frequency spectrum of an image. Detection and removal therefore center on spectral analysis.

### Analysis Pipeline
1. **2D FFT Analysis**: Compute magnitude and phase spectrum
   - Watermarks appear as anomalous patterns in specific frequency bands
   - Compare against natural image statistics to identify watermark signatures
2. **Power Spectral Density**: Identify frequency bands with unnatural energy distribution
3. **Phase Coherence Analysis**: Check for consistent phase patterns across color channels (used by SynthID)

### Frequency Band Classification
| Band | Content | Watermark Type |
|------|---------|---------------|
| Low frequency | Image structure, color | Semantic watermarks (Stable Signature) |
| Mid frequency | Textures, edges | StegaStamp, HiDDeN |
| High frequency | Fine detail, noise | Tree-Ring artifacts, DWT watermarks |

### UnMarker's Spectral Disruption (IEEE S&P 2025)
The most comprehensive frequency-domain attack:

**Stage 1 — High-Frequency Disruption** (non-semantic watermarks):
- Direct Fourier Loss (DFL): `||FT(x) - FT(y)||_1`
- Perceptual constraints via LPIPS-Alex (<256px) or DeepVGG-LPIPS (>256px)
- L2 norm bounds prevent visual artifacts
- Optimizes pixel modifications to maximize spectral difference from watermarked image

**Stage 2 — Low-Frequency Filtering** (semantic watermarks):
- Learns optimizable per-pixel filter kernels
- Filter Regularization Loss (FRL): minimize pixel distances from local medians
- Mean Pool Loss (MPL): constrain patch mean values
- Stacks T filters with varying kernel sizes for multi-directional disruption

**Effectiveness**: 57-100% watermark removal depending on scheme, superior quality vs regeneration attacks.

### FreqMark (2024)
Embeds 48-bit watermarks with >90% accuracy, effective against compression, cropping, and AI regeneration. Represents the current frontier of frequency-domain robustness.

### Implementation with Python
```python
import numpy as np
from scipy.fft import fft2, ifft2, fftshift

def analyze_watermark_spectrum(image: np.ndarray) -> dict:
    """Analyze frequency spectrum for watermark signatures."""
    results = {}
    for ch in range(image.shape[2]):
        channel = image[:, :, ch].astype(np.float64)
        F = fftshift(fft2(channel))
        magnitude = np.abs(F)
        phase = np.angle(F)

        # Compute radial power spectrum
        h, w = magnitude.shape
        cy, cx = h // 2, w // 2
        Y, X = np.ogrid[:h, :w]
        R = np.sqrt((X - cx)**2 + (Y - cy)**2).astype(int)
        radial_profile = np.bincount(R.ravel(), magnitude.ravel()) / np.bincount(R.ravel())

        results[f'ch{ch}_magnitude'] = magnitude
        results[f'ch{ch}_phase'] = phase
        results[f'ch{ch}_radial'] = radial_profile
    return results

def remove_frequency_watermark(image: np.ndarray,
                                freq_coords: list,
                                strength: float = 0.3) -> np.ndarray:
    """Remove watermark by suppressing specific frequency coordinates."""
    output = image.copy().astype(np.float64)
    for ch in range(image.shape[2]):
        F = fft2(output[:, :, ch])
        F_shifted = fftshift(F)

        for (fy, fx) in freq_coords:
            h, w = F_shifted.shape
            cy, cx = h // 2, w // 2
            # Suppress magnitude at watermark frequencies
            for dy, dx in [(fy, fx), (-fy, -fx), (fy, -fx), (-fy, fx)]:
                y, x = cy + dy, cx + dx
                if 0 <= y < h and 0 <= x < w:
                    F_shifted[y, x] *= (1 - strength)

        output[:, :, ch] = np.real(ifft2(fftshift(F_shifted)))
    return np.clip(output, 0, 255).astype(np.uint8)
```

---

## 8. DWT-Based Removal

### Theory
The Discrete Wavelet Transform decomposes an image into frequency subbands:
- **LL** (approximation): Low-frequency content — main image structure
- **LH** (horizontal detail): Horizontal edges and textures
- **HL** (vertical detail): Vertical edges and textures
- **HH** (diagonal detail): High-frequency noise and fine detail

Watermarks embedded via DWT typically modify coefficients in LH, HL, or HH subbands.

### Removal Strategy
1. Apply multi-level DWT decomposition (typically 2-3 levels)
2. Identify subbands containing watermark energy (usually HH and parts of LH/HL)
3. Apply coefficient thresholding or suppression in watermark-bearing subbands
4. Reconstruct via inverse DWT

### Implementation
```python
import pywt
import numpy as np

def dwt_watermark_removal(image: np.ndarray,
                           wavelet: str = 'haar',
                           level: int = 3,
                           threshold_factor: float = 0.5) -> np.ndarray:
    """Remove DWT-embedded watermarks via coefficient suppression."""
    output = image.copy().astype(np.float64)

    for ch in range(image.shape[2]):
        channel = output[:, :, ch]
        coeffs = pywt.wavedec2(channel, wavelet, level=level)

        # Process detail coefficients (skip approximation LL)
        new_coeffs = [coeffs[0]]  # Keep LL unchanged
        for detail_level in coeffs[1:]:
            new_detail = []
            for subband in detail_level:  # LH, HL, HH
                # Soft threshold to suppress watermark energy
                threshold = threshold_factor * np.std(subband)
                subband_clean = pywt.threshold(subband, threshold, mode='soft')
                new_detail.append(subband_clean)
            new_coeffs.append(tuple(new_detail))

        output[:, :, ch] = pywt.waverec2(new_coeffs, wavelet)[:channel.shape[0], :channel.shape[1]]

    return np.clip(output, 0, 255).astype(np.uint8)
```

### Libraries
- **PyWavelets** (`pywt`): Primary Python wavelet library
- Supports Haar, Daubechies (db2-db20), Symlets, Coiflets wavelets
- `pywt.wavedec2` / `pywt.waverec2` for 2D decomposition/reconstruction

### Limitations
- Effective mainly against simple DWT-embedded watermarks
- Modern neural watermarks (StegaStamp, SynthID) spread energy across bands in non-trivial patterns
- Aggressive thresholding degrades image quality

---

## 9. DCT-Based Removal

### Theory
The Discrete Cosine Transform represents image blocks as sums of cosine functions at different frequencies. JPEG compression uses 8x8 block DCT. Watermarks embedded in DCT domain typically modify mid-frequency coefficients.

### Removal Strategy
1. Apply block-based DCT (typically 8x8 blocks)
2. Identify modified mid-frequency coefficients
3. Apply quantization or coefficient replacement
4. Reconstruct via inverse DCT

### Relationship to JPEG Compression
JPEG compression naturally disrupts DCT-based watermarks through quantization:
- Quality 75-85: Partial watermark degradation
- Quality 50-65: Significant watermark disruption
- Quality <50: Near-complete removal but with visible quality loss

### Implementation
```python
import numpy as np
from scipy.fft import dct, idct

def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct2(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

def dct_watermark_removal(image: np.ndarray,
                           block_size: int = 8,
                           quant_factor: float = 0.8) -> np.ndarray:
    """Remove DCT-embedded watermarks via coefficient quantization."""
    output = image.copy().astype(np.float64)

    for ch in range(image.shape[2]):
        channel = output[:, :, ch]
        h, w = channel.shape
        # Pad to multiple of block_size
        pad_h = (block_size - h % block_size) % block_size
        pad_w = (block_size - w % block_size) % block_size
        padded = np.pad(channel, ((0, pad_h), (0, pad_w)), mode='reflect')

        for i in range(0, padded.shape[0], block_size):
            for j in range(0, padded.shape[1], block_size):
                block = padded[i:i+block_size, j:j+block_size]
                dct_block = dct2(block)

                # Quantize mid-frequency coefficients
                # (where watermarks are typically embedded)
                for u in range(block_size):
                    for v in range(block_size):
                        freq = u + v
                        if 3 <= freq <= 10:  # Mid-frequency range
                            dct_block[u, v] = np.round(
                                dct_block[u, v] * quant_factor
                            ) / quant_factor

                padded[i:i+block_size, j:j+block_size] = idct2(dct_block)

        output[:, :, ch] = padded[:h, :w]

    return np.clip(output, 0, 255).astype(np.uint8)
```

### Limitations
- Block artifacts at aggressive quantization levels
- Modern neural watermarks are designed to survive JPEG/DCT quantization
- Most effective against older DCT-domain watermark schemes

---

## 10. Diffusion Pipeline Natural Removal

### Key Finding
**Standard diffusion-based image restoration/editing naturally removes invisible watermarks as a side effect of reconstruction.** This is the single most important finding for RealRestore.

### Theoretical Basis
From "Vanishing Watermarks" (arXiv 2602.20680, Feb 2026):
> "As diffusion processes progress, mutual information between the watermarked image and hidden payload approaches zero."

Diffusion models lack knowledge of watermark patterns once noise has corrupted them beyond recognition, and thus have no reason to regenerate the watermark during denoising.

### Quantitative Evidence

| Watermark | Original Detection | After Diffusion Regen | After Guided Diffusion |
|-----------|-------------------|----------------------|----------------------|
| StegaStamp | 99.8% | 7.4% | 0.0% |
| TrustMark | 99.9% | 12.8% | 0.0% |
| VINE | 100% | 24.5% | 1.6% |
| Stable Signature | High | 0.0% TPR@0.1%FPR | N/A |
| Tree-Ring | 99.3% AUC | Near chance | N/A |

### Minimum Effective Parameters
- **Noise strength**: ~30% (strength=0.3 in img2img) for unguided removal
- **Steps**: 20-50 denoising steps sufficient
- **Quality preservation**: PSNR ~31.8 dB, SSIM ~0.95 at 30% noise
- **Lower noise (4-15%)**: Partial removal, sufficient for most watermarks

### Critical Insight for RealRestore
Since RealRestore already runs images through diffusion-based restoration pipelines (likely with ControlNet, IP-Adapter, or similar structural guidance), **watermark removal is an inherent side effect of the restoration process**. The key parameters are:

1. **Noise level during restoration**: Even moderate noise (10-30%) disrupts watermarks
2. **Number of denoising steps**: More steps = more thorough watermark erasure
3. **VAE encode/decode cycle**: The VAE itself disrupts some watermark types

### What This Means
- No separate watermark removal module may be needed if restoration strength is sufficient
- For guaranteed removal, a dedicated pre-processing step with slightly higher noise can be added
- The trade-off is always: more noise = better removal but more visual change

---

## 11. Universal Attack Methods

### NeurIPS 2024 Challenge Winner — Multi-Strategy Approach
**Overall result**: 95.7% watermark removal with negligible quality impact.

#### Beige-Box (Known Method)
- StegaStamp: VAE fine-tuning + CIELAB color restoration
- Tree-Ring: 7-pixel spatial translation

#### Black-Box (Unknown Method)
1. **Cluster images** by artifact type:
   - Cluster 1: No artifacts → aggressive denoising (strength=0.16)
   - Cluster 2: Spatial boundary patterns → VAE pipeline
   - Cluster 3: Circular Fourier patterns → VAE pipeline
   - Cluster 4: Square Fourier patterns → minimal perturbation (strength=0.04) + translation
2. **Image-to-image diffusion** with SD XL Refiner
   - 500 inference steps, CFG scale=1.0
   - ChatGPT-4 generated captions for semantic guidance
3. **Results**: Detection score 0.043, PSNR ~28 dB

### UnMarker (IEEE S&P 2025)
**Universal spectral disruption** — works without detector feedback.

| Watermark | Removal Rate |
|-----------|-------------|
| HiDDeN | 100% |
| Yu2 | 100% |
| PTW | High |
| StegaStamp | 57% (to 43% detection) |
| Stable Signature | High |
| SynthID | 79% |
| Tree-Ring | High |

**Requirements**: NVIDIA GPU >=32GB VRAM, CUDA 12, ~30GB storage.

### Provably Removable (NeurIPS 2024 Paper)
Theoretical result by Zhao et al.: any watermark that is imperceptible can be removed by a sufficiently powerful generative model, because the watermarked distribution is close to the natural image distribution.

---

## 12. Detection Before Removal

### UWPD / FSNet — Agnostic Watermark Presence Detection
A detection-first approach using a Frequency Shield Network (FSNet):

**Architecture**:
1. Adaptive Spectral Perception Module (ASPM): Learnable frequency gating to amplify high-freq watermark signals
2. ResNet-50 backbone for spatial features
3. Dynamic Multi-Spectral Attention (DMSA): Multi-branch DCT + tri-stream extremum pooling

**Detection accuracy** (leave-one-algorithm-out):
| Watermark | Accuracy |
|-----------|----------|
| HiDDeN | 98.5% |
| DWT | 94.5% |
| DCT | 93.1% |
| StegaStamp | 91.5% |
| Stable Signature | 90.1% |
| SynthID | 89.4% |
| Tree-Ring | 85.9% |
| LSB | ~59% |

### Simple Detection Heuristics
For a practical detection pipeline:
1. **Metadata check**: Look for C2PA manifests, AI-related EXIF tags, PNG text chunks
2. **FFT anomaly detection**: Compare frequency spectrum against natural image statistics
3. **Noise residual analysis**: Subtract denoised version and analyze residual patterns
4. **Specific decoder probing**: Run known decoder models (StegaStamp, Tree-Ring DDIM inversion) if available

---

## 13. Open-Source Tools

### noai-watermark
**Best all-in-one tool for our purposes.**
- **Repo**: https://github.com/mertizci/noai-watermark
- **Install**: `pip install noai-watermark`
- **Targets**: SynthID, StableSignature, TreeRing, C2PA metadata
- **Method**: Diffusion-based regeneration (VAE encode → noise → denoise → decode)
- **Profiles**: Default (simple img2img) and CtrlRegen (ControlNet + DINOv2 IP-Adapter + histogram matching)
- **MPS support**: Auto-detects CUDA > MPS > CPU
- **Key params**: `--strength 0.04` (minimal) to `--strength 0.7` (maximum removal)

Python API:
```python
from watermark_remover import WatermarkRemover
remover = WatermarkRemover(model_id="Lykon/dreamshaper-8", device="mps")
remover.remove_watermark(
    image_path=Path("input.png"),
    output_path=Path("output.png"),
    strength=0.04,
    num_inference_steps=50,
)
```

### WatermarkAttacker (NeurIPS 2024)
- **Repo**: https://github.com/XuandongZhao/WatermarkAttacker
- **Method**: Regeneration attacks (Regen-Diffusion, Regen-VAE)
- **Key files**: `wmattacker.py`, `regen_pipe.py`
- **Params**: `noise_step` (30/60/100) for diffusion, `quality` (1-6) for VAE

### UnMarker (IEEE S&P 2025)
- **Repo**: https://github.com/andrekassis/ai-watermark
- **Method**: Two-stage spectral disruption (high-freq + low-freq)
- **Requirements**: NVIDIA GPU >=32GB VRAM (not suitable for MPS without adaptation)
- **Usage**: `python attack.py -o OUTPUT_DIR -a UnMarker -e StableSignature`

### WatermarkRemover-AI
- **Repo**: https://github.com/D-Ogi/WatermarkRemover-AI
- **Method**: Florence-2 detection + LaMA inpainting (visible watermarks primarily)
- **GUI**: PyWebview interface

### Watermark-Removal-Pytorch
- **Repo**: https://github.com/braindotai/Watermark-Removal-Pytorch
- **Method**: Deep Image Prior (DIP)
- **Best for**: DwtDctSVD, rivaGAN, SSL watermarks

### SynthID-Bypass
- **Repo**: https://github.com/00quebec/Synthid-Bypass
- **Target**: Google SynthID specifically

### Awesome-GenAI-Watermarking
- **Repo**: https://github.com/and-mill/Awesome-GenAI-Watermarking
- **Purpose**: Curated list of watermarking schemes and attacks

---

## 14. Implementation Strategy for RealRestore

### Approach 1: Leverage Existing Restoration Pipeline (Recommended)
Since RealRestore already runs images through diffusion-based restoration, watermark removal is likely a natural side effect. The implementation should:

1. **Verify existing removal**: Test restored images against watermark detectors (SynthID, StegaStamp decoder) to confirm the restoration pipeline already removes watermarks
2. **Tune noise parameters**: If restoration doesn't fully remove watermarks, increase denoising strength slightly
3. **Add metadata stripping**: Always strip EXIF/C2PA metadata post-restoration (trivial to implement)

### Approach 2: Dedicated Pre-Processing Module
If the restoration pipeline alone is insufficient:

```
Input Image
    │
    ├─ 1. Metadata Detection & Stripping
    │     - Check/strip C2PA, EXIF AI tags, PNG chunks
    │
    ├─ 2. Watermark Detection (Optional)
    │     - FFT anomaly scan
    │     - Noise residual analysis
    │     - Known decoder probing
    │
    ├─ 3. Watermark Removal
    │     - Primary: Diffusion regeneration (VAE encode → noise → denoise → decode)
    │       * Strength 0.04-0.15 for minimal quality impact
    │       * Use existing model weights (SD 1.5/SDXL already loaded for restoration)
    │     - Fallback: Frequency-domain suppression for specific known watermarks
    │     - Optional: DWT/DCT coefficient thresholding for legacy watermarks
    │
    └─ 4. Quality Restoration
          - CIELAB color matching against original
          - SSIM-guided parameter tuning
          - Standard restoration pipeline continues
```

### Approach 3: Integrated Pipeline with Auto-Detection
Most sophisticated approach — detect watermark type and apply targeted removal:

```python
class WatermarkHandler:
    def __init__(self, device="mps"):
        self.device = device
        self.vae = None  # Lazy-load from restoration pipeline

    def detect_and_remove(self, image: Image) -> Image:
        # 1. Strip metadata
        image = self.strip_metadata(image)

        # 2. Detect watermark type
        wm_type = self.detect_watermark(image)

        # 3. Apply targeted removal
        if wm_type == "tree_ring":
            image = self.spatial_translation(image, shift=7)
        elif wm_type == "synthid":
            image = self.frequency_suppression(image, self.SYNTHID_FREQS)
        elif wm_type in ("stegastamp", "stable_signature", "unknown"):
            image = self.diffusion_regeneration(image, strength=0.1)

        # 4. Color restoration
        image = self.cielab_restore(image, original=image)

        return image
```

### Memory Considerations for Apple Silicon (64GB Unified Memory)
- Diffusion regeneration reuses models already loaded for restoration (~4-8 GB)
- FFT/DWT/DCT analysis: negligible memory (<100 MB)
- Metadata operations: negligible
- No additional VRAM needed if sharing the restoration pipeline's diffusion model

### Recommended Implementation Priority
1. **Phase 1**: Metadata stripping (trivial, always include)
2. **Phase 2**: Verify restoration pipeline's natural watermark removal capability
3. **Phase 3**: Add diffusion regeneration pre-pass if Phase 2 is insufficient
4. **Phase 4**: Optional frequency-domain analysis for watermark detection/reporting

### Key Academic References
- "Invisible Watermarks: Attacks and Robustness" (arXiv 2412.12511, Dec 2024)
- "NeurIPS 2024 Invisible Watermark Removal Challenge — First Place" (arXiv 2508.21072)
- "Vanishing Watermarks: Diffusion-Based Image Editing Undermines Robust Invisible Watermarking" (arXiv 2602.20680, Feb 2026)
- "UnMarker: A Universal Attack on Defensive Image Watermarking" (IEEE S&P 2025)
- "A Crack in the Bark: Leveraging Public Knowledge to Remove Tree-Ring Watermarks" (USENIX Security 2025)
- "Invisible Image Watermarks Are Provably Removable Using Generative AI" (NeurIPS 2024)
- "Stable Signature is Unstable" (ICLR 2025)
- "UWPD: A General Paradigm for Invisible Watermark Detection Agnostic to Embedding Algorithms" (arXiv 2603.06723)
- "Deep Learning for Image Watermarking: A Comprehensive Review" (Sensors, 2025)
- "Removing Watermarks with Partial Regeneration using Semantic Information" (arXiv 2505.08234)
- "Editing Away the Evidence: Diffusion-Based Image Manipulation and Failure Modes of Robust Watermarking" (arXiv 2603.12949, Mar 2026)
