# MODELS.md — open-source image upscaler landscape (researched 2026-07-25)

Goal: best open-source **photo upscaling** for a personal web tool. The reference test
case is a low-resolution, motion-blurred phone/CCTV-style photo of a person — so
**real-world degradation** (noise, compression, blur) and **face quality** matter far
more than clean-bicubic benchmark scores.

## Recommendation

| Tier | Model(s) | License | Why |
| --- | --- | --- | --- |
| **Default** | **Real-ESRGAN x4plus** + **GFPGAN v1.4** face enhance | BSD-3 / Apache-2.0 | THE battle-tested combo for degraded real photos with faces. Trained on synthetic real-world degradation (not clean bicubic), so it actually fixes noisy/compressed/blurry inputs. Fast (~1–3 s/img on a 4090), tiny weights (64 MB + 350 MB — bake into the image, no model cache needed). |
| **Alt fast** | **AuraSR-v2** (fal.ai GigaGAN) | Apache-2.0 | Sub-second 4× GAN; shines on cleaner inputs and AI-generated images. No face-specific restoration. Offer as a toggle. |
| **Quality (P4, optional)** | **SUPIR** | **Non-commercial** (written permission for commercial) | Best-in-class realism on badly degraded shots (SDXL-based diffusion, prompt-guidable). But 12 GB+ VRAM, 10–50× slower, ~12 GB of weights. Fine for personal use license-wise; heavy cold-start/cost — a later "hero shot" tier, not the default. |

## Evaluated and passed over

| Model | License | Why not |
| --- | --- | --- |
| CodeFormer | S-Lab 1.0 (**non-commercial**) | Great face restorer, but GFPGAN covers the need with Apache-2.0. |
| HAT / DAT | permissive | Benchmark SR champions on *clean bicubic* downscales; underperform GAN/diffusion models on real-world degradation like our test case. |
| SwinIR | Apache-2.0 | 2021-era; superseded by the above. |
| 4x-UltraSharp & community ESRGAN forks | varies (often CC-BY-NC-SA) | Fast, but license patchwork; Real-ESRGAN official weights are BSD-3. |
| FlashVSR / SeedVR2 | — | Video upscalers; out of scope (images). |
| Thera / ResShift / InvSR | research | Interesting papers, no production edge over the picks. |

## Face caveat (important for this use case)

GFPGAN (and every face restorer) **hallucinates plausible detail** — the output face is
a reconstruction, not a recovery. Perfect for making a photo look good; never treat it
as forensic evidence of what someone actually looks like. The UI keeps face-enhance a
visible toggle for exactly this reason.

## Hardware / serving

- Real-ESRGAN + GFPGAN: fits easily in 24 GB (RTX 4090 / ADA_24). Tiles internally, so
  large inputs work. ~1–3 s per 4× upscale.
- Weights are small enough to **bake into the Docker image** (~400 MB total) — no
  model-reference/host-cache dance, no volume for models, fast cold starts.
- AuraSR-v2: `pip install aura-sr`, weights from HF `fal/AuraSR-v2` (Apache-2.0).
- SUPIR (if/when): needs SDXL base + SUPIR ckpt (~12 GB) → that tier WOULD use
  `--model-reference`, 48 GB-class GPU recommended.

## Sources

- [Botmonster: Real-ESRGAN vs Topaz vs SUPIR local comparison](https://botmonster.com/ai/local-ai-image-upscaling-real-esrgan-topaz-supir/)
- [SeedVR2 blog: Topaz vs FlashVSR vs SeedVR2 vs SUPIR vs AuraSR 2026](https://seedvr2.net/blog/comparisons/best-ai-upscaler-comparison-topaz-flashvsr-seedvr2-supir-2026)
- [SUPIR GitHub (license: non-commercial, permission required)](https://github.com/Fanghua-Yu/SUPIR) · [SUPIR LICENSE text](https://huggingface.co/spaces/Upscaler/SUPIR/blob/fae2f4559b8dc6e2ae95973704b7124745162cd7/LICENSE)
- [AuraSR-v2 on HF (Apache-2.0)](https://huggingface.co/fal/AuraSR-v2) · [fal blog: AuraSR V2](https://blog.fal.ai/aurasr-v2/)
- [MyImageUpscaler: ESRGAN vs SwinIR model comparison](https://myimageupscaler.com/comparisons-expanded/ai-models-comparison)
- [VideoProc: open-source image upscaler hands-on](https://www.videoproc.com/resource/open-source-image-upscaler.htm)
