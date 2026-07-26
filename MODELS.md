# MODELS.md — open-source image upscaler landscape (researched 2026-07-25)

Goal: best open-source **photo upscaling** for a personal web tool. The reference test
case is a low-resolution, motion-blurred phone/CCTV-style photo of a person — so
**real-world degradation** (noise, compression, blur) and **face quality** matter far
more than clean-bicubic benchmark scores.

## Recommendation

| Tier | Model(s) | License | Why |
| --- | --- | --- | --- |
| **Default** | **Real-ESRGAN x4plus** + **GFPGAN v1.4** face enhance | Real-ESRGAN **BSD-3**; GFPGAN **Apache-2.0 code** but bundles non-commercial priors (see licensing note) → **fine for personal use, NOT cleanly commercial** | THE battle-tested combo for degraded real photos with faces. Trained on synthetic real-world degradation (not clean bicubic), so it actually fixes noisy/compressed/blurry inputs. Fast (~1–3 s/img on a 4090). Weights total **~585 MB across four files** (see manifest) — this tool caches them via Runpod `--model-reference`. |
| **Alt fast** | **AuraSR-v2** (fal.ai GigaGAN) | Apache-2.0 | Sub-second 4× GAN; shines on cleaner inputs and AI-generated images. No face-specific restoration, 4×-only. Offer as a toggle. Weights ~0.6–1.2 GB load via HF `from_pretrained` → also host-cache or pin (else it downloads every cold start). |
| **Quality (P4, optional)** | **SUPIR** | **Non-commercial** (LICENSE file says MIT but README + issue #51 impose non-commercial; net = non-commercial) | Best-in-class realism on badly degraded shots (SDXL-based diffusion, prompt-guidable). But 12 GB+ VRAM, 10–50× slower, ~12 GB of weights. Fine for personal use license-wise; heavy cold-start/cost — a later "hero shot" tier, not the default. HF-hosted → genuine `--model-reference` candidate. |

## Licensing note (read before assuming "commercial-OK")

- **Real-ESRGAN x4plus** — clean **BSD-3-Clause** (verified against the repo LICENSE). Permissive, commercial-OK.
- **GFPGAN v1.4** — the Python **code is Apache-2.0**, but the shipped model pulls in third-party priors that are **not** commercial-friendly: the **StyleGAN2** facial prior is under NVIDIA's **non-commercial** source license, and DFDNet is **CC-BY-NC-SA 4.0** (both carved out in GFPGAN's own LICENSE). **For this personal, non-commercial tool that's fine** — but do not treat the combo as cleanly commercial.
- **AuraSR-v2 / CodeFormer / SUPIR** — Apache-2.0 / S-Lab-NC / effectively-NC respectively (CodeFormer stays rejected; SUPIR NC is OK for personal use).

## Weight manifest (all four required, ~585 MB) — every cold start must find these locally

Real-ESRGAN + GFPGAN do **not** stop at the two obvious `.pth` files: on first inference GFPGAN pulls two more auxiliary weights via `facexlib` from **GitHub release URLs (not HuggingFace)**. On a scale-to-zero worker the ephemeral FS is wiped each cold start, so if these aren't pre-provisioned, **every cold start hits the network** (slow, and a hard dependency on GitHub staying up). All four must be present before inference:

| File | ~Size | Used by | Default source |
| --- | --- | --- | --- |
| `RealESRGAN_x4plus.pth` | 64 MB | Real-ESRGAN backbone | GitHub release |
| `GFPGANv1.4.pth` | 332 MB | GFPGAN face restore | GitHub release |
| `detection_Resnet50_Final.pth` | 104 MB | facexlib face **detection** | GitHub release |
| `parsing_parsenet.pth` | 85 MB | facexlib face **parsing** | GitHub release |

**Delivery = Runpod model cache** (the project goal). Because `--model-reference` only speaks HuggingFace, the plan is to mirror all four files into **one HF repo** and attach it with `--model-reference`; Runpod host-caches it and the worker loads every file **offline** from `/runpod-volume/huggingface-cache/hub/models--…/snapshots/<hash>/` (handler sets `HF_HUB_OFFLINE=1` and points RealESRGANer / GFPGANer / facexlib at those exact paths). Engineering note: for ~585 MB this is roughly a wash vs baking into the image (the torch+CUDA base is 5–8 GB either way) — model-reference is used here per the project directive and keeps weights versioned/out of the image; the decisive win is that it forces us to pre-place the facexlib files so **nothing downloads at runtime**. A P1 check runs one inference with networking disabled to prove it.

## Build gotcha — basicsr breaks on modern torchvision (both reviewers flagged it)

GFPGAN/Real-ESRGAN depend on **`basicsr`**, effectively unmaintained since ~2022–2023. Its `degradations.py` imports `torchvision.transforms.functional_tensor`, **removed in torchvision ≥ 0.17**, so a fresh install throws `ModuleNotFoundError: No module named 'torchvision.transforms.functional_tensor'` on import. Fix = pin compatible torch/torchvision **or** patch/shim the import (`functional_tensor` → `functional`). This is validated + pinned during P1 before any serverless image is built.

## Evaluated and passed over

| Model | License | Why not |
| --- | --- | --- |
| CodeFormer | S-Lab 1.0 (**non-commercial**) | Great face restorer, but GFPGAN covers the need with Apache-2.0. |
| HAT / DAT | permissive | Benchmark SR champions on *clean bicubic* downscales; underperform GAN/diffusion models on real-world degradation like our test case. |
| SwinIR | Apache-2.0 | 2021-era; superseded by the above. |
| 4x-UltraSharp & community ESRGAN forks | varies (often CC-BY-NC-SA) | Fast, but license patchwork; Real-ESRGAN official weights are BSD-3. |
| FlashVSR / SeedVR2 | — | Video upscalers; out of scope (images). |
| Thera / ResShift / InvSR | research | Interesting papers, no production edge over the picks. |

**Watch list (better on hard cases, but not permissive / heavy — revisit alongside the P4 SUPIR tier):** diffusion face restorers — **OSDFace** (CVPR 2025), **NTIRE 2025/2026 Real-World Face Restoration** challenge winners, **LAFR**, **InstantRestore**. They can beat GFPGAN on severely degraded faces but are research code, mostly **SD/SDXL-derived (CreativeML OpenRAIL-M, use-restricted — not truly permissive)**, heavier VRAM, slower cold starts. Same trade-off SUPIR already represents.

## Face caveat (important for this use case)

GFPGAN (and every face restorer) **hallucinates plausible detail** — the output face is
a reconstruction, not a recovery. Perfect for making a photo look good; never treat it
as forensic evidence of what someone actually looks like. The UI keeps face-enhance a
visible toggle for exactly this reason.

## Hardware / serving

- Real-ESRGAN + GFPGAN: fits easily in 24 GB (RTX 4090 / ADA_24). **Tiling is not automatic** —
  Real-ESRGAN only tiles when `tile > 0` (default 0 = whole image = OOM on large inputs); set a
  default tile (e.g. 400) + `tile_pad`, and catch OOM to retry with a smaller tile. GFPGAN's
  `bg_upsampler` must be wired to the RealESRGANer or the background/no-face regions aren't
  upscaled at all. ~1–3 s per 4× upscale once tuned (measured in P1).
- Weights (~585 MB, four files) delivered via **Runpod model cache** (`--model-reference` → one
  mirrored HF repo), loaded offline from the host cache. See the weight manifest above.
- AuraSR-v2: `pip install aura-sr`, weights from HF `fal/AuraSR-v2` (Apache-2.0) — host-cache/pin
  so it doesn't download per cold start. 4×-only, no face restoration (so `scale:2` + aurasr and
  `face_enhance` + aurasr are rejected as `bad_params`).
- SUPIR (if/when): needs SDXL base + SUPIR ckpt (~12 GB) → that tier WOULD use
  `--model-reference` (HF-hosted), 48 GB-class GPU recommended.

## Sources

- [Botmonster: Real-ESRGAN vs Topaz vs SUPIR local comparison](https://botmonster.com/ai/local-ai-image-upscaling-real-esrgan-topaz-supir/)
- [SeedVR2 blog: Topaz vs FlashVSR vs SeedVR2 vs SUPIR vs AuraSR 2026](https://seedvr2.net/blog/comparisons/best-ai-upscaler-comparison-topaz-flashvsr-seedvr2-supir-2026)
- [SUPIR GitHub (license: non-commercial, permission required)](https://github.com/Fanghua-Yu/SUPIR) · [SUPIR LICENSE text](https://huggingface.co/spaces/Upscaler/SUPIR/blob/fae2f4559b8dc6e2ae95973704b7124745162cd7/LICENSE)
- [AuraSR-v2 on HF (Apache-2.0)](https://huggingface.co/fal/AuraSR-v2) · [fal blog: AuraSR V2](https://blog.fal.ai/aurasr-v2/)
- [MyImageUpscaler: ESRGAN vs SwinIR model comparison](https://myimageupscaler.com/comparisons-expanded/ai-models-comparison)
- [VideoProc: open-source image upscaler hands-on](https://www.videoproc.com/resource/open-source-image-upscaler.htm)
