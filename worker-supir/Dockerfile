# SUPIR "max quality" serverless worker (SDXL diffusion restoration).
# Code + deps baked; the ~18GB weights live on the mounted network volume
# (/runpod-volume/supir-weights, staged once). Runs on a 24GB RTX 4090 (tiled VAE + half).
# torch 2.2 / cu12.1 base -> deploy with --min-cuda-version 12.1.
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive MODE_TO_RUN=serverless PYTHONUNBUFFERED=1
WORKDIR /app

RUN git clone --depth 1 https://github.com/Fanghua-Yu/SUPIR.git /app/SUPIR

# Headless runtime deps (skip the gradio/fastapi web cluster that breaks the resolver).
# Pins learned in pod validation: huggingface_hub 0.20.3 (diffusers 0.16 needs cached_download),
# xformers 0.0.24 (tiled-VAE path hard-requires xformers; matches torch 2.2.0).
RUN pip install --no-cache-dir \
      omegaconf==2.3.0 einops==0.7.0 einops-exts==0.0.4 open-clip-torch==2.17.1 \
      pytorch-lightning==2.1.2 transformers==4.28.1 tokenizers==0.13.3 kornia==0.6.9 \
      k-diffusion==0.1.1.post1 diffusers==0.16.1 timm==0.9.8 opencv-python-headless==4.7.0.72 \
      scipy==1.9.1 facexlib==0.3.0 openai-clip==1.0.1 webdataset==0.2.48 \
      sentencepiece==0.1.98 matplotlib==3.7.1 pandas==2.0.1 accelerate==0.18.0 \
      huggingface_hub==0.20.3 xformers==0.0.24 runpod

# Point SUPIR at the volume-staged weights + disable LLaVA (fixed prompt, no captioning).
RUN python3 - <<'PY'
W="/runpod-volume/supir-weights"
open("/app/SUPIR/CKPT_PTH.py","w").write(
 f'LLAVA_CLIP_PATH = None\nLLAVA_MODEL_PATH = None\n'
 f'SDXL_CLIP1_PATH = "{W}/clip-vit-large-patch14"\n'
 f'SDXL_CLIP2_CKPT_PTH = "{W}/bigG/open_clip_pytorch_model.bin"\n'
 f'SDXL_CKPT = "{W}/sd_xl_base_1.0_0.9vae.safetensors"\n'
 f'SUPIR_CKPT = "{W}/SUPIR-v0Q.ckpt"\n')
p="/app/SUPIR/options/SUPIR_v0.yaml"; s=open(p).read()
s=s.replace("/opt/data/private/AIGC_pretrain/SDXL_cache/sd_xl_base_1.0_0.9vae.safetensors", f"{W}/sd_xl_base_1.0_0.9vae.safetensors")
s=s.replace("/opt/data/private/AIGC_pretrain/SUPIR_cache/SUPIR-v0Q.ckpt", f"{W}/SUPIR-v0Q.ckpt")
s=s.replace("/opt/data/private/AIGC_pretrain/SUPIR_cache/SUPIR-v0F.ckpt", f"{W}/SUPIR-v0Q.ckpt")
open(p,"w").write(s)
print("patched CKPT_PTH.py + SUPIR_v0.yaml")
PY

COPY handler.py /app/handler.py
CMD ["python3", "-u", "/app/handler.py"]
