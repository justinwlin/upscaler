#!/usr/bin/env bash
# SUPIR P4 pod validation — prove SUPIR runs headless (no LLaVA) on an A40, measure VRAM + time.
# Run on the pod:  bash supir_validate.sh 2>&1 | tee /workspace/supir.log
set -uo pipefail
cd /workspace

echo "=== [1/6] clone SUPIR ==="
[ -d SUPIR ] || git clone --depth 1 https://github.com/Fanghua-Yu/SUPIR.git
cd SUPIR

echo "=== [2/6] deps (keep base torch; skip the torch-pinned xformers) ==="
python3 - <<'PY'
# strip torch/torchvision/xformers pins from requirements so we don't downgrade the base torch
import re
lines=open("requirements.txt").read().splitlines()
keep=[l for l in lines if l.strip() and not re.match(r'\s*(torch|torchvision|torchaudio|xformers)\b', l.strip(), re.I)]
open("requirements.trimmed.txt","w").write("\n".join(keep)+"\n")
print("kept",len(keep),"of",len(lines),"req lines")
PY
pip install -q -r requirements.trimmed.txt || echo "WARN: some reqs failed"
pip install -q "huggingface_hub[hf_transfer]" accelerate || true

echo "=== [3/6] download weights (NO LLaVA) ==="
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p /workspace/w
python3 - <<'PY'
from huggingface_hub import hf_hub_download, snapshot_download
import os
W="/workspace/w"
# SUPIR ckpt + SDXL base (with 0.9 vae) from camenduru mirror
for repo,fn in [("camenduru/SUPIR","SUPIR-v0Q.ckpt"),
                ("camenduru/SUPIR","sd_xl_base_1.0_0.9vae.safetensors")]:
    p=hf_hub_download(repo_id=repo, filename=fn, local_dir=W)
    print("got",p, os.path.getsize(p)//1_000_000,"MB")
# CLIP bigG (open_clip .bin) — SDXL text encoder 2
p=hf_hub_download(repo_id="laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
                  filename="open_clip_pytorch_model.bin", local_dir=os.path.join(W,"bigG"))
print("got",p, os.path.getsize(p)//1_000_000,"MB")
# CLIP ViT-L (SDXL text encoder 1) — small HF repo, snapshot it
d=snapshot_download(repo_id="openai/clip-vit-large-patch14", local_dir=os.path.join(W,"clip-vit-large-patch14"))
print("clip-L at",d)
PY

echo "=== [4/6] patch CKPT_PTH.py -> local paths, LLaVA off ==="
python3 - <<'PY'
W="/workspace/w"
c=f'''LLAVA_CLIP_PATH = None
LLAVA_MODEL_PATH = None
SDXL_CLIP1_PATH = "{W}/clip-vit-large-patch14"
SDXL_CLIP2_CKPT_PTH = "{W}/bigG/open_clip_pytorch_model.bin"
SDXL_CKPT = "{W}/sd_xl_base_1.0_0.9vae.safetensors"
SUPIR_CKPT = "{W}/SUPIR-v0Q.ckpt"
'''
open("CKPT_PTH.py","w").write(c)
print(c)
PY

echo "=== [5/6] test image ==="
mkdir -p /workspace/in /workspace/out
python3 - <<'PY'
import urllib.request
from PIL import Image
import io
raw=urllib.request.urlopen(urllib.request.Request(
  "https://raw.githubusercontent.com/xinntao/Real-ESRGAN/master/inputs/00003.png",
  headers={"User-Agent":"t/1"}),timeout=60).read()
img=Image.open(io.BytesIO(raw)).convert("RGB")
img.resize((img.width//2,img.height//2)).save("/workspace/in/test.png")
print("test image saved")
PY

echo "=== [6/6] run SUPIR (no LLaVA, tiled vae, half params) ==="
python3 -c "import torch;torch.cuda.reset_peak_memory_stats()" 2>/dev/null || true
/usr/bin/time -v python3 test.py \
  --img_dir /workspace/in --save_dir /workspace/out \
  --SUPIR_sign Q --upscale 4 --no_llava \
  --use_tile_vae --loading_half_params 2>&1 | tail -60
echo "=== OUTPUTS ==="
ls -la /workspace/out/ 2>/dev/null
echo "=== SUPIR_VALIDATE_DONE ==="
