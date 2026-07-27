#!/usr/bin/env bash
# SUPIR headless re-run: install ONLY the deps test.py needs (skip the gradio/fastapi web cluster
# that causes the resolver conflict), monitor peak VRAM, time one 4x inference.
set -uo pipefail
cd /workspace/SUPIR

echo "=== install headless deps ==="
pip install -q \
  omegaconf==2.3.0 einops==0.7.0 einops-exts==0.0.4 open-clip-torch==2.17.1 \
  pytorch-lightning==2.1.2 transformers==4.28.1 tokenizers==0.13.3 kornia==0.6.9 \
  k-diffusion==0.1.1.post1 diffusers==0.16.1 timm==0.9.8 opencv-python==4.7.0.72 \
  scipy==1.9.1 facexlib==0.3.0 openai-clip==1.0.1 webdataset==0.2.48 \
  sentencepiece==0.1.98 matplotlib==3.7.1 pandas==2.0.1 accelerate==0.18.0 2>&1 | tail -6
echo "install exit: done"

echo "=== VRAM monitor + run ==="
( for i in $(seq 1 900); do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 2; done > /workspace/vram.log 2>/dev/null ) &
MON=$!
START=$(date +%s)
python3 test.py --img_dir /workspace/in --save_dir /workspace/out \
  --SUPIR_sign Q --upscale 4 --no_llava --use_tile_vae --loading_half_params 2>&1 | tail -50
RC=$?
END=$(date +%s)
kill $MON 2>/dev/null
echo "EXIT_CODE $RC"
echo "ELAPSED $((END-START))s"
echo "PEAK_VRAM_MB $(sort -n /workspace/vram.log 2>/dev/null | tail -1)"
echo "=== OUTPUTS ==="; ls -la /workspace/out/
echo "=== SUPIR_RUN2_DONE ==="
