#!/usr/bin/env bash
# P2 deploy — build+push the worker image, then create a Runpod serverless endpoint that
# gets its weights from the Runpod model cache (--model-reference, verified public mirrors).
#
# Prereqs: docker buildx + Docker Hub login (justinrunpod namespace); RUNPOD_API_KEY resolvable.
# Nothing here contains secrets. Endpoint id is printed for you to paste into the site.
set -euo pipefail

IMAGE="justinrunpod/upscale:v1"
VOLUME_ID="z3nw2gsth6"          # vo-results, US-IL-1 (reused; results land in upscale-out/)
GPU="NVIDIA GeForce RTX 4090"

# Weights via model cache — two public HF repos, sha256-verified == official, commit-pinned.
REALESRGAN_REF="https://huggingface.co/amd/realesrgan-x4plus:bda69abcaf525425b371622349e975245ae090c2"
GFPGAN_REF="https://huggingface.co/gmk123/GFPGAN:e881fbc251fdf2a4f133ad8277dd5dadbd1c541a"

# 1) build + push the weight-free worker image (amd64)
docker buildx build --platform linux/amd64 -t "$IMAGE" --push worker/

# 2) create the serverless endpoint.
#    --min-cuda-version 12.4 matches the cu12.4 base (avoids older-driver crash-loop).
#    --model-reference is repeatable; the volume pins the endpoint to US-IL-1.
runpodctl serverless create \
  --name upscale-test \
  --image "$IMAGE" \
  --gpu-id "$GPU" \
  --min-cuda-version 12.4 \
  --network-volume-id "$VOLUME_ID" \
  --model-reference "$REALESRGAN_REF" \
  --model-reference "$GFPGAN_REF" \
  --workers-min 0 --workers-max 1 \
  --container-disk-in-gb 25 \
  --flashboot
# -> capture the endpoint id it prints; poll /health before the first real job.
