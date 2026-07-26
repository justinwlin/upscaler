# Upscaler serverless worker — Real-ESRGAN x4plus + GFPGAN v1.4.
# Weights are NOT baked: they arrive via Runpod's model cache (--model-reference -> HF repo),
# loaded offline from /runpod-volume/huggingface-cache/hub/. Keep the image weight-free.
#
# cu12.4 base (matches the validated pod: torch 2.4.1+cu124). Deploy the endpoint with
# --min-cuda-version 12.4 so it never lands on an older-driver host (empty-log crash-loop).
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    MODE_TO_RUN=serverless \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# basicsr imports torchvision.transforms.functional_tensor (removed in torchvision >= 0.17).
# Patch it to .functional at build time or GFPGAN/Real-ESRGAN won't import.
RUN python3 - <<'PY'
import importlib.util, glob, os
bd = importlib.util.find_spec("basicsr").submodule_search_locations[0]
for fp in glob.glob(os.path.join(bd, "**", "*.py"), recursive=True):
    s = open(fp).read()
    if "functional_tensor" in s:
        open(fp, "w").write(s.replace("torchvision.transforms.functional_tensor",
                                      "torchvision.transforms.functional"))
        print("patched", fp)
PY

COPY handler.py /app/handler.py

CMD ["python3", "-u", "handler.py"]
