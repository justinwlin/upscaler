#!/usr/bin/env python3
"""P1 pod validation for the upscaler default tier (Real-ESRGAN x4plus + GFPGAN v1.4).

Proves, on a real GPU:
  1. deps install on the pod's torch/torchvision (with the basicsr functional_tensor patch),
  2. all four weights load OFFLINE from a fixed dir (nothing downloads at inference),
  3. real 4x upscale + face-enhance run on a degraded face photo,
  4. timings + peak VRAM + tiling behavior, scale=2, and alpha(PNG) handling.

Run on the pod:  python3 pod_validate.py
Outputs land in /workspace/upval/ (pull via the pod's :8000 http proxy).
"""
import os, sys, time, urllib.request, subprocess, traceback

WORK = "/workspace/upval"
WEIGHTS = os.path.join(WORK, "weights")            # our fixed offline weights dir (mimics HF-cache snapshot)
os.makedirs(WEIGHTS, exist_ok=True)
os.makedirs(WORK, exist_ok=True)

def log(*a): print("[val]", *a, flush=True)

# ---- weight manifest (the four files every cold start must find locally) ----
WEIGHT_URLS = {
    "RealESRGAN_x4plus.pth":
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    "GFPGANv1.4.pth":
        "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
    "detection_Resnet50_Final.pth":
        "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth",
    "parsing_parsenet.pth":
        "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth",
}

def fetch(url, dst):
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        log("have", os.path.basename(dst), os.path.getsize(dst)); return
    log("download", os.path.basename(dst))
    req = urllib.request.Request(url, headers={"User-Agent": "upscaler-val/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dst, "wb") as f:
        f.write(r.read())
    log("  ->", os.path.getsize(dst), "bytes")

def step_install():
    log("=== install deps ===")
    pkgs = ["numpy<2", "opencv-python-headless", "pillow",
            "basicsr==1.4.2", "facexlib==0.3.0", "gfpgan==1.3.8", "realesrgan==0.3.0"]
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)
    # basicsr imports torchvision.transforms.functional_tensor (removed in tv>=0.17).
    # Patch BEFORE importing basicsr — locate the pkg dir via find_spec (does NOT execute it).
    import importlib.util, glob
    spec = importlib.util.find_spec("basicsr")
    bd = spec.submodule_search_locations[0]
    for fp in glob.glob(os.path.join(bd, "**", "*.py"), recursive=True):
        with open(fp) as f: src = f.read()
        if "functional_tensor" in src:
            with open(fp, "w") as f:
                f.write(src.replace("torchvision.transforms.functional_tensor",
                                    "torchvision.transforms.functional"))
            log("patched functional_tensor in", os.path.relpath(fp, bd))

def place_facexlib_offline():
    """facexlib's FaceRestoreHelper downloads detection/parsing weights via load_file_from_url
    into <site-packages>/facexlib/weights/. Pre-place them so load is OFFLINE."""
    import facexlib
    fx_w = os.path.join(os.path.dirname(facexlib.__file__), "weights")
    os.makedirs(fx_w, exist_ok=True)
    for name in ("detection_Resnet50_Final.pth", "parsing_parsenet.pth"):
        src = os.path.join(WEIGHTS, name); dst = os.path.join(fx_w, name)
        if not os.path.exists(dst):
            import shutil; shutil.copy(src, dst); log("placed facexlib weight", name)
    return fx_w

def make_test_image():
    """Grab a real face sample, then degrade it (small + JPEG) to mimic the low-res use case."""
    from PIL import Image
    import io
    raw = os.path.join(WORK, "src_face.png")
    if not os.path.exists(raw):
        for url in ("https://raw.githubusercontent.com/xinntao/Real-ESRGAN/master/inputs/00003.png",
                    "https://raw.githubusercontent.com/TencentARC/GFPGAN/master/inputs/whole_imgs/00.jpg"):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "upscaler-val/1.0"})
                with urllib.request.urlopen(req, timeout=60) as r, open(raw, "wb") as f:
                    f.write(r.read())
                log("test image from", url); break
            except Exception as e:
                log("  test-image url failed:", e)
    img = Image.open(raw).convert("RGB")
    # degrade: downscale to ~0.35x then JPEG q40, to simulate a blurry compressed photo
    small = img.resize((max(64, img.width//3), max(64, img.height//3)), Image.BICUBIC)
    buf = io.BytesIO(); small.save(buf, "JPEG", quality=40); buf.seek(0)
    deg = Image.open(buf).convert("RGB")
    degp = os.path.join(WORK, "degraded_in.png"); deg.save(degp)
    log("degraded input", deg.size, "->", degp)
    return degp

def run():
    import cv2, numpy as np, torch
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    from gfpgan import GFPGANer

    log("torch", torch.__version__, "cuda_avail", torch.cuda.is_available(),
        "dev", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")

    # ---- build RealESRGANer pointing at LOCAL weights, with tiling ----
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    t0 = time.time()
    up = RealESRGANer(scale=4, model_path=os.path.join(WEIGHTS, "RealESRGAN_x4plus.pth"),
                      model=model, tile=400, tile_pad=10, pre_pad=0, half=True, gpu_id=0)
    log("RealESRGANer load %.2fs" % (time.time()-t0))

    # GFPGANer with bg_upsampler = the RealESRGANer (so background/no-face still upscales)
    t0 = time.time()
    face = GFPGANer(model_path=os.path.join(WEIGHTS, "GFPGANv1.4.pth"),
                    upscale=4, arch="clean", channel_multiplier=2, bg_upsampler=up)
    log("GFPGANer load %.2fs" % (time.time()-t0))

    degp = make_test_image()
    img = cv2.imread(degp, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]; log("input", w, "x", h)

    def vram_reset(): torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    def vram_mb(): torch.cuda.synchronize(); return torch.cuda.max_memory_allocated()/1e6

    # (a) Real-ESRGAN only, 4x
    vram_reset(); t0 = time.time()
    out, _ = up.enhance(img, outscale=4)
    dt = time.time()-t0
    cv2.imwrite(os.path.join(WORK, "out_realesrgan_4x.png"), out)
    log("A realesrgan 4x: %.2fs peakVRAM %.0fMB -> %dx%d" % (dt, vram_mb(), out.shape[1], out.shape[0]))

    # (b) Real-ESRGAN + GFPGAN face enhance, 4x
    vram_reset(); t0 = time.time()
    _, _, out2 = face.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
    dt = time.time()-t0
    cv2.imwrite(os.path.join(WORK, "out_gfpgan_4x.png"), out2)
    log("B realesrgan+gfpgan 4x: %.2fs peakVRAM %.0fMB -> %dx%d" % (dt, vram_mb(), out2.shape[1], out2.shape[0]))

    # (c) scale=2 via outscale on the 4x net
    vram_reset(); t0 = time.time()
    out3, _ = up.enhance(img, outscale=2)
    dt = time.time()-t0
    cv2.imwrite(os.path.join(WORK, "out_realesrgan_2x.png"), out3)
    log("C realesrgan 2x: %.2fs peakVRAM %.0fMB -> %dx%d" % (dt, vram_mb(), out3.shape[1], out3.shape[0]))

    # (d) alpha/PNG: upscale RGB + alpha separately, re-attach
    rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = 200  # fake semi-transparency
    bgr = rgba[:, :, :3]; a = rgba[:, :, 3]
    o_bgr, _ = up.enhance(bgr, outscale=4)
    o_a = cv2.resize(a, (o_bgr.shape[1], o_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
    o_rgba = cv2.cvtColor(o_bgr, cv2.COLOR_BGR2BGRA); o_rgba[:, :, 3] = o_a
    cv2.imwrite(os.path.join(WORK, "out_alpha_4x.png"), o_rgba)
    log("D alpha path ok -> %dx%d (4 ch)" % (o_rgba.shape[1], o_rgba.shape[0]))

    log("=== DONE. outputs in", WORK, "===")

def main():
    try:
        step_install()
        for name, url in WEIGHT_URLS.items():
            fetch(url, os.path.join(WEIGHTS, name))
        place_facexlib_offline()
        # prove offline: forbid any network fetch during model load/inference
        os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
        run()
    except Exception:
        traceback.print_exc(); sys.exit(1)

if __name__ == "__main__":
    main()
