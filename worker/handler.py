#!/usr/bin/env python3
"""Runpod serverless handler for the upscaler.

Modes: upscale | list | fetch. Default tier = Real-ESRGAN x4plus + GFPGAN v1.4.

Weights are delivered by Runpod's model cache: `--model-reference` points the endpoint at a
mirrored HF repo, so the four .pth files land under
/runpod-volume/huggingface-cache/hub/models--<org>--<name>/snapshots/<hash>/ and load OFFLINE.
The facexlib detection/parsing weights are copied into facexlib's package dir at startup so
nothing downloads at inference time.

Dual-mode: MODE_TO_RUN=serverless runs runpod.serverless.start; anything else exposes
handler() for local pod testing (see worker/localtest.py).
"""
import os, io, time, json, base64, hashlib, glob, shutil, traceback

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import cv2
from PIL import Image, ImageOps

# ---------------- config ----------------
VOL = os.environ.get("VOLUME_DIR", "/runpod-volume")
OUT_ROOT = os.path.join(VOL, "upscale-out")
HF_CACHE = os.path.join(VOL, "huggingface-cache", "hub")
WEIGHTS_DIR_ENV = os.environ.get("WEIGHTS_DIR")          # explicit override (pod testing)
RETENTION_S = 48 * 3600
MAX_IN_MP = float(os.environ.get("MAX_IN_MP", "16"))     # decompression-bomb / OOM guard
INLINE_MAX_B64 = int(os.environ.get("INLINE_MAX_B64", str(2_000_000)))  # ~2MB encoded inline cap
DEFAULT_TILE = int(os.environ.get("TILE", "400"))
WEIGHT_FILES = ("RealESRGAN_x4plus.pth", "GFPGANv1.4.pth",
                "detection_Resnet50_Final.pth", "parsing_parsenet.pth")

_UP = None      # RealESRGANer
_FACE = None    # GFPGANer
_AURA = None    # AuraSR (lazy, optional)
_WEIGHTS = None


class UpErr(Exception):
    def __init__(self, code, message):
        super().__init__(message); self.code = code; self.message = message


# ---------------- weight resolution (offline) ----------------
def resolve_weights():
    """Find the dir holding all four .pth files: explicit env, or the HF model-cache snapshot."""
    global _WEIGHTS
    if _WEIGHTS:
        return _WEIGHTS
    candidates = []
    if WEIGHTS_DIR_ENV:
        candidates.append(WEIGHTS_DIR_ENV)
    # HF cache snapshots (model-reference delivery): pick any snapshot containing all four files
    for snap in glob.glob(os.path.join(HF_CACHE, "models--*", "snapshots", "*")):
        candidates.append(snap)
    for d in candidates:
        if all(os.path.exists(os.path.join(d, f)) for f in WEIGHT_FILES):
            _WEIGHTS = d
            _place_facexlib(d)
            return d
    # some HF layouts nest files; do a recursive fallback search under the cache
    found = {}
    for f in WEIGHT_FILES:
        hits = glob.glob(os.path.join(HF_CACHE, "**", f), recursive=True)
        if hits:
            found[f] = hits[0]
    if len(found) == len(WEIGHT_FILES):
        # stage them into one dir so downstream paths are uniform
        staged = os.path.join(VOL, "_weights_staged")
        os.makedirs(staged, exist_ok=True)
        for f, src in found.items():
            dst = os.path.join(staged, f)
            if not os.path.exists(dst):
                os.symlink(src, dst)
        _WEIGHTS = staged
        _place_facexlib(staged)
        return staged
    raise UpErr("model_load_failed",
                "weights not found in model cache; expected %s under %s" % (list(WEIGHT_FILES), HF_CACHE))


def _place_facexlib(weights_dir):
    """facexlib downloads detection/parsing weights into <pkg>/weights via load_file_from_url.
    Pre-place ours so the load is offline."""
    import facexlib
    fx_w = os.path.join(os.path.dirname(facexlib.__file__), "weights")
    os.makedirs(fx_w, exist_ok=True)
    for name in ("detection_Resnet50_Final.pth", "parsing_parsenet.pth"):
        src, dst = os.path.join(weights_dir, name), os.path.join(fx_w, name)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy(src, dst)


def get_realesrgan():
    global _UP
    if _UP is None:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
        w = resolve_weights()
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        _UP = RealESRGANer(scale=4, model_path=os.path.join(w, "RealESRGAN_x4plus.pth"),
                           model=model, tile=DEFAULT_TILE, tile_pad=10, pre_pad=0, half=True, gpu_id=0)
    return _UP


def get_gfpgan():
    global _FACE
    if _FACE is None:
        from gfpgan import GFPGANer
        w = resolve_weights()
        _FACE = GFPGANer(model_path=os.path.join(w, "GFPGANv1.4.pth"),
                         upscale=4, arch="clean", channel_multiplier=2, bg_upsampler=get_realesrgan())
    return _FACE


# ---------------- image io ----------------
def decode_image(image_b64):
    try:
        raw = base64.b64decode(image_b64)
    except Exception:
        raise UpErr("decode_failed", "image_b64 is not valid base64")
    try:
        pil = Image.open(io.BytesIO(raw))
        pil = ImageOps.exif_transpose(pil)           # apply orientation tag (phones)
    except Image.DecompressionBombError:
        raise UpErr("too_many_pixels", "image exceeds the pixel limit")
    except Exception:
        raise UpErr("decode_failed", "could not decode image")
    mp = (pil.width * pil.height) / 1e6
    if mp > MAX_IN_MP:
        raise UpErr("too_many_pixels", "input is %.1f MP > %.0f MP cap" % (mp, MAX_IN_MP))
    has_alpha = pil.mode in ("RGBA", "LA") or (pil.mode == "P" and "transparency" in pil.info)
    if has_alpha:
        pil = pil.convert("RGBA")
        arr = np.array(pil)                          # H,W,4 RGBA
        bgr = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2BGR)
        alpha = arr[:, :, 3]
        return bgr, alpha, (pil.width, pil.height)
    pil = pil.convert("RGB")
    bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return bgr, None, (pil.width, pil.height)


def enhance_with_oom_retry(runner, img, outscale):
    """Real-ESRGAN OOM -> retry once at a smaller tile before giving up."""
    import torch
    try:
        out, _ = runner.enhance(img, outscale=outscale)
        return out
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        old = runner.tile_size
        runner.tile_size = max(128, old // 2)
        try:
            out, _ = runner.enhance(img, outscale=outscale)
            return out
        finally:
            runner.tile_size = old
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            raise UpErr("oom", "ran out of GPU memory even after tiling; use a smaller image")
        raise


# ---------------- modes ----------------
def do_upscale(inp, job=None):
    def prog(msg):
        if job is not None:
            try: __import__("runpod").serverless.progress_update(job, msg)
            except Exception: pass

    scale = int(inp.get("scale", 4))
    if scale not in (2, 4):
        raise UpErr("bad_params", "scale must be 2 or 4")
    model = inp.get("model", "realesrgan")
    face_enhance = bool(inp.get("face_enhance", True))
    out_fmt = inp.get("output", "png").lower()
    if out_fmt not in ("png", "jpg", "jpeg"):
        raise UpErr("bad_params", "output must be png or jpg")
    if model == "aurasr" and (scale == 2 or face_enhance):
        raise UpErr("bad_params", "aurasr is 4x-only with no face restoration")
    if "image_b64" not in inp:
        raise UpErr("bad_params", "image_b64 is required")

    prog("decoding")
    bgr, alpha, (in_w, in_h) = decode_image(inp["image_b64"])

    prog("upscaling")
    import torch
    torch.cuda.reset_peak_memory_stats()
    if model == "aurasr":
        out = aurasr_upscale(bgr)
    elif face_enhance:
        prog("face enhance")
        face = get_gfpgan()
        _, _, out = face.enhance(bgr, has_aligned=False, only_center_face=False, paste_back=True)
        if scale == 2:
            out = cv2.resize(out, (in_w * 2, in_h * 2), interpolation=cv2.INTER_AREA)
    else:
        out = enhance_with_oom_retry(get_realesrgan(), bgr, outscale=scale)

    # re-attach upscaled alpha for PNG (JPEG can't carry alpha)
    if alpha is not None and out_fmt == "png":
        a = cv2.resize(alpha, (out.shape[1], out.shape[0]), interpolation=cv2.INTER_LINEAR)
        out = cv2.cvtColor(out, cv2.COLOR_BGR2BGRA); out[:, :, 3] = a

    prog("saving")
    ext = "png" if out_fmt == "png" else "jpg"
    enc_params = [] if ext == "png" else [cv2.IMWRITE_JPEG_QUALITY, 92]
    ok, buf = cv2.imencode("." + ext, out, enc_params)
    if not ok:
        raise UpErr("model_load_failed", "failed to encode output")
    data = buf.tobytes()
    # thumbnail (<=200KB jpeg, RGB) for instant preview / gallery
    thumb = make_thumb(out)

    out_h, out_w = out.shape[0], out.shape[1]
    sha = hashlib.sha256(data).hexdigest()
    job_dir = save_result(job_id_of(job), data, ext, thumb,
                          dict(in_width=in_w, in_height=in_h, width=out_w, height=out_h,
                               bytes=len(data), sha256=sha, model=model, scale=scale,
                               face_enhance=face_enhance, ext=ext, created=time.time()))
    b64 = base64.b64encode(data).decode()
    inline = b64 if len(b64) <= INLINE_MAX_B64 else None
    return {"job_dir": job_dir, "in_width": in_w, "in_height": in_h,
            "width": out_w, "height": out_h, "bytes": len(data), "sha256": sha,
            "ext": ext, "thumb_b64": thumb, "image_b64": inline}


def make_thumb(bgr, max_side=384):
    h, w = bgr.shape[:2]
    s = min(1.0, max_side / max(h, w))
    small = cv2.resize(bgr[:, :, :3], (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    q = 80
    while q >= 30:
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok and buf.nbytes <= 200_000:
            return base64.b64encode(buf.tobytes()).decode()
        q -= 10
    return base64.b64encode(buf.tobytes()).decode()


def job_id_of(job):
    if isinstance(job, dict) and job.get("id"):
        return str(job["id"])
    return "local-%d" % int(time.time() * 1000)


def save_result(job_id, data, ext, thumb_b64, meta):
    """Atomic: write to a temp dir, then rename into place so readers never see partial files."""
    os.makedirs(OUT_ROOT, exist_ok=True)
    final = os.path.join(OUT_ROOT, job_id)
    tmp = final + ".tmp"
    if os.path.exists(tmp): shutil.rmtree(tmp)
    os.makedirs(tmp)
    with open(os.path.join(tmp, "image." + ext), "wb") as f: f.write(data)
    with open(os.path.join(tmp, "thumb.b64"), "w") as f: f.write(thumb_b64)
    with open(os.path.join(tmp, "meta.json"), "w") as f: json.dump(meta, f)  # marker list keys off
    if os.path.exists(final): shutil.rmtree(final)
    os.rename(tmp, final)
    return job_id


def do_list(inp):
    jobs = []
    if not os.path.isdir(OUT_ROOT):
        return {"jobs": []}
    for name in os.listdir(OUT_ROOT):
        d = os.path.join(OUT_ROOT, name)
        mp = os.path.join(d, "meta.json")
        if name.endswith(".tmp") or not os.path.exists(mp):
            continue
        try:
            meta = json.load(open(mp))
            tb = open(os.path.join(d, "thumb.b64")).read()
        except Exception:
            continue
        jobs.append({"job_dir": name, "created": meta.get("created"),
                     "in_width": meta.get("in_width"), "in_height": meta.get("in_height"),
                     "width": meta.get("width"), "height": meta.get("height"),
                     "bytes": meta.get("bytes"), "sha256": meta.get("sha256"),
                     "ext": meta.get("ext", "png"), "thumb_b64": tb})
    jobs.sort(key=lambda j: j.get("created") or 0, reverse=True)
    return {"jobs": jobs}


def do_fetch(inp):
    job_dir = inp.get("job_dir")
    if not job_dir or "/" in job_dir or job_dir.startswith("."):
        raise UpErr("bad_params", "invalid job_dir")
    d = os.path.join(OUT_ROOT, job_dir)
    if not os.path.isdir(d) or not os.path.exists(os.path.join(d, "meta.json")):
        raise UpErr("not_found", "job_dir not found (may have been purged)")
    meta = json.load(open(os.path.join(d, "meta.json")))
    path = os.path.join(d, "image." + meta.get("ext", "png"))
    total = os.path.getsize(path)
    offset = int(inp.get("offset", 0))
    length = int(inp.get("length", 1_572_864))
    if offset < 0 or offset > total:
        raise UpErr("bad_params", "offset out of range")
    n = max(0, min(length, total - offset))
    with open(path, "rb") as f:
        f.seek(offset); chunk = f.read(n)
    return {"data_b64": base64.b64encode(chunk).decode(), "offset": offset,
            "bytes": n, "total": total, "eof": (offset + n) >= total}


def purge_old():
    if not os.path.isdir(OUT_ROOT):
        return
    now = time.time()
    for name in os.listdir(OUT_ROOT):
        d = os.path.join(OUT_ROOT, name)
        try:
            if now - os.path.getmtime(d) > RETENTION_S:
                shutil.rmtree(d, ignore_errors=True)
        except Exception:
            pass


def aurasr_upscale(bgr):
    global _AURA
    if _AURA is None:
        from aura_sr import AuraSR
        _AURA = AuraSR.from_pretrained("fal/AuraSR-v2")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    out = _AURA.upscale_4x_overlapped(Image.fromarray(rgb))
    return cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)


# ---------------- entrypoint ----------------
def handler(job):
    try:
        inp = (job or {}).get("input", {}) if isinstance(job, dict) else {}
        mode = inp.get("mode", "upscale")
        try:
            purge_old()
        except Exception:
            pass
        if mode == "upscale":
            return do_upscale(inp, job=job)
        if mode == "list":
            return do_list(inp)
        if mode == "fetch":
            return do_fetch(inp)
        raise UpErr("bad_mode", "unknown mode: %s" % mode)
    except UpErr as e:
        return {"error": {"code": e.code, "message": e.message}}
    except Exception as e:
        traceback.print_exc()
        return {"error": {"code": "internal", "message": str(e)}}


if __name__ == "__main__" and os.environ.get("MODE_TO_RUN", "serverless") == "serverless":
    import runpod
    runpod.serverless.start({"handler": handler})
