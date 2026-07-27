#!/usr/bin/env python3
"""Runpod serverless handler for the SUPIR "max quality" tier (SDXL diffusion restoration).

Separate endpoint from the fast Real-ESRGAN tier. Same request/response contract
(modes upscale|list|fetch, error envelope {"ok":false,"code","message"}) so the same
frontend drives both — it just points SUPIR jobs at this endpoint.

Weights (~18GB) live on the mounted network volume at /runpod-volume/supir-weights/ (staged
once); the SUPIR repo code + deps are baked into the image. LLaVA is skipped (fixed prompt).
Runs on a 24GB RTX 4090 with tiled VAE + half params (validated ~12GB peak).
"""
import os, io, time, json, base64, hashlib, shutil, traceback

VOL = os.environ.get("VOLUME_DIR", "/runpod-volume")
WEIGHTS = os.environ.get("SUPIR_WEIGHTS", os.path.join(VOL, "supir-weights"))
OUT_ROOT = os.path.join(VOL, "supir-out")
RETENTION_S = 48 * 3600
MAX_IN_MP = float(os.environ.get("MAX_IN_MP", "1.5"))   # SUPIR input cap (it upscales small imgs)
EDM_STEPS = int(os.environ.get("EDM_STEPS", "50"))
INLINE_MAX_B64 = int(os.environ.get("INLINE_MAX_B64", str(2_000_000)))

import numpy as np
from PIL import Image, ImageOps

_MODEL = None


class UpErr(Exception):
    def __init__(self, code, message):
        super().__init__(message); self.code = code; self.message = message


def get_model():
    """Cold-start: build SUPIR once, mirror test.py's setup (half params + tiled VAE)."""
    global _MODEL
    if _MODEL is None:
        import sys, torch
        if "/app/SUPIR" not in sys.path:
            sys.path.insert(0, "/app/SUPIR")   # repo root, so `import SUPIR` resolves
        os.chdir("/app/SUPIR")   # config uses relative paths (options/SUPIR_v0.yaml)
        from SUPIR.util import create_SUPIR_model, convert_dtype
        m = create_SUPIR_model("options/SUPIR_v0.yaml", SUPIR_sign="Q")
        m.ae_dtype = convert_dtype("bf16")
        m.model.dtype = convert_dtype("fp16")
        m = m.half()                      # --loading_half_params
        m = m.to("cuda")
        m.init_tile_vae(encoder_tile_size=512, decoder_tile_size=64)   # --use_tile_vae
        _MODEL = m
    return _MODEL


def decode_image(image_b64):
    try:
        raw = base64.b64decode(image_b64)
    except Exception:
        raise UpErr("decode_failed", "image_b64 is not valid base64")
    try:
        pil = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
    except Image.DecompressionBombError:
        raise UpErr("too_many_pixels", "image exceeds the pixel limit")
    except Exception:
        raise UpErr("decode_failed", "could not decode image")
    mp = (pil.width * pil.height) / 1e6
    if mp > MAX_IN_MP:
        # SUPIR is for restoring small/degraded images; downscale big inputs to stay in VRAM
        s = (MAX_IN_MP / mp) ** 0.5
        pil = pil.resize((max(1, int(pil.width * s)), max(1, int(pil.height * s))), Image.LANCZOS)
    return pil


def run_supir(pil, scale):
    import torch
    from SUPIR.util import PIL2Tensor, Tensor2PIL
    model = get_model()
    lq, h0, w0 = PIL2Tensor(pil, upscale=scale, min_size=1024)
    lq = lq.unsqueeze(0).to("cuda")[:, :3, :, :]
    a_prompt = ("Cinematic, High Contrast, highly detailed, taken using a Canon EOS R camera, "
                "hyper detailed photo-realistic maximum detail, 32k, Color Grading, ultra HD, "
                "extreme meticulous detailing, skin pore detailing, hyper sharpness, perfect "
                "without deformations.")
    n_prompt = ("painting, oil painting, illustration, drawing, art, sketch, cartoon, CG Style, "
                "3D render, unreal engine, blurring, dirty, messy, worst quality, low quality, "
                "frames, watermark, signature, jpeg artifacts, deformed, lowres, over-smooth")
    torch.cuda.reset_peak_memory_stats()
    samples = model.batchify_sample(
        lq, [""], num_steps=EDM_STEPS, restoration_scale=-1, s_churn=5, s_noise=1.01,
        cfg_scale=4.0, control_scale=1.0, seed=1234, num_samples=1,
        p_p=a_prompt, n_p=n_prompt, color_fix_type="Wavelet",
        use_linear_CFG=True, use_linear_control_scale=False,
        cfg_scale_start=1.0, control_scale_start=0.0)
    return Tensor2PIL(samples[0], h0, w0)


def do_upscale(inp, job=None):
    def prog(msg):
        if job is not None:
            try: __import__("runpod").serverless.progress_update(job, msg)
            except Exception: pass
    if "image_b64" not in inp:
        raise UpErr("bad_params", "image_b64 is required")
    scale = int(inp.get("scale", 2))
    if scale not in (1, 2, 4):
        raise UpErr("bad_params", "scale must be 1, 2 or 4")
    out_fmt = inp.get("output", "png").lower()
    if out_fmt not in ("png", "jpg", "jpeg"):
        raise UpErr("bad_params", "output must be png or jpg")

    prog("decoding")
    pil = decode_image(inp["image_b64"])
    in_w, in_h = pil.size
    prog("restoring (SUPIR diffusion — this takes a couple minutes)")
    try:
        out_pil = run_supir(pil, scale)
    except Exception as e:
        if "out of memory" in str(e).lower():
            raise UpErr("oom", "ran out of GPU memory; try a smaller image")
        raise

    prog("saving")
    ext = "png" if out_fmt == "png" else "jpg"
    buf = io.BytesIO()
    out_pil.save(buf, "PNG" if ext == "png" else "JPEG", quality=95)
    data = buf.getvalue()
    thumb = make_thumb(out_pil)
    sha = hashlib.sha256(data).hexdigest()
    job_dir = save_result(job_id_of(job), data, ext, thumb,
                          dict(in_width=in_w, in_height=in_h, width=out_pil.width, height=out_pil.height,
                               bytes=len(data), sha256=sha, model="supir", scale=scale, ext=ext,
                               created=time.time()))
    b64 = base64.b64encode(data).decode()
    return {"job_dir": job_dir, "in_width": in_w, "in_height": in_h,
            "width": out_pil.width, "height": out_pil.height, "bytes": len(data),
            "sha256": sha, "ext": ext, "thumb_b64": thumb,
            "image_b64": b64 if len(b64) <= INLINE_MAX_B64 else None}


def make_thumb(pil, max_side=384):
    p = pil.copy(); p.thumbnail((max_side, max_side))
    for q in (80, 60, 45, 30):
        buf = io.BytesIO(); p.convert("RGB").save(buf, "JPEG", quality=q)
        if buf.getbuffer().nbytes <= 200_000 or q == 30:
            return base64.b64encode(buf.getvalue()).decode()


def job_id_of(job):
    if isinstance(job, dict) and job.get("id"):
        return str(job["id"])
    return "local-%d" % int(time.time() * 1000)


def save_result(job_id, data, ext, thumb_b64, meta):
    os.makedirs(OUT_ROOT, exist_ok=True)
    final = os.path.join(OUT_ROOT, job_id); tmp = final + ".tmp"
    if os.path.exists(tmp): shutil.rmtree(tmp)
    os.makedirs(tmp)
    with open(os.path.join(tmp, "image." + ext), "wb") as f: f.write(data)
    with open(os.path.join(tmp, "thumb.b64"), "w") as f: f.write(thumb_b64)
    with open(os.path.join(tmp, "meta.json"), "w") as f: json.dump(meta, f)
    if os.path.exists(final): shutil.rmtree(final)
    os.rename(tmp, final)
    return job_id


def do_list(inp):
    jobs = []
    if not os.path.isdir(OUT_ROOT): return {"jobs": []}
    for name in os.listdir(OUT_ROOT):
        d = os.path.join(OUT_ROOT, name); mp = os.path.join(d, "meta.json")
        if name.endswith(".tmp") or not os.path.exists(mp): continue
        try:
            meta = json.load(open(mp)); tb = open(os.path.join(d, "thumb.b64")).read()
        except Exception: continue
        jobs.append({"job_dir": name, "created": meta.get("created"),
                     "in_width": meta.get("in_width"), "in_height": meta.get("in_height"),
                     "width": meta.get("width"), "height": meta.get("height"),
                     "bytes": meta.get("bytes"), "sha256": meta.get("sha256"),
                     "ext": meta.get("ext", "png"), "thumb_b64": tb, "model": "supir"})
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
    offset = int(inp.get("offset", 0)); length = int(inp.get("length", 1_572_864))
    if offset < 0 or offset > total: raise UpErr("bad_params", "offset out of range")
    n = max(0, min(length, total - offset))
    with open(path, "rb") as f: f.seek(offset); chunk = f.read(n)
    return {"data_b64": base64.b64encode(chunk).decode(), "offset": offset,
            "bytes": n, "total": total, "eof": (offset + n) >= total}


def purge_old():
    if not os.path.isdir(OUT_ROOT): return
    now = time.time()
    for name in os.listdir(OUT_ROOT):
        d = os.path.join(OUT_ROOT, name)
        try:
            if now - os.path.getmtime(d) > RETENTION_S: shutil.rmtree(d, ignore_errors=True)
        except Exception: pass


def handler(job):
    try:
        inp = (job or {}).get("input", {}) if isinstance(job, dict) else {}
        mode = inp.get("mode", "upscale")
        try: purge_old()
        except Exception: pass
        if mode == "upscale": return do_upscale(inp, job=job)
        if mode == "list": return do_list(inp)
        if mode == "fetch": return do_fetch(inp)
        raise UpErr("bad_mode", "unknown mode: %s" % mode)
    except UpErr as e:
        return {"ok": False, "code": e.code, "message": e.message}
    except Exception as e:
        traceback.print_exc()
        return {"ok": False, "code": "internal", "message": str(e)}


if __name__ == "__main__" and os.environ.get("MODE_TO_RUN", "serverless") == "serverless":
    import runpod
    runpod.serverless.start({"handler": handler})
