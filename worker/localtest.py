#!/usr/bin/env python3
"""Pod-side test of handler.py without Runpod: exercises upscale/list/fetch against local dirs.

Usage on the pod (after weights are present):
  WEIGHTS_DIR=/workspace/upval/weights VOLUME_DIR=/workspace/voltest \
    MODE_TO_RUN=pod python3 localtest.py /workspace/upval/degraded_in.png
"""
import os, sys, base64, json, time
os.environ.setdefault("MODE_TO_RUN", "pod")

import handler as H


def b64_of(path):
    return base64.b64encode(open(path, "rb").read()).decode()


def main():
    img_path = sys.argv[1] if len(sys.argv) > 1 else "/workspace/upval/degraded_in.png"
    print("weights:", os.environ.get("WEIGHTS_DIR"), "volume:", H.VOL)

    # 1) upscale + face enhance 4x
    t0 = time.time()
    r = H.handler({"id": "test-%d" % int(time.time()), "input": {
        "mode": "upscale", "image_b64": b64_of(img_path), "scale": 4,
        "face_enhance": True, "model": "realesrgan", "output": "png"}})
    assert r.get("ok") is not False, r
    print("upscale %.2fs" % (time.time() - t0),
          {k: r[k] for k in ("job_dir", "in_width", "in_height", "width", "height", "bytes")},
          "inline:", r["image_b64"] is not None, "thumb_kb:", len(r["thumb_b64"]) // 1000)
    job_dir = r["job_dir"]; total = r["bytes"]

    # 2) list
    lr = H.handler({"input": {"mode": "list"}})
    assert lr.get("ok") is not False, lr
    print("list ->", len(lr["jobs"]), "job(s); newest:", lr["jobs"][0]["job_dir"] if lr["jobs"] else None)

    # 3) fetch in 1.5MB slices, reassemble, verify size + sha
    import hashlib
    off, parts = 0, []
    while True:
        fr = H.handler({"input": {"mode": "fetch", "job_dir": job_dir, "offset": off, "length": 1_572_864}})
        assert fr.get("ok") is not False, fr
        parts.append(base64.b64decode(fr["data_b64"])); off += fr["bytes"]
        if fr["eof"]:
            break
    blob = b"".join(parts)
    print("fetch reassembled", len(blob), "bytes (expected", total, ") sha_ok:",
          hashlib.sha256(blob).hexdigest() == r["sha256"])

    # 4) error paths
    e1 = H.handler({"input": {"mode": "bogus"}})
    e2 = H.handler({"input": {"mode": "upscale", "image_b64": b64_of(img_path),
                              "model": "aurasr", "face_enhance": True}})
    e3 = H.handler({"input": {"mode": "fetch", "job_dir": "does-not-exist"}})
    print("errors:", e1.get("code"), e2.get("code"), e3.get("code"))
    print("ALL_OK")


if __name__ == "__main__":
    main()
