#!/usr/bin/env python3
"""Live end-to-end check of the deployed endpoint: upscale -> list -> chunked fetch (+ sha) -> errors.
Usage: RUNPOD_API_KEY=... python3 endpoint_e2e.py <endpoint_id> <image_path>
"""
import sys, os, json, time, base64, hashlib, urllib.request

KEY = os.environ["RUNPOD_API_KEY"]
EP = sys.argv[1]
IMG = sys.argv[2] if len(sys.argv) > 2 else "/Users/justin/Desktop/upscaler/.samples/degraded_in.png"
BASE = f"https://api.runpod.ai/v2/{EP}"
H = {"Authorization": "Bearer " + KEY, "Content-Type": "application/json"}


def post(path, body):
    r = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers=H)
    return json.load(urllib.request.urlopen(r, timeout=60))


def run_sync(inp, label=""):
    jid = post("/run", {"input": inp})["id"]
    t0 = time.time()
    while True:
        s = json.load(urllib.request.urlopen(urllib.request.Request(BASE + "/status/" + jid, headers=H), timeout=60))
        st = s.get("status")
        if st in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
            print(f"[{label}] {st} in {time.time()-t0:.1f}s (delay={s.get('delayTime')} exec={s.get('executionTime')})")
            return s.get("output")
        time.sleep(3)


def main():
    b64 = base64.b64encode(open(IMG, "rb").read()).decode()
    out = run_sync({"mode": "upscale", "image_b64": b64, "scale": 4, "face_enhance": True,
                    "model": "realesrgan", "output": "png"}, "upscale")
    assert out and out.get("ok") is not False, out
    print("  ->", {k: out.get(k) for k in ("job_dir", "in_width", "in_height", "width", "height", "bytes")},
          "inline:", out.get("image_b64") is not None)
    job_dir, total, sha = out["job_dir"], out["bytes"], out["sha256"]

    lst = run_sync({"mode": "list"}, "list")
    print("  list jobs:", len(lst.get("jobs", [])), "newest:", lst["jobs"][0]["job_dir"] if lst.get("jobs") else None)

    off, parts = 0, []
    while True:
        fo = run_sync({"mode": "fetch", "job_dir": job_dir, "offset": off, "length": 1_572_864}, "fetch")
        parts.append(base64.b64decode(fo["data_b64"])); off += fo["bytes"]
        if fo["eof"]:
            break
    blob = b"".join(parts)
    print("  fetch reassembled", len(blob), "== expected", total, "sha_ok:", hashlib.sha256(blob).hexdigest() == sha)

    e = run_sync({"mode": "bogus"}, "err-bad_mode")
    print("  err ok/code:", (e or {}).get("ok"), (e or {}).get("code"))
    print("ENDPOINT_E2E_OK" if (len(blob) == total and hashlib.sha256(blob).hexdigest() == sha) else "MISMATCH")


if __name__ == "__main__":
    main()
