# ARCHITECTURE.md — upscaler

A personal image-upscaler website: static frontend on **GitHub Pages**, "backend" is a
**Runpod serverless endpoint** the browser calls directly. No server of ours anywhere.

## The key insight: no backend needed

`api.runpod.ai` sends `Access-Control-Allow-Origin: *` with `Authorization` allowed
(**verified live 2026-07-25** via preflight OPTIONS — 204 with the right CORS headers).
So the browser can POST to the endpoint directly. That means:

- The site is pure static files → free GitHub Pages hosting, nothing to operate.
- **Auth = Runpod itself.** The user pastes their Runpod API key into the page; it's
  kept in `localStorage` only and sent only as the `Authorization: Bearer` header to
  `api.runpod.ai`. The site never validates, stores, or transmits it anywhere else.
  Wrong/missing key → Runpod's own 401 — exactly the "reject unauthorized" model.
- The repo contains **zero secrets** (key lives in the visitor's browser).

**Honest caveats of the no-backend model:**
- **The whole design rests on that one CORS behavior.** If Runpod ever tightens
  `api.runpod.ai` (drops `Access-Control-Allow-Origin: *` or disallows `Authorization`),
  the site is inoperable by design — there's no backend to fall back to. Accepted risk.
- **XSS = key theft.** The key sits in `localStorage` on a static page; *any* JS that runs
  on the page (a third-party script, a compromised CDN dep, a supply-chain hit) can read it.
  Mitigation is non-negotiable: `index.html` ships **zero third-party scripts/CDNs** (all JS
  inline/self-hosted), a strict **CSP `<meta>`**, and a visible **"Clear key / forget on this
  device"** control. `localStorage` holds **only** the key + endpoint id — never images/base64.
- **Identity = "anyone who can invoke the endpoint," not "this key."** The history volume is
  scoped to the *endpoint*, so `list`/`fetch` return every result on it regardless of which
  account key called. Correct for this single-user tool; sharing the key (or a team/org key)
  would leak others' images. State it so the assumption is explicit.

```
┌────────────── browser (GitHub Pages static site) ──────────────┐
│ paste API key (localStorage) · drop image · pick scale/face    │
│        │ base64 ≤ ~9.5MB (client-side downscale guard)         │
│        ▼                                                        │
│  POST api.runpod.ai/v2/<ep>/run   {input:{mode:upscale,…}}     │
│  GET  …/status/<jobId>  ← poll 2s → progress UI                │
│  (IN_QUEUE → IN_PROGRESS [+handler stage] → COMPLETED)         │
│        ▼                                                        │
│  small result: inline b64 → show + download                    │
│  big result:   volume → mode:fetch chunks → reassemble blob    │
│  "Recent" tab: mode:list → thumbnails → fetch on click         │
└────────────────────────────────────────────────────────────────┘
                             ▼
        Runpod serverless `upscale` endpoint (scale-to-zero, 4090)
        Real-ESRGAN x4plus + GFPGAN (weights baked into image)
                             ▼
        network volume /runpod-volume/upscale-out/<job>/  (~48h retention)
```

## Endpoint API (handler modes)

```jsonc
// mode: upscale — the main job
{"input": {"mode": "upscale", "image_b64": "…", "scale": 4,        // 2|4
           "face_enhance": true, "model": "realesrgan",            // realesrgan|aurasr
           "output": "png"}}                                        // png|jpg
// -> {"job_dir": "…", "in_width": W0, "in_height": H0,
//     "width": W, "height": H, "bytes": N, "sha256": "…",   // sha256 of the full output file
//     "thumb_b64": "…(≤200KB jpeg)…",
//     "image_b64": "…" | null}     // inline ONLY if encoded b64 ≤ output cap (~1.5MB raw); else null → fetch

// mode: list — cross-device history (endpoint-scoped: any key on the account sees all)
{"input": {"mode": "list"}}
// -> {"jobs": [{"job_dir","created","in_width","in_height","width","height","bytes","sha256","thumb_b64"}…]}

// mode: fetch — chunked download from the volume; each slice base64s ONE raw byte range
{"input": {"mode": "fetch", "job_dir": "…", "offset": 0, "length": 1572864}}   // ~1.5MB raw slices
// -> {"data_b64": "…", "offset": O, "bytes": B, "total": T, "eof": bool}
//    server CLAMPS: B = min(length, total-offset); offset==total -> bytes:0, eof:true (NOT an error);
//    eof true when offset+B == total. total == the `bytes` field from upscale/list (client loop invariant).

// EVERY mode shares one error envelope in the job OUTPUT (also parse-able on FAILED jobs):
// -> {"ok": false, "code": "...", "message": "..."}
//    codes: bad_mode | bad_params | decode_failed | too_many_pixels | oom
//         | model_load_failed | not_found (fetch/list on a purged or unknown job_dir) | internal
//    ⚠ do NOT put a top-level "error" key in the handler's return: the Runpod SDK RESERVES it and
//      drops the payload from /status (verified live 2026-07-26 — a returned {"error":…} came back
//      COMPLETED with no output at all). Success returns the normal dict (no "ok" field).
```

**Handler contract details (all validated in P1, closing reviewer gaps):**
- **Error envelope on every path.** A handled failure returns `{"error":{code,message}}`; an
  uncaught exception lands as a Runpod `FAILED` job whose message the poller also reads. The
  frontend parses one shape either way.
- **Pixel guard, not just bytes.** The ~7 MB request cap does *not* bound megapixels — a small
  highly-compressed JPEG can decode to hundreds of MP after 4× → OOM, and PIL raises
  `DecompressionBombError` past ~89 MP. Handler enforces an explicit **input megapixel cap** and
  returns `too_many_pixels`; the client also downscales by pixel count, not only file size.
- **EXIF orientation** is applied on decode (`ImageOps.exif_transpose`) — phone JPEGs carry an
  orientation tag that PIL/cv2 don't auto-apply; skipping it returns rotated/mirrored output.
- **Alpha/transparency.** Real-ESRGAN (RRDBNet) is 3-channel and GFPGAN ignores alpha. For a PNG
  input the alpha channel is upscaled separately and re-attached; `output:"jpg"` flattens onto a
  stated background (thumbnails are JPEG and can't show transparency).
- **Tiling + OOM.** Default `tile=400` + `tile_pad`; on CUDA OOM, retry once at a smaller tile
  before returning `oom`. GFPGAN's `bg_upsampler` is the RealESRGANer so no-face/background regions
  still upscale. GFPGAN with **no face present** is safe (returns the bg-upsampled image), not an error.
- **Param rejection.** `scale:2` with realesrgan = `outscale=2` (upsample-then-downscale on the 4× net).
  `aurasr` is 4×-only with no face restore → `scale:2`+aurasr and `face_enhance`+aurasr return `bad_params`.
- **Atomic writes.** Each result is written to a temp name and `rename`d into place (write full image →
  thumb → a marker file `list` keys off), so a concurrent `list`/`fetch` never observes a partial dir.
- **`job_dir` = `<stamp>-<rand>`** (random suffix, not an input hash) → no dedup collision / concurrent-write
  race on identical resubmissions.

The volume/list/fetch/purge pattern is lifted from voiceover-lab's handler (proven live there):
results land in `/runpod-volume/upscale-out/<stamp>-<rand>/`, every worker boot purges dirs older
than **48 h** — the volume is a hand-off buffer ("come back on a different computer within a day or
two"), not archival storage. Purge is **time-based only, not size-based**: a burst of large 16 MP
PNGs inside 48 h can fill the volume and fail writes (no quota handling — acceptable for one user).
A `fetch` at ~47 h can race the purge → client treats `not_found` as the normal "expired" case.

## Progress tracker

Two layers, both surfaced in the UI:
1. **Queue level** (free): `/status` returns `IN_QUEUE / IN_PROGRESS / COMPLETED / FAILED`
   — **and also `CANCELLED` / `TIMED_OUT`**, which the poller must handle — plus
   `delayTime`/`executionTime`. Use `delayTime` (waiting for a worker, i.e. cold start) vs
   `executionTime` (actually processing) to tell "booting a GPU" apart from "working," so the
   ~70 s first-job cold start reads as "starting GPU worker — first run can take ~1–2 min," not a hang.
2. **In-job stages**: the handler calls `runpod.serverless.progress_update(job, "…")` at each
   stage (`decoding → upscaling → face enhance → saving`). **Best-effort only:** only the *latest*
   message is exposed, only while `IN_PROGRESS`, and it depends on the worker's runpod-SDK version —
   the UI treats it as a label over the queue-state machine, never assumes it sees every transition,
   and does not rely on it after `COMPLETED`.

**Poller robustness (reviewer gaps):**
- Completed results are retained in `/status` only **~30 min**; a stale `jobId` then 404s. The UI
  falls back to `mode:list`/`fetch` (the volume) — this is *why* the volume exists.
- **Network backoff:** a fixed 2 s poll with no retry spuriously "fails" on a transient 5xx from
  `api.runpod.ai`; wrap polls in retry-with-backoff and only surface failure after N misses.
- **Idempotency:** a Runpod auto-retry of a non-idempotent handler would create duplicate `job_dir`s;
  the random-suffix dir + "one input → one submit" client keep this benign.

## Payload limits (client-side guards)

- **Request cap (`/run`, ~10 MB):** base64 inflates ×1.33, so target **~6 MB encoded** input
  (verify the exact `/run` cap at build time). Bigger images are downscaled client-side by
  **pixel count**, with a notice. Trap: canvas re-encoding a downscaled photo to **PNG can be
  larger than the source JPEG** — the guard re-encodes its output as **JPEG** (loop quality down
  until under budget) and checks the *encoded* size, not pixels.
- **Response/output cap (separate, smaller than the request cap):** results returned via `/status`
  are size-limited, and base64 inflates the response too. So the **inline threshold is small**
  (~1.5 MB raw → ~2 MB encoded, verified in P1) — a 4× of a 1 MP input = 16 MP PNG (10–30 MB) far
  exceeds it → volume + **chunked fetch** (~1.5 MB raw slices, each independently base64'd). The
  thumbnail (≤200 KB JPEG) is always inline for instant feedback.

## Frontend (single `index.html`, vanilla JS, no build step)

- **Key/endpoint fields:** API key (password input, masked, localStorage, blunt "stored in
  plaintext in this browser" note) + endpoint id (localStorage, not hardcoded). A **"Clear key /
  forget on this device"** button is required (localStorage persists indefinitely otherwise).
- **Input:** drag-drop, file `<input>`, **and paste-from-clipboard** (`paste` event →
  `clipboardData.files`/image — the most common input for this kind of tool). On mobile: a file
  input with `capture` for camera; drag-drop doesn't exist on touch.
- **Controls:** scale 2×/4×, face-enhance toggle (default ON), model picker.
- **Flow:** submit → job card with live status (cold-start copy on `delayTime`; best-effort stage
  label on `IN_PROGRESS`) → result panel. **Show input dims+size, output dims+size, achieved scale**
  (all already returned by the handler — core "did it work" feedback). Download button sets a derived
  `download` filename, e.g. `originalname_4x_realesrgan.png`.
- **FAILED handling (must-have, else every failure is an infinite spinner):** render the error and
  distinguish `401` (bad/missing key — the "reject unauthorized" path) · `FAILED` (show the handler's
  `error.message`) · `TIMED_OUT` · `CANCELLED` · endpoint-unreachable/CORS. Offer retry.
- **Before/after slider (vanilla, no library):** two stacked images in a container, top one driven by
  `clip-path: inset(...)` from a range input / pointer-drag; scale the small "before" up via CSS to the
  same display box so the comparison aligns. Uses pointer/touch events (not mouse-only) for mobile.
  Holds a decoded ~16 MP image + the original in memory — fine on desktop, watch on mobile.
- **Recent tab:** enter key on any machine → `mode:list` → thumbnail grid → click → **per-chunk decode
  then byte-concat** (`atob` each chunk → `Uint8Array`, push into an array, `new Blob(parts,{type})` +
  `URL.createObjectURL` + temporary `<a download>`, then `revokeObjectURL`). **Do NOT** join the base64
  strings and `atob` once — chunk boundaries aren't multiples of 3 so padding corrupts the output.
  Optional whole-file `crypto.subtle.digest` sha256 verify (secure-context; GitHub Pages HTTPS is fine).
- Face-hallucination disclaimer next to the toggle **and** a persistent small caption on face-enhanced
  *results* ("face detail is reconstructed, not recovered") — the risky moment is viewing the output.

## Deploy discipline (house rules, all learned the hard way)

1. **Pod first**: validate Real-ESRGAN + GFPGAN on a dev pod with the real test image
   before building any serverless image.
2. Test endpoint → verify with real jobs from the actual frontend → then promote.
   Never trust a run without confirming which image served it (stale-worker gotcha).
3. `runpodctl serverless create` with **`--min-cuda-version 12.8`** (cu12.8 base image
   on older-driver 4090 hosts = empty-log crash-loop) — and remember `runpodctl
   template create` makes POD templates; create the serverless template via API/MCP
   with `isServerless: true`.
4. **Weights via Runpod model cache** (`--model-reference`, the project directive). `--model-reference`
   is **repeatable** and speaks HuggingFace, so instead of minting a new repo (needs an HF token) we
   attach **two public HF repos whose files are verified byte-identical (sha256) to the official
   GitHub-release weights** — pinned to a commit so they can't be swapped later:
   - `https://huggingface.co/amd/realesrgan-x4plus:bda69abcaf525425b371622349e975245ae090c2` → `RealESRGAN_x4plus.pth`
   - `https://huggingface.co/gmk123/GFPGAN:e881fbc251fdf2a4f133ad8277dd5dadbd1c541a` → `GFPGANv1.4.pth`, `detection_Resnet50_Final.pth`, `parsing_parsenet.pth`

   Runpod host-caches both; the worker loads every file **offline** from
   `/runpod-volume/huggingface-cache/hub/models--…/snapshots/<hash>/` (`HF_HUB_OFFLINE=1`). The handler's
   `resolve_weights()` globs across *all* snapshot dirs for the four filenames (so multiple repos are
   fine) and copies the facexlib pair into facexlib's package dir so nothing downloads at inference.
   **Security:** using third-party mirrors is only safe because we verified the sha256 of every file
   against the official weights *and* pinned the commit — `torch.load` executes pickle, so an unverified
   mirror would be an RCE vector. If either repo ever disappears, mint your own HF repo from the four
   official files and swap the two refs. Container disk 20–25 GB, **no weights baked**.
   (Engineering note: for ~585 MB model-cache ≈ baking cost-wise since the torch/CUDA base dominates
   either way; model-reference is used per directive and its real payoff is guaranteeing the facexlib
   files are pre-placed so nothing downloads at runtime.)
   **basicsr gotcha:** the image build patches `torchvision.transforms.functional_tensor` → `functional`
   (removed in torchvision ≥ 0.17), else GFPGAN won't import (see MODELS.md).
5. The volume DC pins the endpoint — pick the volume DC by intersecting
   volume-capable DCs × 4090 stock (`get-gpu-type` per-DC availability) at deploy time. (The existing
   `vo-results` volume in **US-IL-1** can be reused with a `upscale-out/` subdir.)
6. `/runsync` returns after ~90 s on cold starts → the frontend only ever uses
   `/run` + `/status` polling (which we want anyway for progress). Verify the exact **request and
   output payload caps** with a real job during P2 and record them here.
7. GitHub Pages repo must be public for free Pages (or stay private on a Pro plan) —
   fine either way: the repo never contains keys or endpoint-private data. Endpoint id
   is entered in the UI and kept in localStorage, not committed.

## Build phases

- **P0 — research & architecture** ✅ (this repo)
- **P1 — pod validation**: Real-ESRGAN x4plus + GFPGAN on a dev pod against the real
  low-res test photo; measure time/VRAM; pick tile settings; compare AuraSR-v2 side by
  side.
- **P2 — worker + serverless**: dual-mode handler (upscale/list/fetch + progress
  updates + purge), volume, test endpoint verified from the real frontend, promote.
- **P3 — site**: `index.html` on GitHub Pages (drag-drop, progress, before/after,
  Recent tab), README quickstart.
- **P4 — optional quality tier**: SUPIR endpoint (non-commercial license OK for
  personal use; 48 GB GPU + model-reference) as a "max quality, slow" toggle.
