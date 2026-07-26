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
// -> {"job_dir": "…", "width": W, "height": H, "bytes": N, "sha256": "…",
//     "thumb_b64": "…(≤200KB jpeg)…",
//     "image_b64": "…"}            // inline ONLY if result ≤ ~8MB, else fetch it

// mode: list — cross-device history (anyone with the API key sees it; single-user tool)
{"input": {"mode": "list"}}
// -> {"jobs": [{"job_dir","created","width","height","bytes","sha256","thumb_b64"}…]}

// mode: fetch — chunked download from the volume (10MB slices, sha256-verify client-side)
{"input": {"mode": "fetch", "job_dir": "…", "offset": 0, "length": 10485760}}
// -> {"data_b64": "…", "offset": 0, "bytes": N, "total": T, "eof": false}
```

The volume/list/fetch/purge pattern is lifted verbatim from voiceover-lab's handler
(proven live there): results land in `/runpod-volume/upscale-out/<stamp>-<hash>/`,
every worker boot purges dirs older than **48 h** — the volume is a hand-off buffer
("come back on a different computer within a day or two"), not archival storage.

## Progress tracker

Two layers, both surfaced in the UI:
1. **Queue level** (free): `/status` returns `IN_QUEUE / IN_PROGRESS / COMPLETED /
   FAILED` plus `delayTime`/`executionTime` — the poller renders this as a status line.
2. **In-job stages**: the handler calls `runpod.serverless.progress_update(job, "…")`
   at each stage (`decoding → upscaling → face enhance → saving`) — Runpod exposes the
   latest message in the `/status` response while the job runs, so the UI can show a
   real stage, not just a spinner.

## Payload limits (client-side guards)

- `/run` accepts ~10 MB request payloads → base64 inflates ×1.33, so inputs are capped
  at ~7 MB file size; bigger images are canvas-downscaled client-side (with a notice)
  before upload. (A 12 MP phone JPEG is usually ~3–5 MB — fine.)
- 4× output of a 1 MP input = 16 MP PNG (10–30 MB) → too big for an inline response →
  volume + chunked fetch. Thumbnail is always inline for instant feedback.

## Frontend (single `index.html`, vanilla JS, no build step)

- Fields: API key (password input, localStorage, "stored only in this browser" note),
  endpoint id (localStorage; not hardcoded in the repo), drag-drop/upload, scale 2×/4×,
  face-enhance toggle (default ON), model picker.
- Flow: submit → job card with live status → result panel (before/after slider,
  download button).
- **Recent tab**: enter key on any machine → `mode:list` → thumbnail grid → click →
  chunked fetch → download. This is the cross-computer story; no accounts needed
  because the API key IS the identity.
- Face-hallucination disclaimer displayed next to the face-enhance toggle (see MODELS.md).

## Deploy discipline (house rules, all learned the hard way)

1. **Pod first**: validate Real-ESRGAN + GFPGAN on a dev pod with the real test image
   before building any serverless image.
2. Test endpoint → verify with real jobs from the actual frontend → then promote.
   Never trust a run without confirming which image served it (stale-worker gotcha).
3. `runpodctl serverless create` with **`--min-cuda-version 12.8`** (cu12.8 base image
   on older-driver 4090 hosts = empty-log crash-loop) — and remember `runpodctl
   template create` makes POD templates; create the serverless template via API/MCP
   with `isServerless: true`.
4. Weights baked into the image (~400 MB) → no model-reference needed for the default
   tier; container disk 20–25 GB.
5. The volume DC pins the endpoint — pick the volume DC by intersecting
   volume-capable DCs × 4090 stock (`get-gpu-type` per-DC availability) at deploy time.
6. `/runsync` returns after ~90 s on cold starts → the frontend only ever uses
   `/run` + `/status` polling (which we want anyway for progress).
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
