# upscaler

Personal image-upscaler website: **GitHub Pages frontend → your Runpod serverless
endpoint, directly from the browser.** No backend server, no accounts, no stored keys —
you paste your Runpod API key into the page (kept in your browser's localStorage only)
and Runpod itself rejects anyone without it.

**Status: P0 — architecture + model research done. Implementation not started.**

## Read order
1. **`MODELS.md`** — model research: Real-ESRGAN+GFPGAN default (permissive licenses),
   AuraSR-v2 alt, SUPIR as optional slow "max quality" tier; who was rejected and why.
2. **`ARCHITECTURE.md`** — the plan: browser→Runpod direct (CORS verified live),
   handler modes (upscale/list/fetch), progress tracking, 48 h volume history for
   cross-device access, payload limits, deploy discipline, phases P1–P4.

## The shape (tl;dr)

- **Upload** an image + paste API key → browser POSTs to `api.runpod.ai/v2/<ep>/run`.
- **Progress**: poll `/status` — queue state + in-job stage (via
  `runpod.serverless.progress_update`).
- **Result**: inline for small outputs; big ones land on a network volume and stream
  back in chunks. Before/after view + download.
- **Recent**: results are kept ~48 h on the volume — open the site on any computer,
  paste the same key, see and re-download recent upscales.
- **Model**: Real-ESRGAN x4plus + GFPGAN face-enhance (BSD-3/Apache-2.0), weights baked
  into the worker image. Face enhance is a visible toggle — restored faces are
  plausible reconstructions, not forensic recovery.

## House rules
- No keys, endpoint ids, or user images ever committed to this repo.
- Pod-first validation, test-endpoint-first deploys, `--min-cuda-version 12.8`.
