# Najm Pipeline

A Flask service that runs a YOLO + EasyOCR vision pipeline on accident images
stored in Supabase. For each incident it:

1. Downloads the picture from `accident-images/<incident_id>/` (Supabase Storage)
2. Detects severity (Minor / Moderate / Severe)
3. Reads the top 2 license plates (English + Arabic)
4. Writes the results back into the matching row of `public.incidents`

It can be triggered three ways:

- `POST /analyze` (manual call from a script, curl, or your web app)
- `POST /webhook/storage` (Supabase Storage webhook — fires automatically on upload)
- Direct script run (`python Pipeline.py` for testing on `test.png`)

---

## Requirements

- **macOS / Linux** (tested on macOS, Python 3.14)
- **Python 3.14** in a venv (already shipped in `venv/` in this repo)
- A **Supabase project** with:
  - A `public.incidents` table (with `incident_id` text column + the
    severity / plate columns the pipeline writes)
  - A Storage bucket named `accident-images`
- **ngrok** (only if you want the Supabase webhook auto-trigger to reach your
  laptop)

The model weights (`Severity_best_model.pt`, `plate_best.pt`, `region_best.pt`,
`accident_yolov8n_best (1).pt`) are committed alongside the code.

---

## One-time setup

### 1. Create `.env` from the template

```bash
cp .env.example .env
```

Open `.env` and fill in real values from
**Supabase Dashboard → Project Settings → API**:

```env
SUPABASE_URL=https://<your-project-ref>.supabase.co
SUPABASE_KEY=sb_secret_xxxxxxxxxxxxxxxxxxxxxxxx
WEBHOOK_SECRET=any-random-string-you-make-up
```

| Var | Where to get it | Notes |
|---|---|---|
| `SUPABASE_URL` | Project Settings → API → Project URL | |
| `SUPABASE_KEY` | Project Settings → API → **Secret keys** (NOT Publishable) | Needed to bypass RLS so the pipeline can list storage and update rows |
| `WEBHOOK_SECRET` | You invent it | Optional. If set, the Supabase webhook must send the same value in `X-Webhook-Secret`. Remove the line to disable the check. |

> ⚠️ Never commit `.env` or share the Secret key. `.gitignore` already excludes it.

### 2. (Skip if `venv/` already exists) Recreate the venv

```bash
python3.14 -m venv venv
./venv/bin/python3.14 -m pip install --upgrade pip
./venv/bin/python3.14 -m pip install \
  flask flask-cors requests python-dotenv \
  opencv-python ultralytics easyocr torch torchvision pillow numpy
```

---

## Run the server

```bash
./start.sh
```

That's it. The script:

- Checks `.env` exists
- Launches the pipeline using the venv's Python

You'll see model-loading logs, then:

```
 * Running on http://127.0.0.1:5000
 * Running on http://192.168.1.x:5000
Press CTRL+C to quit
```

To stop: **Ctrl+C**.

### Endpoints

| Method | Path | What it does |
|---|---|---|
| GET | `/health` | Liveness check → `{"status":"ok"}` |
| POST | `/analyze` | Run pipeline for a given `incident_id`. JSON or form body. |
| POST | `/webhook/storage` | Supabase Storage webhook receiver |

---

## Manual test

In a **second terminal** while the server is running:

```bash
curl -i -X POST http://localhost:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{"incident_id":"INC-2026-05-018"}' \
  --max-time 120
```

Prerequisites for the request to succeed:

- A row with `incident_id = INC-2026-05-018` exists in `public.incidents`
- An image is uploaded to `accident-images/INC-2026-05-018/<anything>.jpg`

If the folder is empty, the pipeline polls every 10 s for up to 60 s before
returning `404 no_image_in_folder`.

---

## Auto-trigger via Supabase webhook (optional)

So you don't have to curl anything — the moment you upload an image to
`accident-images/<incident_id>/`, the pipeline runs automatically.

### A. Expose the pipeline publicly with ngrok

```bash
brew install ngrok/ngrok/ngrok                       # one-time
ngrok config add-authtoken <your-token-from-ngrok>   # one-time
ngrok http 5000                                      # every session
```

Copy the `Forwarding` URL (e.g. `https://e687-….ngrok-free.app`).

> ⚠️ Free ngrok URLs change every restart. You'll need to update the Supabase
> webhook URL whenever ngrok restarts.

### B. Create the Supabase Database Webhook

In Supabase Dashboard → **Database → Webhooks → Create a new hook**:

| Field | Value |
|---|---|
| Name | `accident-image-uploaded` |
| Table | Schema `storage`, Table **`objects`** |
| Events | ☑ Insert (only) |
| Type | HTTP Request |
| Method | `POST` |
| URL | `https://<your-ngrok-subdomain>.ngrok-free.app/webhook/storage` |
| HTTP Headers | `Content-Type: application/json` <br> `X-Webhook-Secret: <same value as WEBHOOK_SECRET in .env>` |

Click **Create webhook**.

### C. Test it

1. Make sure `./start.sh` and `ngrok http 5000` are both running.
2. Insert a row into `public.incidents` with `incident_id = INC-TEST-001`.
3. In Supabase Storage, upload an image to `accident-images/INC-TEST-001/`.
4. In the pipeline terminal you should see:
   ```
   [webhook] type=INSERT bucket=accident-images name=INC-TEST-001/photo.jpg
   [storage] found 'photo.jpg' on attempt 1
   [analyze] incident_id=INC-TEST-001  using image from storage -> /var/folders/.../tmpXXX.jpg
   "POST /webhook/storage HTTP/1.1" 200 -
   ```
5. The row in `public.incidents` now has `severity`, `confidence`,
   `plate_1_text`, `plate_1_confidence`, `plate_2_text`, `plate_2_confidence`,
   and a fresh `updated_at`.

---

## Project layout

```
.
├── app.py                     Flask wrapper: endpoints, Supabase storage fetch, webhook handler
├── Pipeline.py                YOLO + EasyOCR vision pipeline (the actual ML)
├── start.sh                   "npm start" equivalent — checks .env and launches app.py
├── .env.example               Template — copy to .env and fill in real values
├── .env                       Your real secrets (gitignored)
├── .gitignore
├── venv/                      Python 3.14 virtualenv with all deps installed
├── *.pt                       YOLO model weights
├── test.png                   Sample image for direct Pipeline.py runs
└── Output/                    Per-incident debug images (annotated plates, region crops, CSVs)
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `command not found: python` | venv has no `python` shim | Use `./venv/bin/python3.14` or `./start.sh` |
| `ModuleNotFoundError: No module named 'requests'` | You're hitting system Python | Use `./start.sh` (uses the venv) |
| `Address already in use` on port 5000 | Old server still running, or macOS AirPlay Receiver | `lsof -ti :5000 \| xargs kill -9`, or disable AirPlay Receiver in System Settings |
| `502` with `your-project.supabase.co` in detail | `.env` still has placeholder values | Put real Supabase URL + Secret key, restart |
| `500 SUPABASE_URL/SUPABASE_KEY not configured` | `.env` not loaded (empty values, quotes, wrong format) | Run the diagnostic in the "Verify .env" section below; restart |
| `404 no_image_in_folder` after 60 s | Image was never uploaded to that folder | Check `accident-images/<incident_id>/` in Supabase Storage |
| `incident_not_found` | No row in `incidents` matching the `incident_id` | Insert the row first |
| Webhook returns `401 unauthorized` | `X-Webhook-Secret` ≠ `WEBHOOK_SECRET` in `.env` | Make them match exactly, restart |
| Webhook returns `500` | Pipeline crashed mid-run | Look at the traceback in the pipeline terminal (printed after `[analyze] PIPELINE FAILURE`) |

### Verify `.env` is being loaded

```bash
./venv/bin/python3.14 -c "from dotenv import load_dotenv; import os; load_dotenv(); print('URL:', repr(os.environ.get('SUPABASE_URL'))); print('KEY length:', len(os.environ.get('SUPABASE_KEY','') or ''))"
```

Both should be non-empty. If `URL` is `None` or `KEY length: 0`, fix `.env`.

### See raw `.env` (catches hidden chars)

```bash
cat -A .env
```

Each line should end in `$` (newline). No `M-` or weird control chars in the middle.

---

## Notes on output formatting

- **Severity:** Title-case string (`Severe`, `Moderate`, `Minor`) — matches the
  Supabase `CHECK` constraint.
- **Plate text:** Combined English + Arabic, e.g. `ZER1048 | ز ع ر ١٠٤٨`.
  Arabic letters are spaced apart for readability (Arabic numerals stay joined).
- **Confidence:** 0–100, two decimals.

---

## Caveats

- The pipeline runs on **CPU** by default (no CUDA on Mac). Expect ~5–15 s
  startup for model loading and a few seconds per request.
- The Flask dev server is fine for local testing only. For production, swap to
  `gunicorn` (`gunicorn -w 1 -b 0.0.0.0:5000 app:app`) and host behind a real
  proxy.
- ngrok free URLs rotate on every restart — re-update the Supabase webhook URL
  each time, or upgrade to a paid plan for a static domain.
