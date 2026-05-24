"""
Flask wrapper around Pipeline.py.

Endpoints:
  GET  /health    - liveness check
  POST /analyze   - form-data or JSON body:
                      incident_id  (required, str — e.g. "INC-2026-04-001")
                                   the row in public.incidents to update,
                                   AND the folder name inside the
                                   Supabase Storage bucket `accident-images`
                                   from which the picture is fetched
                                   (first image file in that folder is used).
                                   The web app must have created the row and
                                   uploaded the image before calling /analyze.

Returns JSON:
  {
    "severity": "Severe" | "Moderate" | "Minor",
    "severity_confidence": 87.42,
    "plate_1_text": "ZER1048 | زعر١٠٤٨",
    "plate_1_confidence": 91.20,
    "plate_2_text": "...",
    "plate_2_confidence": 84.55,
    "supabase": { ...patched row... } | {"error": "...", ...} | null
  }

NOTE: importing Pipeline triggers model loading (5-15s). That happens once at
startup, not per request.
"""

import os
import tempfile
import time
import traceback
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS

# Load SUPABASE_URL / SUPABASE_KEY (and anything else) from a local .env file
# so you can just run `./start.sh` (or `python3 app.py`) without exporting vars.
load_dotenv()

# Importing Pipeline loads YOLO + EasyOCR models at module import time.
import Pipeline

app = Flask(__name__)
# Allow browser clients (React app) to call this API cross-origin.
# For prod, narrow this to your real frontend origin instead of "*".
CORS(app, resources={r"/*": {"origins": "*"}})

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

ALLOWED_EXTS = {"png", "jpg", "jpeg", "webp", "bmp"}

STORAGE_BUCKET = "accident-images"

# Wait-for-upload settings: if the web app calls /analyze before the picture
# finishes uploading, poll the folder up to STORAGE_WAIT_TOTAL seconds total,
# checking every STORAGE_WAIT_INTERVAL seconds.
STORAGE_WAIT_INTERVAL = 10   # seconds between attempts
STORAGE_WAIT_TOTAL    = 60   # give up after this many seconds

# Optional shared secret for the Supabase Storage webhook. If set, the
# webhook caller must send it in the X-Webhook-Secret header.
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "")


def _ext_ok(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTS


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


def _run_analysis_for_incident(incident_id):
    """
    Shared logic used by both /analyze and the Supabase Storage webhook.
    Returns (flask_response, status_code).
    """
    tmp_path, fetch_error = _fetch_image_from_storage(incident_id)
    if fetch_error is not None:
        return fetch_error
    print(f"[analyze] incident_id={incident_id}  using image from storage -> {tmp_path}")

    try:
        result = Pipeline.run_pipeline_on_image(tmp_path)
    except Exception as e:
        print("[analyze] PIPELINE FAILURE for incident_id=", incident_id)
        traceback.print_exc()
        return jsonify({"error": "pipeline failure", "detail": str(e)}), 500
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    supabase_response = _update_supabase_incident(
        incident_id=incident_id,
        severity=result["severity"],
        severity_confidence=result["severity_confidence"],
        plate_1_text=result["plate_1_text"],
        plate_1_confidence=result["plate_1_confidence"],
        plate_2_text=result["plate_2_text"],
        plate_2_confidence=result["plate_2_confidence"],
    )

    status_code = 200
    if isinstance(supabase_response, dict) and supabase_response.get("error"):
        status_code = supabase_response.get("status") or 500
    return jsonify({**result, "supabase": supabase_response}), status_code


@app.post("/analyze")
def analyze():
    if request.is_json:
        incident_id = (request.get_json(silent=True) or {}).get("incident_id", "")
    else:
        incident_id = request.form.get("incident_id", "")
    incident_id = (incident_id or "").strip()
    if not incident_id:
        return jsonify({"error": "missing 'incident_id' (e.g. INC-2026-04-001)"}), 400
    return _run_analysis_for_incident(incident_id)


@app.post("/webhook/storage")
def webhook_storage():
    """
    Endpoint for the Supabase Database Webhook on `storage.objects` (INSERT).
    Payload shape:
        {
          "type": "INSERT",
          "table": "objects",
          "schema": "storage",
          "record": {
              "name":      "INC-2026-05-020/photo.jpg",
              "bucket_id": "accident-images",
              ...
          },
          "old_record": null
        }
    Extracts the incident_id from the object path (everything before the first '/'),
    then runs the same analysis pipeline.
    """
    if WEBHOOK_SECRET:
        provided = request.headers.get("X-Webhook-Secret", "")
        if provided != WEBHOOK_SECRET:
            return jsonify({"error": "unauthorized"}), 401

    payload = request.get_json(silent=True) or {}
    record = payload.get("record") or {}
    bucket = record.get("bucket_id")
    name   = record.get("name") or ""

    print(f"[webhook] type={payload.get('type')} bucket={bucket} name={name}")

    if bucket != STORAGE_BUCKET:
        # Not our bucket — acknowledge and ignore so Supabase doesn't keep retrying.
        return jsonify({"ignored": True, "reason": "different bucket", "bucket": bucket}), 200

    if "/" not in name:
        return jsonify({"ignored": True, "reason": "object is at bucket root, no incident folder",
                        "name": name}), 200

    incident_id = name.split("/", 1)[0].strip()
    if not incident_id:
        return jsonify({"ignored": True, "reason": "empty incident_id parsed from name",
                        "name": name}), 200

    if not _ext_ok(name):
        return jsonify({"ignored": True, "reason": "not an image file", "name": name}), 200

    return _run_analysis_for_incident(incident_id)


def _list_image_in_folder(incident_id):
    """
    One attempt at listing the incident's folder.
    Returns (image_name, None) on success, (None, (response, status)) on hard error,
    or (None, None) if the folder is simply empty (caller may retry).
    """
    list_url = f"{SUPABASE_URL}/storage/v1/object/list/{STORAGE_BUCKET}"
    headers = {
        "apikey":        SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type":  "application/json",
    }
    try:
        resp = requests.post(
            list_url,
            headers=headers,
            json={"prefix": f"{incident_id}/", "limit": 100,
                  "sortBy": {"column": "name", "order": "asc"}},
            timeout=15,
        )
    except requests.RequestException as e:
        return None, (jsonify({"error": "storage_list_request_failed", "detail": str(e)}), 502)

    if resp.status_code >= 400:
        return None, (jsonify({"error": "storage_list_rejected",
                               "status": resp.status_code,
                               "detail": resp.text}), 502)
    try:
        entries = resp.json()
    except ValueError:
        return None, (jsonify({"error": "storage_list_non_json", "detail": resp.text}), 502)

    image_name = next((e["name"] for e in entries
                       if isinstance(e, dict) and e.get("name") and _ext_ok(e["name"])), None)
    return image_name, None  # image_name may be None → folder still empty


def _fetch_image_from_storage(incident_id):
    """
    Pull the picture for this incident from Supabase Storage:
        bucket = accident-images
        folder = <incident_id>/
        file   = first image found in that folder

    If the folder is empty (web app hasn't finished uploading yet), polls every
    STORAGE_WAIT_INTERVAL seconds up to STORAGE_WAIT_TOTAL seconds before giving up.

    Returns (tmp_path, None) on success, or (None, (flask_response, status)) on failure.
    The caller is responsible for deleting tmp_path.
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None, (jsonify({"error": "SUPABASE_URL/SUPABASE_KEY not configured"}), 500)

    deadline = time.monotonic() + STORAGE_WAIT_TOTAL
    attempt = 0
    image_name = None
    while True:
        attempt += 1
        image_name, hard_error = _list_image_in_folder(incident_id)
        if hard_error is not None:
            return None, hard_error          # auth / network failure: don't keep retrying
        if image_name:
            print(f"[storage] found '{image_name}' on attempt {attempt}")
            break
        if time.monotonic() >= deadline:
            return None, (jsonify({"error": "no_image_in_folder",
                                   "bucket": STORAGE_BUCKET,
                                   "folder": incident_id,
                                   "waited_seconds": STORAGE_WAIT_TOTAL,
                                   "attempts": attempt}), 404)
        print(f"[storage] folder '{incident_id}/' empty (attempt {attempt}), "
              f"retrying in {STORAGE_WAIT_INTERVAL}s...")
        time.sleep(STORAGE_WAIT_INTERVAL)

    object_path = f"{incident_id}/{image_name}"
    get_url = f"{SUPABASE_URL}/storage/v1/object/{STORAGE_BUCKET}/{object_path}"
    try:
        get_resp = requests.get(get_url,
                                headers={"apikey": SUPABASE_KEY,
                                         "Authorization": f"Bearer {SUPABASE_KEY}"},
                                timeout=30)
    except requests.RequestException as e:
        return None, (jsonify({"error": "storage_download_failed", "detail": str(e)}), 502)

    if get_resp.status_code >= 400:
        return None, (jsonify({"error": "storage_download_rejected",
                               "status": get_resp.status_code,
                               "object_path": object_path,
                               "detail": get_resp.text}), 502)

    suffix = "." + image_name.rsplit(".", 1)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(get_resp.content)
        return tmp.name, None


def _update_supabase_incident(**fields):
    """
    PATCH the existing row in public.incidents matched by incident_id (text key).

    Updates only the ML-derived columns:
        severity, confidence, plate_1_text, plate_1_confidence,
        plate_2_text, plate_2_confidence
    Other columns (location, lat/lng, date, time, status, video_url) are left
    alone — the web app set them when it created the row.

    Returns:
        - The updated row(s) on success
        - {"error": "...", "status": int, ...} on HTTP/API failure
        - {"error": "incident_not_found", ...} if no row matched the incident_id
        - None if SUPABASE_URL/SUPABASE_KEY env vars aren't set (no-op)
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None

    incident_id = fields["incident_id"]
    patch = {
        "severity":           fields["severity"],
        "confidence":         fields["severity_confidence"],
        "plate_1_text":       fields["plate_1_text"]  or None,
        "plate_1_confidence": fields["plate_1_confidence"],
        "plate_2_text":       fields["plate_2_text"]  or None,
        "plate_2_confidence": fields["plate_2_confidence"],
        "updated_at":         datetime.now(timezone.utc).isoformat(),
    }

    try:
        resp = requests.patch(
            f"{SUPABASE_URL}/rest/v1/incidents",
            params={"incident_id": f"eq.{incident_id}"},
            headers={
                "apikey":        SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}",
                "Content-Type":  "application/json",
                "Prefer":        "return=representation",
            },
            json=patch,
            timeout=15,
        )
    except requests.RequestException as e:
        return {"error": "request_failed", "detail": str(e), "status": 502}

    if resp.status_code >= 400:
        return {"error": "supabase_rejected_update",
                "status": resp.status_code,
                "detail": resp.text}

    try:
        body = resp.json()
    except ValueError:
        return {"error": "non_json_response", "detail": resp.text, "status": 502}

    # PATCH with no matching row returns 200 + empty list. Detect that.
    if isinstance(body, list) and len(body) == 0:
        return {"error": "incident_not_found",
                "incident_id": incident_id,
                "hint": "Web app must create the incident row before /analyze is called.",
                "status": 404}

    return body


if __name__ == "__main__":
    # debug=False so models are loaded once (debug=True double-imports the module)
    app.run(host="0.0.0.0", port=5000, debug=False)
