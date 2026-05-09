"""
Flask wrapper around Pipeline.py.

Endpoints:
  GET  /health    - liveness check
  POST /analyze   - multipart/form-data:
                      image        (required, file)
                      incident_id  (required, str — e.g. "INC-2026-04-001")
                                   the row in public.incidents to update.
                                   The web app must have created the row first.

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
from datetime import datetime, timezone

import requests
from flask import Flask, request, jsonify

# Importing Pipeline loads YOLO + EasyOCR models at module import time.
import Pipeline

app = Flask(__name__)

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

ALLOWED_EXTS = {"png", "jpg", "jpeg", "webp", "bmp"}


def _ext_ok(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTS


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/analyze")
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "missing 'image' file in multipart form-data"}), 400

    f = request.files["image"]
    if f.filename == "" or not _ext_ok(f.filename):
        return jsonify({"error": f"unsupported file (allowed: {sorted(ALLOWED_EXTS)})"}), 400

    incident_id = request.form.get("incident_id", "").strip()
    if not incident_id:
        return jsonify({"error": "missing 'incident_id' (e.g. INC-2026-04-001)"}), 400

    suffix = "." + f.filename.rsplit(".", 1)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        f.save(tmp.name)
        tmp_path = tmp.name

    try:
        result = Pipeline.run_pipeline_on_image(tmp_path)
    except Exception as e:
        return jsonify({"error": "pipeline failure", "detail": str(e)}), 500
    finally:
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
