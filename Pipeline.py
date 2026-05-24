# Generated from: Pipeline.ipynb
# Converted at: 2026-05-09T14:59:24.245Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# --- Colab-specific lines disabled for local run ---
# from google.colab import drive
# drive.mount('/content/drive')
# !pip install ultralytics easyocr opencv-python pandas matplotlib -q
# !pip install easyocr

from ultralytics import YOLO

# Early model loads removed — real loads happen later with local paths.

# #Final


# =========================================================
# CLOUD ACCIDENT PIPELINE
# Runs on:
# 1) TEST_IMAGE_PATH
# 2) all images inside INPUT_IMAGES_DIR
#
# Logic:
# - No accident detection model
# - Severity is decided once for the whole accident
# - Final severity uses ACCUMULATIVE CONFIDENCE VOTING
# - Each image is counted only once for severity
# - Multiple plate candidates can be selected
# - Region model runs on FULL IMAGE
# - OCR runs only on original ET and EN crops
# =========================================================

import os
import cv2
import glob
import torch
import easyocr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ultralytics import YOLO


# =========================================================
# 1) PATHS
# =========================================================

ACCIDENT_ID = "ACC_001"

_HERE = os.path.dirname(os.path.abspath(__file__))

TEST_IMAGE_PATH = os.path.join(_HERE, "test.png")
INPUT_IMAGES_DIR = os.path.join(_HERE, "_no_frames")  # empty/non-existent: skips folder scan

OUTPUT_DIR = os.path.join(_HERE, "Output")

SEVERITY_MODEL_PATH = os.path.join(_HERE, "Severity_best_model.pt")
PLATE_MODEL_PATH = os.path.join(_HERE, "plate_best.pt")
REGION_MODEL_PATH = os.path.join(_HERE, "region_best.pt")


# =========================================================
# 2) OUTPUT FOLDERS
# =========================================================

ACCIDENT_OUTPUT_DIR = os.path.join(OUTPUT_DIR, ACCIDENT_ID)

ORIGINAL_IMAGES_DIR = os.path.join(ACCIDENT_OUTPUT_DIR, "original_images")
ANNOTATED_DIR = os.path.join(ACCIDENT_OUTPUT_DIR, "annotated_images")
PLATE_CROPS_DIR = os.path.join(ACCIDENT_OUTPUT_DIR, "plate_crops")
REGION_DEBUG_DIR = os.path.join(ACCIDENT_OUTPUT_DIR, "region_debug_images")
OCR_CROPS_DIR = os.path.join(ACCIDENT_OUTPUT_DIR, "ocr_region_crops")
CSV_DIR = os.path.join(ACCIDENT_OUTPUT_DIR, "csv_results")

for folder in [
    ACCIDENT_OUTPUT_DIR,
    ORIGINAL_IMAGES_DIR,
    ANNOTATED_DIR,
    PLATE_CROPS_DIR,
    REGION_DEBUG_DIR,
    OCR_CROPS_DIR,
    CSV_DIR
]:
    os.makedirs(folder, exist_ok=True)


# =========================================================
# 3) SETTINGS
# =========================================================

SEVERITY_CONF = 0.25
PLATE_CONF = 0.25
REGION_CONF = 0.25

SEVERITY_IMGSZ = 640
PLATE_IMGSZ = 640
REGION_IMGSZ = 640

MAX_PLATES_TO_KEEP = 2

ACCIDENT_SEVERITY_LABELS = ["minor", "moderate", "severe"]

SHOW_RESULTS = False
SAVE_OUTPUTS = True


# =========================================================
# 4) LOAD MODELS
# =========================================================

severity_model = YOLO(SEVERITY_MODEL_PATH)
plate_model = YOLO(PLATE_MODEL_PATH)
region_model = YOLO(REGION_MODEL_PATH)

print("Models loaded.")
print("Severity labels:", severity_model.names)
print("Plate labels:", plate_model.names)
print("Region labels:", region_model.names)


# =========================================================
# 5) LOAD OCR
# =========================================================

use_gpu = torch.cuda.is_available()
reader_en = easyocr.Reader(["en"], gpu=use_gpu)

print("EasyOCR loaded.")
print("CUDA available:", use_gpu)


# =========================================================
# 6) LABEL MAPS
# =========================================================

REGION_SHORT_NAMES = {
    "arabic-number": "AN",
    "arabic-text": "AT",
    "english-number": "EN",
    "english-text": "ET"
}

REGION_COLORS = {
    "AN": (255, 0, 0),
    "AT": (0, 255, 0),
    "EN": (0, 0, 255),
    "ET": (0, 165, 255)
}

ENGLISH_VALID_TEXT = "ABJDRSXTEGKLZNHUV"
ENGLISH_VALID_NUMBERS = "0123456789"

ENGLISH_NUM_TO_TEXT = {
    "5": "S",
    "0": "O",
    "1": "I",
    "8": "B",
    "2": "Z"
}

ENGLISH_TEXT_TO_NUM = {
    "S": "5",
    "O": "0",
    "I": "1",
    "B": "8",
    "Z": "2"
}

ENGLISH_TO_ARABIC = {
    "A": "ا", "B": "ب", "J": "ح",
    "D": "د", "R": "ر", "S": "س",
    "X": "ص", "T": "ط", "E": "ع",
    "G": "ق", "K": "ك", "L": "ل",
    "Z": "م", "N": "ن", "H": "ه",
    "U": "و", "V": "ى",
    "0": "٠", "1": "١", "2": "٢", "3": "٣",
    "4": "٤", "5": "٥", "6": "٦",
    "7": "٧", "8": "٨", "9": "٩"
}


# =========================================================
# 7) DISPLAY HELPER
# =========================================================

def display_bgr(img_bgr, title="Image", figsize=(9, 7)):
    if img_bgr is None or img_bgr.size == 0:
        print("Cannot display empty image:", title)
        return

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=figsize)
    plt.imshow(img_rgb)
    plt.title(title)
    plt.axis("off")
    plt.show()


# =========================================================
# 8) GENERAL HELPERS
# =========================================================

def crop_with_padding(img, box, pad=0):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = map(int, box)

    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)

    if x2 <= x1 or y2 <= y1:
        return np.array([])

    return img[y1:y2, x1:x2]


def box_center(box):
    x1, y1, x2, y2 = map(float, box)
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def is_box_center_inside_plate(region_box, plate_box):
    cx, cy = box_center(region_box)
    px1, py1, px2, py2 = map(float, plate_box)

    return px1 <= cx <= px2 and py1 <= cy <= py2


def clip_box_to_image(box, img_shape):
    h, w = img_shape[:2]
    x1, y1, x2, y2 = map(float, box)

    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w - 1, x2))
    y2 = max(0, min(h - 1, y2))

    return np.array([x1, y1, x2, y2], dtype=np.float32)


def is_valid_box(box):
    x1, y1, x2, y2 = map(float, box)
    return x2 > x1 and y2 > y1


def expand_box(box, expand_ratio=0.10):
    x1, y1, x2, y2 = map(float, box)

    w = x2 - x1
    h = y2 - y1

    dx = w * expand_ratio
    dy = h * expand_ratio

    return np.array([
        x1 - dx,
        y1 - dy,
        x2 + dx,
        y2 + dy
    ], dtype=np.float32)


def get_expand_ratio(region_name):
    if region_name == "EN":
        return 0.12
    elif region_name == "ET":
        return 0.08
    return 0.10


def draw_label(img, text, x, y, color=(0, 255, 0)):
    x = int(max(0, x))
    y = int(max(25, y))

    label_width = len(text) * 10 + 12

    cv2.rectangle(
        img,
        (x, max(0, y - 25)),
        (x + label_width, y + 5),
        color,
        -1
    )

    cv2.putText(
        img,
        text,
        (x + 5, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2,
        cv2.LINE_AA
    )


def average_conf(results):
    if not results:
        return 0.0

    return float(np.mean([r[2] for r in results]))


def sharpness_score(img_bgr):
    if img_bgr is None or img_bgr.size == 0:
        return 0.0

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()

    return float(min(variance / 500.0, 1.0))


# =========================================================
# 9) PLATE ANGLE + CANDIDATE SCORING
# =========================================================

def estimate_plate_angle(plate_crop_bgr):
    if plate_crop_bgr is None or plate_crop_bgr.size == 0:
        return None

    gray = cv2.cvtColor(plate_crop_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    h, w = plate_crop_bgr.shape[:2]

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=30,
        minLineLength=max(20, w // 4),
        maxLineGap=10
    )

    if lines is None:
        return None

    angles = []

    for line in lines:
        x1, y1, x2, y2 = line[0]

        dx = x2 - x1
        dy = y2 - y1

        if dx == 0:
            continue

        angle = np.degrees(np.arctan2(dy, dx))

        if -45 <= angle <= 45:
            angles.append(angle)

    if len(angles) == 0:
        return None

    return float(np.median(angles))


def compute_plate_candidate_score(plate_conf, plate_box, plate_angle, img_shape, plate_crop):
    h, w = img_shape[:2]
    image_area = max(1, w * h)

    x1, y1, x2, y2 = map(float, plate_box)
    plate_area = max(0, x2 - x1) * max(0, y2 - y1)

    size_score = min((plate_area / image_area) * 100.0, 1.0)

    if plate_angle is None:
        straightness_score = 0.5
    else:
        straightness_score = 1.0 - min(abs(plate_angle) / 30.0, 1.0)

    sharp_score = sharpness_score(plate_crop)

    final_score = (
        0.40 * float(plate_conf) +
        0.25 * float(size_score) +
        0.20 * float(straightness_score) +
        0.15 * float(sharp_score)
    )

    return final_score, size_score, straightness_score, sharp_score


# =========================================================
# 10) OCR HELPERS - ORIGINAL CROPS ONLY
# =========================================================

def clean_english_text(text):
    text = text.upper().replace(" ", "")
    text = "".join(ENGLISH_NUM_TO_TEXT.get(c, c) for c in text)
    return "".join(ch for ch in text if ch in ENGLISH_VALID_TEXT)


def clean_english_number(text):
    text = text.upper().replace(" ", "")
    text = "".join(ENGLISH_TEXT_TO_NUM.get(c, c) for c in text)
    return "".join(ch for ch in text if ch in ENGLISH_VALID_NUMBERS)


def map_english_to_arabic(text):
    return "".join(ENGLISH_TO_ARABIC.get(c, c) for c in text)


def reverse(text):
    return text[::-1]


def run_easyocr_original_only(region_name, crop_bgr):
    if crop_bgr is None or crop_bgr.size == 0:
        return "", "", 0.0

    if region_name == "ET":
        cleaner = clean_english_text
    elif region_name == "EN":
        cleaner = clean_english_number
    else:
        return "", "", 0.0

    results = reader_en.readtext(
        crop_bgr,
        detail=1,
        paragraph=False
    )

    raw_text = "".join(r[1] for r in results) if results else ""
    clean_text = cleaner(raw_text)
    conf = average_conf(results)

    return raw_text, clean_text, conf


# =========================================================
# 11) MODEL FUNCTIONS
# =========================================================

def run_severity_classification(img_bgr):
    results = severity_model.predict(
        source=img_bgr,
        imgsz=SEVERITY_IMGSZ,
        conf=SEVERITY_CONF,
        verbose=False
    )

    r = results[0]

    if hasattr(r, "probs") and r.probs is not None:
        class_id = int(r.probs.top1)
        conf = float(r.probs.top1conf)
        label = severity_model.names[class_id]
        return label, conf

    if r.boxes is not None and len(r.boxes) > 0:
        best_box = max(r.boxes, key=lambda b: float(b.conf[0]))
        class_id = int(best_box.cls[0])
        conf = float(best_box.conf[0])
        label = severity_model.names[class_id]
        return label, conf

    return "unknown", 0.0


def run_plate_detection(img_bgr):
    results = plate_model.predict(
        source=img_bgr,
        imgsz=PLATE_IMGSZ,
        conf=PLATE_CONF,
        verbose=False
    )[0]

    if results.boxes is None or len(results.boxes) == 0:
        return []

    boxes = results.boxes.xyxy.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy().astype(int)
    confs = results.boxes.conf.cpu().numpy()

    plates = []

    for box, cls_id, conf in zip(boxes, classes, confs):
        label = plate_model.names[int(cls_id)]

        plates.append({
            "box": box,
            "cls_id": int(cls_id),
            "label": label,
            "conf": float(conf)
        })

    return plates


def run_region_detection_full_image(img_bgr):
    """
    Important:
    The region model was trained on full images.
    So we run it on the full original image, not on plate crops.
    Then we filter regions whose center is inside the selected plate box.
    """

    results = region_model.predict(
        source=img_bgr,
        imgsz=REGION_IMGSZ,
        conf=REGION_CONF,
        verbose=False
    )[0]

    if results.boxes is None or len(results.boxes) == 0:
        return []

    boxes = results.boxes.xyxy.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy().astype(int)
    confs = results.boxes.conf.cpu().numpy()

    regions = []

    for box, cls_id, conf in zip(boxes, classes, confs):
        full_label = region_model.names[int(cls_id)]
        short_label = REGION_SHORT_NAMES.get(full_label, full_label)

        regions.append({
            "box": box,
            "cls_id": int(cls_id),
            "full_label": full_label,
            "short_label": short_label,
            "conf": float(conf)
        })

    return regions


# =========================================================
# 12) SEVERITY DECISION FOR THE WHOLE ACCIDENT
# =========================================================

def decide_final_severity(scan_rows):
    """
    Final severity logic:
    1. Count each image only once.
    2. Ignore unknown predictions.
    3. Accumulate confidence for each severity class.
    4. Choose the class with the highest accumulated confidence.

    This avoids one severe prediction forcing the whole accident to severe.
    """

    unique_image_rows = {}

    for row in scan_rows:
        image_path = row["image_path"]
        severity_label = str(row["severity_label"]).lower()

        if severity_label == "unknown":
            continue

        if image_path not in unique_image_rows:
            unique_image_rows[image_path] = row

    valid_rows = list(unique_image_rows.values())

    if len(valid_rows) == 0:
        return {
            "final_severity_label": "unknown",
            "final_severity_confidence": 0.0,
            "severity_decision_method": "no severity detections",
            "severity_class_scores": {}
        }

    class_scores = {}

    for row in valid_rows:
        label = str(row["severity_label"]).lower()
        conf = float(row["severity_confidence"])

        if label not in class_scores:
            class_scores[label] = 0.0

        class_scores[label] += conf

    final_label = max(class_scores, key=class_scores.get)
    final_score = class_scores[final_label]
    total_score = sum(class_scores.values())

    final_conf = final_score / total_score if total_score > 0 else 0.0

    return {
        "final_severity_label": final_label,
        "final_severity_confidence": round(final_conf, 4),
        "severity_decision_method": "accumulative confidence voting by unique image",
        "severity_class_scores": class_scores
    }


# =========================================================
# 13) STAGE 1: SCAN ONE IMAGE
# =========================================================

def scan_single_image(img_path, source_type="folder_image"):
    img_bgr = cv2.imread(img_path)

    if img_bgr is None:
        print("Could not read:", img_path)
        return [], None

    image_name = os.path.basename(img_path)

    original_save_path = os.path.join(
        ORIGINAL_IMAGES_DIR,
        f"{source_type}_{image_name}"
    )

    if SAVE_OUTPUTS:
        cv2.imwrite(original_save_path, img_bgr)

    severity_label, severity_conf = run_severity_classification(img_bgr)

    final_accident_detected_by_image = (
        severity_label.lower() in ACCIDENT_SEVERITY_LABELS
    )

    plates = run_plate_detection(img_bgr)

    scan_rows = []

    if len(plates) == 0:
        scan_rows.append({
            "accident_id": ACCIDENT_ID,
            "source_type": source_type,
            "image_name": image_name,
            "image_path": img_path,
            "saved_original_path": original_save_path,

            "severity_label": severity_label,
            "severity_confidence": round(severity_conf, 4),
            "image_accident_detected": final_accident_detected_by_image,

            "plate_detected": False,
            "plate_index_in_image": "",
            "plate_confidence": 0.0,
            "plate_box_xyxy": "",
            "plate_angle": "",
            "plate_candidate_score": 0.0,
            "plate_size_score": 0.0,
            "straightness_score": 0.0,
            "sharpness_score": 0.0,
            "plate_crop_path": ""
        })

        return scan_rows, img_bgr

    for idx, plate in enumerate(plates, start=1):
        plate_box = plate["box"]
        plate_conf = plate["conf"]

        plate_crop = crop_with_padding(img_bgr, plate_box, pad=5)

        plate_angle = estimate_plate_angle(plate_crop)

        plate_score, size_score, straight_score, sharp_score = compute_plate_candidate_score(
            plate_conf=plate_conf,
            plate_box=plate_box,
            plate_angle=plate_angle,
            img_shape=img_bgr.shape,
            plate_crop=plate_crop
        )

        base_name = os.path.splitext(image_name)[0]
        plate_crop_name = f"{ACCIDENT_ID}_{source_type}_{base_name}_plate_{idx}.jpg"
        plate_crop_path = os.path.join(PLATE_CROPS_DIR, plate_crop_name)

        if SAVE_OUTPUTS and plate_crop is not None and plate_crop.size > 0:
            cv2.imwrite(plate_crop_path, plate_crop)

        row = {
            "accident_id": ACCIDENT_ID,
            "source_type": source_type,
            "image_name": image_name,
            "image_path": img_path,
            "saved_original_path": original_save_path,

            "severity_label": severity_label,
            "severity_confidence": round(severity_conf, 4),
            "image_accident_detected": final_accident_detected_by_image,

            "plate_detected": True,
            "plate_index_in_image": idx,
            "plate_confidence": round(plate_conf, 4),
            "plate_box_xyxy": list(map(float, plate_box)),
            "plate_angle": round(plate_angle, 4) if plate_angle is not None else "",
            "plate_candidate_score": round(plate_score, 4),
            "plate_size_score": round(size_score, 4),
            "straightness_score": round(straight_score, 4),
            "sharpness_score": round(sharp_score, 4),
            "plate_crop_path": plate_crop_path
        }

        scan_rows.append(row)

    return scan_rows, img_bgr


# =========================================================
# 14) STAGE 1: SCAN TEST IMAGE + ALL FOLDER IMAGES
# =========================================================

def scan_all_images():
    valid_exts = ["*.jpg", "*.jpeg", "*.png", "*.webp"]

    image_items = []

    if os.path.exists(TEST_IMAGE_PATH):
        image_items.append({
            "path": TEST_IMAGE_PATH,
            "source_type": "test_image"
        })
    else:
        print("Warning: Test image not found:", TEST_IMAGE_PATH)

    folder_paths = []

    for ext in valid_exts:
        folder_paths.extend(glob.glob(os.path.join(INPUT_IMAGES_DIR, ext)))

    folder_paths = sorted(folder_paths)

    for path in folder_paths:
        image_items.append({
            "path": path,
            "source_type": "folder_image"
        })

    seen_paths = set()
    unique_items = []

    for item in image_items:
        if item["path"] not in seen_paths:
            unique_items.append(item)
            seen_paths.add(item["path"])

    print("=" * 80)
    print("SCANNING TEST IMAGE + ALL ACCIDENT IMAGES")
    print("=" * 80)
    print("Found images:", len(unique_items))

    all_scan_rows = []

    for i, item in enumerate(unique_items, start=1):
        img_path = item["path"]
        source_type = item["source_type"]

        print(
            f"[{i}/{len(unique_items)}] Scanning:",
            os.path.basename(img_path),
            "| source:",
            source_type
        )

        rows, _ = scan_single_image(
            img_path,
            source_type=source_type
        )

        if rows:
            all_scan_rows.extend(rows)

    return all_scan_rows


# =========================================================
# 15) SELECT BEST PLATE CANDIDATES
# =========================================================

def select_best_plate_candidates(scan_rows, max_plates=2):
    valid_candidates = [
        row for row in scan_rows
        if row["plate_detected"] is True
    ]

    if len(valid_candidates) == 0:
        return []

    sorted_candidates = sorted(
        valid_candidates,
        key=lambda x: x["plate_candidate_score"],
        reverse=True
    )

    return sorted_candidates[:max_plates]


# =========================================================
# 16) REGION + OCR FOR SELECTED PLATES
# =========================================================

def process_selected_plate(candidate, plate_number, final_severity_info):
    img_path = candidate["image_path"]
    img_name = candidate["image_name"]
    source_type = candidate["source_type"]
    base_name = os.path.splitext(img_name)[0]

    img_bgr = cv2.imread(img_path)

    if img_bgr is None:
        print("Could not read selected image:", img_path)
        return None

    annotated = img_bgr.copy()
    region_debug = img_bgr.copy()

    plate_box = np.array(candidate["plate_box_xyxy"], dtype=np.float32)
    px1, py1, px2, py2 = map(int, plate_box)

    cv2.rectangle(
        annotated,
        (px1, py1),
        (px2, py2),
        (255, 0, 0),
        2
    )

    cv2.rectangle(
        region_debug,
        (px1, py1),
        (px2, py2),
        (255, 0, 0),
        2
    )

    draw_label(
        annotated,
        f"Selected Plate {plate_number}: {candidate['plate_confidence']:.2f}",
        px1,
        max(25, py1 - 10),
        (255, 0, 0)
    )

    draw_label(
        annotated,
        f"Accident Severity: {final_severity_info['final_severity_label']} "
        f"{final_severity_info['final_severity_confidence']:.2f}",
        20,
        35,
        (0, 0, 255)
    )

    draw_label(
        annotated,
        f"Plate Score: {candidate['plate_candidate_score']}",
        20,
        70,
        (0, 255, 255)
    )

    # =====================================================
    # IMPORTANT:
    # Region model runs on the FULL image.
    # It was trained on full images.
    # =====================================================

    all_regions = run_region_detection_full_image(img_bgr)

    filtered_regions = []

    for region in all_regions:
        if region["short_label"] not in ["AN", "AT", "EN", "ET"]:
            continue

        if is_box_center_inside_plate(region["box"], plate_box):
            filtered_regions.append(region)

    best_regions = {}

    for region in filtered_regions:
        name = region["short_label"]

        if name not in best_regions or region["conf"] > best_regions[name]["conf"]:
            best_regions[name] = region

    for name, region in best_regions.items():
        x1, y1, x2, y2 = map(int, region["box"])
        color = REGION_COLORS.get(name, (255, 255, 255))

        cv2.rectangle(
            region_debug,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        draw_label(
            region_debug,
            f"{name}: {region['conf']:.2f}",
            x1,
            max(25, y1 - 10),
            color
        )

    # =====================================================
    # OCR on original ET and EN crops only
    # =====================================================

    region_outputs = {}
    english_text = ""
    english_number = ""

    for region_name in ["ET", "EN"]:

        if region_name not in best_regions:
            region_outputs[region_name] = {
                "raw_text": "",
                "clean_text": "",
                "det_conf": 0.0,
                "ocr_conf": 0.0,
                "crop_path": ""
            }
            continue

        region = best_regions[region_name]
        box = region["box"]

        expanded = expand_box(box, get_expand_ratio(region_name))
        expanded = clip_box_to_image(expanded, img_bgr.shape)

        if not is_valid_box(expanded):
            continue

        region_crop = crop_with_padding(img_bgr, expanded, pad=0)

        if region_crop is None or region_crop.size == 0:
            continue

        raw_text, clean_text, ocr_conf = run_easyocr_original_only(
            region_name,
            region_crop
        )

        crop_name = f"{ACCIDENT_ID}_{source_type}_{base_name}_plate{plate_number}_{region_name}_crop.jpg"
        crop_path = os.path.join(OCR_CROPS_DIR, crop_name)

        if SAVE_OUTPUTS:
            cv2.imwrite(crop_path, region_crop)

        region_outputs[region_name] = {
            "raw_text": raw_text,
            "clean_text": clean_text,
            "det_conf": region["conf"],
            "ocr_conf": ocr_conf,
            "crop_path": crop_path
        }

        if region_name == "ET":
            english_text = clean_text
        elif region_name == "EN":
            english_number = clean_text

    english_plate = f"{english_text}{english_number}"
    arabic_text = reverse(map_english_to_arabic(english_text))
    arabic_number = map_english_to_arabic(english_number)
    arabic_plate = f"{arabic_text}{arabic_number}"

    annotated_path = os.path.join(
        ANNOTATED_DIR,
        f"{ACCIDENT_ID}_{source_type}_{base_name}_plate{plate_number}_annotated.jpg"
    )

    region_debug_path = os.path.join(
        REGION_DEBUG_DIR,
        f"{ACCIDENT_ID}_{source_type}_{base_name}_plate{plate_number}_regions.jpg"
    )

    if SAVE_OUTPUTS:
        cv2.imwrite(annotated_path, annotated)
        cv2.imwrite(region_debug_path, region_debug)

    final_row = {
        "accident_id": ACCIDENT_ID,

        "final_severity_label": final_severity_info["final_severity_label"],
        "final_severity_confidence": final_severity_info["final_severity_confidence"],
        "severity_decision_method": final_severity_info["severity_decision_method"],
        "severity_class_scores": final_severity_info["severity_class_scores"],

        "selected_plate_number": plate_number,
        "source_type": source_type,
        "source_image_name": img_name,
        "source_image_path": img_path,

        "plate_confidence": candidate["plate_confidence"],
        "plate_box_xyxy": candidate["plate_box_xyxy"],
        "plate_angle": candidate["plate_angle"],
        "plate_candidate_score": candidate["plate_candidate_score"],
        "plate_size_score": candidate["plate_size_score"],
        "straightness_score": candidate["straightness_score"],
        "sharpness_score": candidate["sharpness_score"],
        "plate_crop_path": candidate["plate_crop_path"],

        "all_region_detections": ",".join([r["short_label"] for r in all_regions]),
        "filtered_regions_inside_plate": ",".join(best_regions.keys()),

        "ET_region_conf": round(region_outputs.get("ET", {}).get("det_conf", 0.0), 4),
        "EN_region_conf": round(region_outputs.get("EN", {}).get("det_conf", 0.0), 4),

        "ET_raw_ocr_text": region_outputs.get("ET", {}).get("raw_text", ""),
        "EN_raw_ocr_text": region_outputs.get("EN", {}).get("raw_text", ""),

        "ET_cleaned_text": english_text,
        "EN_cleaned_number": english_number,

        "english_plate": english_plate,
        "arabic_text_generated": arabic_text,
        "arabic_number_generated": arabic_number,
        "arabic_plate_generated": arabic_plate,

        "ET_ocr_conf": round(region_outputs.get("ET", {}).get("ocr_conf", 0.0), 4),
        "EN_ocr_conf": round(region_outputs.get("EN", {}).get("ocr_conf", 0.0), 4),

        "ET_crop_path": region_outputs.get("ET", {}).get("crop_path", ""),
        "EN_crop_path": region_outputs.get("EN", {}).get("crop_path", ""),

        "annotated_path": annotated_path,
        "region_debug_path": region_debug_path
    }

    print("\n" + "=" * 80)
    print(f"SELECTED PLATE {plate_number} RESULT")
    print("=" * 80)
    print("Source type:", source_type)
    print("Source image:", img_name)
    print("Plate score:", candidate["plate_candidate_score"])
    print("Filtered regions:", list(best_regions.keys()))
    print("ET raw OCR:", final_row["ET_raw_ocr_text"])
    print("EN raw OCR:", final_row["EN_raw_ocr_text"])
    print("English plate:", english_plate)
    print("Arabic plate:", arabic_plate)

    if SHOW_RESULTS:
        display_bgr(
            annotated,
            title=f"{source_type} | {img_name} | Selected Plate {plate_number}",
            figsize=(9, 7)
        )

        display_bgr(
            region_debug,
            title=f"{source_type} | {img_name} | Regions Inside Plate {plate_number}",
            figsize=(9, 7)
        )

    return final_row


# =========================================================
# 17) RUN FULL PIPELINE
# =========================================================

if __name__ == "__main__":
    scan_rows = scan_all_images()

    df_scan = pd.DataFrame(scan_rows)

    scan_csv_path = os.path.join(
        CSV_DIR,
        f"{ACCIDENT_ID}_scan_results.csv"
    )

    df_scan.to_csv(
        scan_csv_path,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nScan CSV saved to:", scan_csv_path)

    final_severity_info = decide_final_severity(scan_rows)

    print("\n" + "=" * 80)
    print("FINAL ACCIDENT SEVERITY")
    print("=" * 80)
    print("Final severity:", final_severity_info["final_severity_label"])
    print("Confidence:", final_severity_info["final_severity_confidence"])
    print("Method:", final_severity_info["severity_decision_method"])
    print("Class scores:", final_severity_info["severity_class_scores"])

    selected_plate_candidates = select_best_plate_candidates(
        scan_rows,
        max_plates=MAX_PLATES_TO_KEEP
    )

    print("\n" + "=" * 80)
    print("SELECTED PLATE CANDIDATES")
    print("=" * 80)
    print("Number of selected plates:", len(selected_plate_candidates))

    final_rows = []

    for plate_number, candidate in enumerate(selected_plate_candidates, start=1):
        final_row = process_selected_plate(
            candidate=candidate,
            plate_number=plate_number,
            final_severity_info=final_severity_info
        )

        if final_row is not None:
            final_rows.append(final_row)

    df_final = pd.DataFrame(final_rows)

    final_csv_path = os.path.join(
        CSV_DIR,
        f"{ACCIDENT_ID}_final_results.csv"
    )

    df_final.to_csv(
        final_csv_path,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n" + "=" * 80)
    print("PIPELINE FINISHED")
    print("=" * 80)

    print("Accident ID:", ACCIDENT_ID)
    print("Scan CSV:", scan_csv_path)
    print("Final CSV:", final_csv_path)
    print("Original images:", ORIGINAL_IMAGES_DIR)
    print("Annotated images:", ANNOTATED_DIR)
    print("Plate crops:", PLATE_CROPS_DIR)
    print("Region debug images:", REGION_DEBUG_DIR)
    print("OCR crops:", OCR_CROPS_DIR)

    print("\nTop plate candidates:")

    if len(df_scan) > 0 and "plate_candidate_score" in df_scan.columns:
        print(
            df_scan[df_scan["plate_detected"] == True]
            .sort_values(by="plate_candidate_score", ascending=False)
            .head(10)
        )

    print("\nFinal result:")
    print(df_final)



# # Draft


# =========================================================
# FULL PIPELINE:
# 1) Test image
# 2) All frames
# 3) Select best straight plate frame
# 4) Run OCR only on best frame
# 5) Accident model draws its own boxes using .plot()
# =========================================================

import os
import cv2
import glob
import torch
import easyocr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ultralytics import YOLO


# =========================================================
# 1) PATHS
# =========================================================

TEST_IMAGE_PATH = os.path.join(_HERE, "test.png")
INPUT_FRAMES_DIR = os.path.join(_HERE, "_no_frames")  # empty/non-existent: skips folder scan

OUTPUT_DIR = os.path.join(_HERE, "Output")

ACCIDENT_MODEL_PATH = os.path.join(_HERE, "accident_yolov8n_best (1).pt")
SEVERITY_MODEL_PATH = os.path.join(_HERE, "Severity_best_model.pt")
PLATE_MODEL_PATH = os.path.join(_HERE, "plate_best.pt")
REGION_MODEL_PATH = os.path.join(_HERE, "region_best.pt")


# =========================================================
# 2) OUTPUT FOLDERS
# =========================================================

ANNOTATED_DIR = os.path.join(OUTPUT_DIR, "annotated_frames")
BEST_FRAME_DIR = os.path.join(OUTPUT_DIR, "best_plate_frame")
PLATE_CROPS_DIR = os.path.join(OUTPUT_DIR, "plate_crops")
REGION_DEBUG_DIR = os.path.join(OUTPUT_DIR, "region_debug_images")
OCR_CROPS_DIR = os.path.join(OUTPUT_DIR, "ocr_region_crops")
OCR_VARIANTS_DIR = os.path.join(OUTPUT_DIR, "ocr_best_variants")

for folder in [
    OUTPUT_DIR,
    ANNOTATED_DIR,
    BEST_FRAME_DIR,
    PLATE_CROPS_DIR,
    REGION_DEBUG_DIR,
    OCR_CROPS_DIR,
    OCR_VARIANTS_DIR
]:
    os.makedirs(folder, exist_ok=True)


# =========================================================
# 3) SETTINGS
# =========================================================

ACCIDENT_CONF = 0.25
SEVERITY_CONF = 0.25
PLATE_CONF = 0.25
REGION_CONF = 0.25

ACCIDENT_IMGSZ = 640
SEVERITY_IMGSZ = 640
PLATE_IMGSZ = 640
REGION_IMGSZ = 640

# Check accident_model.names after loading.
# If it prints {0: 'Accident'}, keep this as 0.
# If it prints {1: 'Accident'}, change this to 1.
ACCIDENT_CLASS_ID = 0

SAVE_OUTPUTS = True
SHOW_TEST_IMAGE = False
SHOW_BEST_FRAME = False

ACCIDENT_SEVERITY_LABELS = ["minor", "moderate", "severe"]


# =========================================================
# 4) LOAD MODELS
# =========================================================

accident_model = YOLO(ACCIDENT_MODEL_PATH)
severity_model = YOLO(SEVERITY_MODEL_PATH)
plate_model = YOLO(PLATE_MODEL_PATH)
region_model = YOLO(REGION_MODEL_PATH)

print("Models loaded.")
print("Accident labels:", accident_model.names)
print("Severity labels:", severity_model.names)
print("Plate labels:", plate_model.names)
print("Region labels:", region_model.names)


# =========================================================
# 5) LOAD EASYOCR
# =========================================================

use_gpu = torch.cuda.is_available()
reader_en = easyocr.Reader(["en"], gpu=use_gpu)

print("EasyOCR loaded.")
print("CUDA available:", use_gpu)


# =========================================================
# 6) LABEL MAPS
# =========================================================

REGION_SHORT_NAMES = {
    "arabic-number": "AN",
    "arabic-text": "AT",
    "english-number": "EN",
    "english-text": "ET"
}

REGION_COLORS = {
    "AN": (255, 0, 0),
    "AT": (0, 255, 0),
    "EN": (0, 0, 255),
    "ET": (0, 165, 255)
}


# =========================================================
# 7) OCR RULES
# =========================================================

ENGLISH_VALID_TEXT = "ABJDRSXTEGKLZNHUV"
ENGLISH_VALID_NUMBERS = "0123456789"

ENGLISH_NUM_TO_TEXT = {
    "5": "S",
    "0": "O",
    "1": "I",
    "8": "B",
    "2": "Z"
}

ENGLISH_TEXT_TO_NUM = {
    "S": "5",
    "O": "0",
    "I": "1",
    "B": "8",
    "Z": "2"
}

ENGLISH_TO_ARABIC = {
    "A": "ا", "B": "ب", "J": "ح",
    "D": "د", "R": "ر", "S": "س",
    "X": "ص", "T": "ط", "E": "ع",
    "G": "ق", "K": "ك", "L": "ل",
    "Z": "م", "N": "ن", "H": "ه",
    "U": "و", "V": "ى",
    "0": "٠", "1": "١", "2": "٢", "3": "٣",
    "4": "٤", "5": "٥", "6": "٦",
    "7": "٧", "8": "٨", "9": "٩"
}


# =========================================================
# 8) DISPLAY HELPERS
# =========================================================

def display_bgr(img_bgr, title="Image", figsize=(8, 6)):
    if img_bgr is None or img_bgr.size == 0:
        print("Cannot display empty image:", title)
        return

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=figsize)
    plt.imshow(img_rgb)
    plt.title(title)
    plt.axis("off")
    plt.show()


def display_gray_or_bgr(img, title="Image", figsize=(5, 2)):
    if img is None or img.size == 0:
        print("Cannot display empty image:", title)
        return

    plt.figure(figsize=figsize)

    if len(img.shape) == 2:
        plt.imshow(img, cmap="gray")
    else:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    plt.title(title)
    plt.axis("off")
    plt.show()


# =========================================================
# 9) GENERAL HELPERS
# =========================================================

def crop_with_padding(img, box, pad=0):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = map(int, box)

    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)

    if x2 <= x1 or y2 <= y1:
        return np.array([])

    return img[y1:y2, x1:x2]


def box_center(box):
    x1, y1, x2, y2 = map(float, box)
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def is_box_center_inside_plate(region_box, plate_box):
    cx, cy = box_center(region_box)
    px1, py1, px2, py2 = map(float, plate_box)

    return px1 <= cx <= px2 and py1 <= cy <= py2


def clip_box_to_image(box, img_shape):
    h, w = img_shape[:2]
    x1, y1, x2, y2 = map(float, box)

    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w - 1, x2))
    y2 = max(0, min(h - 1, y2))

    return np.array([x1, y1, x2, y2], dtype=np.float32)


def is_valid_box(box):
    x1, y1, x2, y2 = map(float, box)
    return x2 > x1 and y2 > y1


def expand_box(box, expand_ratio=0.10):
    x1, y1, x2, y2 = map(float, box)

    w = x2 - x1
    h = y2 - y1

    dx = w * expand_ratio
    dy = h * expand_ratio

    return np.array([
        x1 - dx,
        y1 - dy,
        x2 + dx,
        y2 + dy
    ], dtype=np.float32)


def get_expand_ratio(region_name):
    if region_name == "EN":
        return 0.12
    elif region_name == "ET":
        return 0.08
    return 0.10


def draw_label(img, text, x, y, color=(0, 255, 0)):
    x = int(max(0, x))
    y = int(max(25, y))

    label_width = len(text) * 10 + 10

    cv2.rectangle(
        img,
        (x, max(0, y - 25)),
        (x + label_width, y + 5),
        color,
        -1
    )

    cv2.putText(
        img,
        text,
        (x + 5, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2,
        cv2.LINE_AA
    )


def average_conf(results):
    if not results:
        return 0.0
    return float(np.mean([r[2] for r in results]))


# =========================================================
# 10) PLATE ANGLE + CANDIDATE SCORING
# =========================================================

def estimate_plate_angle(plate_crop_bgr):
    if plate_crop_bgr is None or plate_crop_bgr.size == 0:
        return None

    gray = cv2.cvtColor(plate_crop_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    h, w = plate_crop_bgr.shape[:2]

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=30,
        minLineLength=max(20, w // 4),
        maxLineGap=10
    )

    if lines is None:
        return None

    angles = []

    for line in lines:
        x1, y1, x2, y2 = line[0]

        dx = x2 - x1
        dy = y2 - y1

        if dx == 0:
            continue

        angle = np.degrees(np.arctan2(dy, dx))

        if -45 <= angle <= 45:
            angles.append(angle)

    if len(angles) == 0:
        return None

    return float(np.median(angles))


def compute_plate_candidate_score(plate_conf, plate_box, plate_angle, img_shape):
    h, w = img_shape[:2]
    image_area = max(1, w * h)

    x1, y1, x2, y2 = map(float, plate_box)
    plate_area = max(0, x2 - x1) * max(0, y2 - y1)

    size_score = min((plate_area / image_area) * 100.0, 1.0)

    if plate_angle is None:
        straightness_score = 0.5
    else:
        straightness_score = 1.0 - min(abs(plate_angle) / 30.0, 1.0)

    final_score = (
        0.45 * float(plate_conf) +
        0.40 * float(straightness_score) +
        0.15 * float(size_score)
    )

    return final_score, size_score, straightness_score


# =========================================================
# 11) OCR HELPERS
# =========================================================

def clean_english_text(text):
    text = text.upper().replace(" ", "")
    text = "".join(ENGLISH_NUM_TO_TEXT.get(c, c) for c in text)
    return "".join(ch for ch in text if ch in ENGLISH_VALID_TEXT)


def clean_english_number(text):
    text = text.upper().replace(" ", "")
    text = "".join(ENGLISH_TEXT_TO_NUM.get(c, c) for c in text)
    return "".join(ch for ch in text if ch in ENGLISH_VALID_NUMBERS)


def map_english_to_arabic(text):
    return "".join(ENGLISH_TO_ARABIC.get(c, c) for c in text)


def reverse(text):
    return text[::-1]


def preprocess_for_ocr_variants(crop_img, scale=2):
    if crop_img is None or crop_img.size == 0:
        return {}

    h, w = crop_img.shape[:2]

    up = cv2.resize(
        crop_img,
        (max(1, w * scale), max(1, h * scale)),
        interpolation=cv2.INTER_CUBIC
    )

    if len(up.shape) == 3:
        gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    else:
        gray = up.copy()

    gray_eq = cv2.equalizeHist(gray)
    blur = cv2.bilateralFilter(gray_eq, 5, 75, 75)

    _, binary = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    inverted = 255 - binary

    adaptive = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )

    return {
        "gray": gray,
        "equalized": gray_eq,
        "binary": binary,
        "inverted": inverted,
        "adaptive": adaptive
    }


def score_ocr(text, conf, expected_len):
    if not text:
        return 0.0

    length_ratio = len(text) / expected_len

    if len(text) != expected_len:
        length_ratio *= 0.20

    if expected_len == 3 and len(text) <= 1:
        return 0.05 * conf

    if expected_len == 4 and len(text) <= 2:
        return 0.10 * conf

    return (
        0.55 * conf +
        0.30 * min(length_ratio, 1.0) +
        0.15 * (1.0 if len(text) == expected_len else 0.0)
    )


def run_easyocr_raw(region_name, crop_bgr):
    if crop_bgr is None or crop_bgr.size == 0:
        return "", 0.0

    if region_name == "ET":
        cleaner = clean_english_text
    elif region_name == "EN":
        cleaner = clean_english_number
    else:
        return "", 0.0

    results = reader_en.readtext(
        crop_bgr,
        detail=1,
        paragraph=False
    )

    raw_text = "".join(r[1] for r in results) if results else ""
    clean_text = cleaner(raw_text)
    conf = average_conf(results)

    return clean_text, conf


def ocr_simple_first_then_variants(region_name, crop_bgr):
    if crop_bgr is None or crop_bgr.size == 0:
        return "", 0.0, "none", None

    if region_name == "ET":
        expected_len = 3
    elif region_name == "EN":
        expected_len = 4
    else:
        return "", 0.0, "none", None

    best_text = ""
    best_conf = 0.0
    best_mode = "none"
    best_img = None
    best_score = 0.0

    text, conf = run_easyocr_raw(region_name, crop_bgr)
    score = score_ocr(text, conf, expected_len)

    best_text = text
    best_conf = conf
    best_mode = "original"
    best_img = crop_bgr
    best_score = score

    if len(text) == expected_len and conf >= 0.30:
        return best_text, best_conf, best_mode, best_img

    variants = preprocess_for_ocr_variants(crop_bgr, scale=2)

    for mode_name, img_variant in variants.items():
        if img_variant is None or img_variant.size == 0:
            continue

        text_v, conf_v = run_easyocr_raw(region_name, img_variant)
        score_v = score_ocr(text_v, conf_v, expected_len)

        if score_v > best_score:
            best_text = text_v
            best_conf = conf_v
            best_mode = mode_name
            best_img = img_variant
            best_score = score_v

    return best_text, best_conf, best_mode, best_img


# =========================================================
# 12) MODEL FUNCTIONS
# =========================================================

def run_accident_detection(img_bgr):
    """
    This checks accident detection only.
    The actual drawing is done later using accident_results[0].plot().
    """

    results = accident_model.predict(
        source=img_bgr,
        imgsz=ACCIDENT_IMGSZ,
        conf=ACCIDENT_CONF,
        verbose=False
    )

    r = results[0]

    accident_detected = False
    best_conf = 0.0

    if r.boxes is None or len(r.boxes) == 0:
        return accident_detected, best_conf

    for box in r.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        if cls_id == ACCIDENT_CLASS_ID:
            accident_detected = True
            best_conf = max(best_conf, conf)

    return accident_detected, best_conf


def get_accident_plotted_image(img_bgr):
    """
    This is the part you asked for:
    the accident model itself draws the bounding boxes.
    """

    accident_results = accident_model.predict(
        source=img_bgr,
        imgsz=ACCIDENT_IMGSZ,
        conf=ACCIDENT_CONF,
        verbose=False
    )

    plotted_img = accident_results[0].plot()

    return plotted_img


def run_severity_classification(img_bgr):
    results = severity_model.predict(
        source=img_bgr,
        imgsz=SEVERITY_IMGSZ,
        conf=SEVERITY_CONF,
        verbose=False
    )

    r = results[0]

    if hasattr(r, "probs") and r.probs is not None:
        class_id = int(r.probs.top1)
        conf = float(r.probs.top1conf)
        label = severity_model.names[class_id]
        return label, conf

    if r.boxes is not None and len(r.boxes) > 0:
        best_box = max(r.boxes, key=lambda b: float(b.conf[0]))
        class_id = int(best_box.cls[0])
        conf = float(best_box.conf[0])
        label = severity_model.names[class_id]
        return label, conf

    return "unknown", 0.0


def get_best_plate_box(img_bgr):
    results = plate_model.predict(
        source=img_bgr,
        imgsz=PLATE_IMGSZ,
        conf=PLATE_CONF,
        verbose=False
    )[0]

    if results.boxes is None or len(results.boxes) == 0:
        return None, 0.0

    boxes = results.boxes.xyxy.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()

    best_idx = int(np.argmax(confs))
    best_box = boxes[best_idx]
    best_conf = float(confs[best_idx])

    return best_box, best_conf


def get_top_n_plate_boxes(img_bgr, n=2):
    results = plate_model.predict(
        source=img_bgr,
        imgsz=PLATE_IMGSZ,
        conf=PLATE_CONF,
        verbose=False
    )[0]

    if results.boxes is None or len(results.boxes) == 0:
        return []

    boxes = results.boxes.xyxy.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()

    scored = []
    for box, conf in zip(boxes, confs):
        plate_crop = crop_with_padding(img_bgr, box, pad=5)
        angle = estimate_plate_angle(plate_crop) if plate_crop is not None and plate_crop.size > 0 else 0.0
        score, _, _ = compute_plate_candidate_score(
            plate_conf=float(conf),
            plate_box=box,
            plate_angle=angle,
            img_shape=img_bgr.shape
        )
        scored.append((box, float(conf), float(angle), float(score)))

    scored.sort(key=lambda x: x[3], reverse=True)
    return scored[:n]


def run_region_detection_full_image(img_bgr):
    results = region_model.predict(
        source=img_bgr,
        imgsz=REGION_IMGSZ,
        conf=REGION_CONF,
        verbose=False
    )[0]

    if results.boxes is None or len(results.boxes) == 0:
        return []

    boxes = results.boxes.xyxy.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy().astype(int)
    confs = results.boxes.conf.cpu().numpy()

    regions = []

    for box, cls_id, conf in zip(boxes, classes, confs):
        full_label = region_model.names[int(cls_id)]
        short_label = REGION_SHORT_NAMES.get(full_label, full_label)

        regions.append({
            "box": box,
            "cls_id": int(cls_id),
            "full_label": full_label,
            "short_label": short_label,
            "conf": float(conf)
        })

    return regions


# =========================================================
# 13) STAGE 1: ANALYZE FRAME WITHOUT OCR
# =========================================================

def analyze_frame_without_ocr(img_path, source_type="frame"):
    img_bgr = cv2.imread(img_path)

    if img_bgr is None:
        print("Could not read:", img_path)
        return None

    img_name = os.path.basename(img_path)

    accident_model_detected, accident_conf = run_accident_detection(img_bgr)

    severity_label, severity_conf = run_severity_classification(img_bgr)

    severity_says_accident = severity_label.lower() in ACCIDENT_SEVERITY_LABELS

    final_accident_detected = bool(
        accident_model_detected or severity_says_accident
    )

    plate_box, plate_conf = get_best_plate_box(img_bgr)
    plate_detected = plate_box is not None

    plate_angle = None
    plate_score = 0.0
    plate_size_score = 0.0
    straightness_score = 0.0

    if plate_detected:
        plate_crop = crop_with_padding(img_bgr, plate_box, pad=5)
        plate_angle = estimate_plate_angle(plate_crop)

        plate_score, plate_size_score, straightness_score = compute_plate_candidate_score(
            plate_conf=plate_conf,
            plate_box=plate_box,
            plate_angle=plate_angle,
            img_shape=img_bgr.shape
        )

    return {
        "source_type": source_type,
        "frame_name": img_name,
        "frame_path": img_path,

        "accident_model_detected": accident_model_detected,
        "accident_model_confidence": round(accident_conf, 4),

        "severity_label": severity_label,
        "severity_confidence": round(severity_conf, 4),
        "severity_says_accident": severity_says_accident,
        "final_accident_detected": final_accident_detected,

        "plate_detected": plate_detected,
        "plate_confidence": round(plate_conf, 4) if plate_detected else 0.0,
        "plate_box_xyxy": list(map(float, plate_box)) if plate_detected else "",

        "plate_angle": round(plate_angle, 4) if plate_angle is not None else "",
        "plate_candidate_score": round(plate_score, 4),
        "plate_size_score": round(plate_size_score, 4),
        "straightness_score": round(straightness_score, 4)
    }


# =========================================================
# 14) STAGE 2: REGION + OCR ON SELECTED FRAME
# =========================================================

def run_region_and_ocr_on_selected_frame(candidate, display=True, save_outputs=True):
    img_path = candidate["frame_path"]
    source_type = candidate["source_type"]
    img_name = candidate["frame_name"]
    base_name = os.path.splitext(img_name)[0]

    img_bgr = cv2.imread(img_path)

    if img_bgr is None:
        print("Could not read selected image:", img_path)
        return candidate

    # =====================================================
    # IMPORTANT:
    # Let the accident model itself draw its own boxes.
    # This is the same idea as: results[0].plot()
    # =====================================================

    annotated = get_accident_plotted_image(img_bgr)

    # This one is only for plate-region debugging
    region_debug = img_bgr.copy()

    # Add final accident decision label
    accident_status = "Accident Detected" if candidate["final_accident_detected"] else "No Accident"

    draw_label(
        annotated,
        accident_status,
        20,
        35,
        (0, 0, 255) if candidate["final_accident_detected"] else (0, 255, 0)
    )

    # =====================================================
    # PLATE DETECTION
    # =====================================================

    plate_detected = candidate["plate_detected"]

    if not plate_detected:
        print("No plate detected in selected frame. OCR skipped.")

        annotated_path = ""

        if save_outputs and SAVE_OUTPUTS:
            annotated_path = os.path.join(
                ANNOTATED_DIR,
                f"{source_type}_{base_name}_annotated.jpg"
            )
            cv2.imwrite(annotated_path, annotated)

        candidate.update({
            "annotated_path": annotated_path
        })

        if display:
            display_bgr(
                annotated,
                title=f"{img_name} | Accident Model Plot",
                figsize=(9, 7)
            )

        return candidate

    plate_box = np.array(candidate["plate_box_xyxy"], dtype=np.float32)
    plate_conf = candidate["plate_confidence"]

    px1, py1, px2, py2 = map(int, plate_box)

    cv2.rectangle(
        annotated,
        (px1, py1),
        (px2, py2),
        (255, 0, 0),
        2
    )

    cv2.rectangle(
        region_debug,
        (px1, py1),
        (px2, py2),
        (255, 0, 0),
        2
    )

    draw_label(
        annotated,
        f"Plate: {plate_conf:.2f}",
        px1,
        max(25, py1 - 10),
        (255, 0, 0)
    )

    draw_label(
        annotated,
        f"Best frame score: {candidate['plate_candidate_score']}",
        20,
        70,
        (0, 255, 255)
    )

    draw_label(
        annotated,
        f"Angle: {candidate['plate_angle']}",
        20,
        105,
        (0, 255, 255)
    )

    plate_crop = crop_with_padding(img_bgr, plate_box, pad=5)

    plate_crop_path = ""

    if save_outputs and SAVE_OUTPUTS and plate_crop is not None and plate_crop.size > 0:
        plate_crop_path = os.path.join(
            PLATE_CROPS_DIR,
            f"{source_type}_{base_name}_plate.jpg"
        )
        cv2.imwrite(plate_crop_path, plate_crop)

    # =====================================================
    # REGION DETECTION
    # =====================================================

    all_regions = run_region_detection_full_image(img_bgr)

    filtered_regions = []

    for region in all_regions:
        if region["short_label"] not in ["AN", "AT", "EN", "ET"]:
            continue

        if is_box_center_inside_plate(region["box"], plate_box):
            filtered_regions.append(region)

    best_regions = {}

    for region in filtered_regions:
        name = region["short_label"]

        if name not in best_regions or region["conf"] > best_regions[name]["conf"]:
            best_regions[name] = region

    for name, region in best_regions.items():
        x1, y1, x2, y2 = map(int, region["box"])
        color = REGION_COLORS.get(name, (255, 255, 255))

        cv2.rectangle(
            region_debug,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        draw_label(
            region_debug,
            f"{name}: {region['conf']:.2f}",
            x1,
            max(25, y1 - 10),
            color
        )

    # =====================================================
    # OCR ON ET AND EN
    # =====================================================

    region_outputs = {}
    english_text = ""
    english_number = ""

    for region_name in ["ET", "EN"]:

        if region_name not in best_regions:
            region_outputs[region_name] = {
                "text": "",
                "det_conf": 0.0,
                "ocr_conf": 0.0,
                "ocr_mode": "none",
                "crop_path": "",
                "best_variant_path": "",
                "crop_img": None,
                "best_img": None
            }
            continue

        region = best_regions[region_name]
        box = region["box"]

        expanded = expand_box(box, get_expand_ratio(region_name))
        expanded = clip_box_to_image(expanded, img_bgr.shape)

        if not is_valid_box(expanded):
            continue

        region_crop = crop_with_padding(img_bgr, expanded, pad=0)

        if region_crop is None or region_crop.size == 0:
            continue

        text, ocr_conf, ocr_mode, best_img = ocr_simple_first_then_variants(
            region_name,
            region_crop
        )

        crop_path = ""
        best_variant_path = ""

        if save_outputs and SAVE_OUTPUTS:
            crop_path = os.path.join(
                OCR_CROPS_DIR,
                f"{source_type}_{base_name}_{region_name}_crop.jpg"
            )
            cv2.imwrite(crop_path, region_crop)

            if best_img is not None and best_img.size > 0:
                best_variant_path = os.path.join(
                    OCR_VARIANTS_DIR,
                    f"{source_type}_{base_name}_{region_name}_best_{ocr_mode}.jpg"
                )
                cv2.imwrite(best_variant_path, best_img)

        region_outputs[region_name] = {
            "text": text,
            "det_conf": region["conf"],
            "ocr_conf": ocr_conf,
            "ocr_mode": ocr_mode,
            "crop_path": crop_path,
            "best_variant_path": best_variant_path,
            "crop_img": region_crop,
            "best_img": best_img
        }

        if region_name == "ET":
            english_text = text
        elif region_name == "EN":
            english_number = text

    english_plate = f"{english_text}{english_number}"
    arabic_text = reverse(map_english_to_arabic(english_text))
    arabic_number = map_english_to_arabic(english_number)
    arabic_plate = f"{arabic_text}{arabic_number}"

    # =====================================================
    # SAVE OUTPUTS
    # =====================================================

    annotated_path = ""
    region_debug_path = ""

    if save_outputs and SAVE_OUTPUTS:
        annotated_path = os.path.join(
            ANNOTATED_DIR,
            f"{source_type}_{base_name}_annotated.jpg"
        )

        region_debug_path = os.path.join(
            REGION_DEBUG_DIR,
            f"{source_type}_{base_name}_regions.jpg"
        )

        cv2.imwrite(annotated_path, annotated)
        cv2.imwrite(region_debug_path, region_debug)

    candidate.update({
        "all_region_detections": ",".join([r["short_label"] for r in all_regions]),
        "filtered_regions_inside_plate": ",".join(best_regions.keys()),

        "english_text_ET": english_text,
        "english_number_EN": english_number,
        "english_plate": english_plate,

        "arabic_text_generated": arabic_text,
        "arabic_number_generated": arabic_number,
        "arabic_plate_generated": arabic_plate,

        "ET_ocr_conf": round(region_outputs.get("ET", {}).get("ocr_conf", 0.0), 4),
        "EN_ocr_conf": round(region_outputs.get("EN", {}).get("ocr_conf", 0.0), 4),

        "ET_ocr_mode": region_outputs.get("ET", {}).get("ocr_mode", "none"),
        "EN_ocr_mode": region_outputs.get("EN", {}).get("ocr_mode", "none"),

        "annotated_path": annotated_path,
        "region_debug_path": region_debug_path,
        "plate_crop_path": plate_crop_path,

        "ET_crop_path": region_outputs.get("ET", {}).get("crop_path", ""),
        "EN_crop_path": region_outputs.get("EN", {}).get("crop_path", ""),

        "ET_best_variant_path": region_outputs.get("ET", {}).get("best_variant_path", ""),
        "EN_best_variant_path": region_outputs.get("EN", {}).get("best_variant_path", "")
    })

    print("\nOCR RESULT FOR SELECTED FRAME")
    print("Selected image:", img_name)
    print("Final accident detected:", candidate["final_accident_detected"])
    print("Accident model detected:", candidate["accident_model_detected"])
    print("Accident model confidence:", candidate["accident_model_confidence"])
    print("Severity label:", candidate["severity_label"])
    print("Severity confidence:", candidate["severity_confidence"])
    print("Plate angle:", candidate["plate_angle"])
    print("Plate score:", candidate["plate_candidate_score"])
    print("Filtered regions:", list(best_regions.keys()))
    print("English plate:", english_plate)
    print("Arabic plate:", arabic_plate)
    print("ET OCR mode:", candidate["ET_ocr_mode"], "| conf:", candidate["ET_ocr_conf"])
    print("EN OCR mode:", candidate["EN_ocr_mode"], "| conf:", candidate["EN_ocr_conf"])

    if display:
        display_bgr(
            annotated,
            title=f"{img_name} | Accident Model Plot + Plate Result",
            figsize=(9, 7)
        )

        if plate_crop is not None and plate_crop.size > 0:
            display_bgr(
                plate_crop,
                title=f"{img_name} | Plate Crop",
                figsize=(6, 3)
            )

        display_bgr(
            region_debug,
            title=f"{img_name} | Filtered Region Boxes",
            figsize=(9, 7)
        )

        for region_name in ["ET", "EN"]:
            crop_img = region_outputs.get(region_name, {}).get("crop_img", None)
            best_img = region_outputs.get(region_name, {}).get("best_img", None)
            mode = region_outputs.get(region_name, {}).get("ocr_mode", "none")

            if crop_img is not None and crop_img.size > 0:
                display_bgr(
                    crop_img,
                    title=f"{img_name} | OCR Crop {region_name}",
                    figsize=(5, 2)
                )

            if best_img is not None and best_img.size > 0:
                display_gray_or_bgr(
                    best_img,
                    title=f"{img_name} | Best OCR Image {region_name} ({mode})",
                    figsize=(5, 2)
                )

    return candidate


# =========================================================
# 14b) FLASK-FRIENDLY ENTRY POINT (top-2 plates per image)
# =========================================================

def run_pipeline_on_image(image_path):
    """
    Run severity + top-2 plate OCR on a single image.
    Returns a flat dict ready for the API response / Supabase insert.
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    severity_label, severity_conf = run_severity_classification(img_bgr)

    top_plates = get_top_n_plate_boxes(img_bgr, n=2)
    all_regions = run_region_detection_full_image(img_bgr)

    plates_out = []
    for plate_box, plate_conf, _angle, _score in top_plates:
        regions_in_plate = [
            r for r in all_regions
            if r["short_label"] in ["AN", "AT", "EN", "ET"]
            and is_box_center_inside_plate(r["box"], plate_box)
        ]

        best_regions = {}
        for region in regions_in_plate:
            name = region["short_label"]
            if name not in best_regions or region["conf"] > best_regions[name]["conf"]:
                best_regions[name] = region

        english_text = ""
        english_number = ""
        ocr_confs = []

        for region_name in ["ET", "EN"]:
            if region_name not in best_regions:
                continue
            region = best_regions[region_name]
            expanded = expand_box(region["box"], get_expand_ratio(region_name))
            expanded = clip_box_to_image(expanded, img_bgr.shape)
            if not is_valid_box(expanded):
                continue
            region_crop = crop_with_padding(img_bgr, expanded, pad=0)
            if region_crop is None or region_crop.size == 0:
                continue
            text, ocr_conf, _, _ = ocr_simple_first_then_variants(region_name, region_crop)
            ocr_confs.append(float(ocr_conf))
            if region_name == "ET":
                english_text = text
            elif region_name == "EN":
                english_number = text

        english_plate = f"{english_text}{english_number}"
        # Arabic is generated by mapping the English OCR result through a fixed table.
        arabic_text_part = reverse(map_english_to_arabic(english_text))
        arabic_number_part = map_english_to_arabic(english_number)
        # Space between each Arabic letter for readability (numbers stay joined).
        arabic_text_spaced = " ".join(arabic_text_part)
        arabic_plate = f"{arabic_text_spaced} {arabic_number_part}".strip()

        # Single combined column: "ZER1048 | زعر١٠٤٨" (skip separator if either side is empty)
        if english_plate and arabic_plate:
            plate_text = f"{english_plate} | {arabic_plate}"
        else:
            plate_text = english_plate or arabic_plate

        avg_ocr_conf = sum(ocr_confs) / len(ocr_confs) if ocr_confs else 0.0
        # Combined confidence: plate-detection × OCR (0–100). If no OCR, use detection only.
        if avg_ocr_conf > 0:
            combined = plate_conf * avg_ocr_conf * 100
        else:
            combined = plate_conf * 100
        plates_out.append({
            "text": plate_text,
            "confidence": round(combined, 2)
        })

    while len(plates_out) < 2:
        plates_out.append({"text": "", "confidence": 0.0})

    # Severity casing must match the Supabase CHECK constraint (Title case)
    severity_clean = severity_label.capitalize() if severity_label else "Unknown"

    return {
        "severity": severity_clean,
        "severity_confidence": round(float(severity_conf) * 100, 2),
        "plate_1_text": plates_out[0]["text"],
        "plate_1_confidence": plates_out[0]["confidence"],
        "plate_2_text": plates_out[1]["text"],
        "plate_2_confidence": plates_out[1]["confidence"],
    }


# =========================================================
# 15) RUN TEST IMAGE (script mode only)
# =========================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("RUNNING TEST IMAGE")
    print("=" * 80)

    test_candidate = analyze_frame_without_ocr(
        TEST_IMAGE_PATH,
        source_type="test"
    )

    test_final = None

    if test_candidate is not None:
        print("Test image analysis:")
        print(test_candidate)

        test_final = run_region_and_ocr_on_selected_frame(
            test_candidate,
            display=SHOW_TEST_IMAGE,
            save_outputs=True
        )

    # =========================================================
    # 16) RUN ALL FRAMES: FIRST PASS
    # =========================================================

    valid_exts = ["*.jpg", "*.jpeg", "*.png", "*.webp"]

    frame_paths = []

    for ext in valid_exts:
        frame_paths.extend(glob.glob(os.path.join(INPUT_FRAMES_DIR, ext)))

    frame_paths = sorted(frame_paths)

    print("\n" + "=" * 80)
    print("SCANNING ALL FRAMES")
    print("=" * 80)
    print("Found frames:", len(frame_paths))

    frame_candidates = []

    for i, frame_path in enumerate(frame_paths, start=1):
        print(f"[{i}/{len(frame_paths)}] Scanning:", os.path.basename(frame_path))

        candidate = analyze_frame_without_ocr(
            frame_path,
            source_type="frame"
        )

        if candidate is not None:
            frame_candidates.append(candidate)

    # =========================================================
    # 17) SELECT BEST FRAME FOR OCR
    # =========================================================

    df_scan = pd.DataFrame(frame_candidates)

    scan_csv_path = os.path.join(
        OUTPUT_DIR,
        "frame_scan_candidates.csv"
    )

    df_scan.to_csv(
        scan_csv_path,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nFrame scan saved to:", scan_csv_path)

    valid_plate_candidates = [
        c for c in frame_candidates
        if c["plate_detected"] is True
    ]

    best_frame_result = None

    if len(valid_plate_candidates) == 0:
        print("\nNo plate was detected in any frame. OCR cannot run on frames.")

    else:
        best_candidate = max(
            valid_plate_candidates,
            key=lambda c: c["plate_candidate_score"]
        )

        print("\n" + "=" * 80)
        print("BEST FRAME SELECTED FOR OCR")
        print("=" * 80)

        print("Best frame:", best_candidate["frame_name"])
        print("Final accident detected:", best_candidate["final_accident_detected"])
        print("Accident model detected:", best_candidate["accident_model_detected"])
        print("Accident confidence:", best_candidate["accident_model_confidence"])
        print("Plate confidence:", best_candidate["plate_confidence"])
        print("Plate angle:", best_candidate["plate_angle"])
        print("Straightness score:", best_candidate["straightness_score"])
        print("Plate size score:", best_candidate["plate_size_score"])
        print("Final candidate score:", best_candidate["plate_candidate_score"])

        best_frame_result = run_region_and_ocr_on_selected_frame(
            best_candidate,
            display=SHOW_BEST_FRAME,
            save_outputs=True
        )

    # =========================================================
    # 18) SAVE FINAL RESULTS
    # =========================================================

    final_rows = []

    if test_final is not None:
        final_rows.append(test_final)

    if best_frame_result is not None:
        final_rows.append(best_frame_result)

    df_final = pd.DataFrame(final_rows)

    final_csv_path = os.path.join(
        OUTPUT_DIR,
        "final_best_frame_ocr_results.csv"
    )

    df_final.to_csv(
        final_csv_path,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n" + "=" * 80)
    print("PIPELINE FINISHED")
    print("=" * 80)

    print("Frame scan CSV:", scan_csv_path)
    print("Final OCR CSV:", final_csv_path)
    print("Annotated images:", ANNOTATED_DIR)
    print("Best frame outputs:", BEST_FRAME_DIR)
    print("Plate crops:", PLATE_CROPS_DIR)
    print("Region debug images:", REGION_DEBUG_DIR)
    print("OCR crops:", OCR_CROPS_DIR)
    print("OCR best variants:", OCR_VARIANTS_DIR)

    print("\nTop 10 plate candidates:")

    if len(df_scan) > 0:
        print(
            df_scan.sort_values(
                by="plate_candidate_score",
                ascending=False
            ).head(10)
        )

    print("\nFinal OCR result:")
    print(df_final)

    # Also demonstrate the Flask-style summary on the test image
    print("\n" + "=" * 80)
    print("FLASK SUMMARY (top-2 plates)")
    print("=" * 80)
    try:
        print(run_pipeline_on_image(TEST_IMAGE_PATH))
    except Exception as e:
        print("run_pipeline_on_image failed:", e)