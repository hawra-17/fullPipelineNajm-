# =========================================================
# CLOUD ACCIDENT PIPELINE - IMPROVED EASYOCR VERSION
# =========================================================
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
# - OCR runs only on ET and EN crops
#
# OCR improvement added:
# - Resize crop x3
# - Convert to grayscale
# - Use EasyOCR allowlist
# - Lower region confidence to 0.20
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
OCR_PROCESSED_DIR = os.path.join(ACCIDENT_OUTPUT_DIR, "ocr_processed_crops")
CSV_DIR = os.path.join(ACCIDENT_OUTPUT_DIR, "csv_results")

for folder in [
    ACCIDENT_OUTPUT_DIR,
    ORIGINAL_IMAGES_DIR,
    ANNOTATED_DIR,
    PLATE_CROPS_DIR,
    REGION_DEBUG_DIR,
    OCR_CROPS_DIR,
    OCR_PROCESSED_DIR,
    CSV_DIR
]:
    os.makedirs(folder, exist_ok=True)


# =========================================================
# 3) SETTINGS
# =========================================================

SEVERITY_CONF = 0.25
PLATE_CONF = 0.25

# Changed from 0.25 to 0.20
# This keeps more possible ET/EN region boxes.
REGION_CONF = 0.20

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
# 5) LOAD EASYOCR
# =========================================================

use_gpu = torch.cuda.is_available()

reader_en = easyocr.Reader(
    ["en"],
    gpu=use_gpu,
    verbose=False
)

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
# 10) OCR HELPERS - IMPROVED EASYOCR
# =========================================================

def clean_english_text(text):
    text = str(text).upper().replace(" ", "")
    text = "".join(ENGLISH_NUM_TO_TEXT.get(c, c) for c in text)
    return "".join(ch for ch in text if ch in ENGLISH_VALID_TEXT)


def clean_english_number(text):
    text = str(text).upper().replace(" ", "")
    text = "".join(ENGLISH_TEXT_TO_NUM.get(c, c) for c in text)
    return "".join(ch for ch in text if ch in ENGLISH_VALID_NUMBERS)


def map_english_to_arabic(text):
    return "".join(ENGLISH_TO_ARABIC.get(c, c) for c in text)


def reverse(text):
    return text[::-1]


def preprocess_for_easyocr(crop_bgr):
    """
    Improves OCR input without changing the Najm pipeline workflow.

    Steps:
    1) Resize crop x3
    2) Convert to grayscale
    """

    if crop_bgr is None or crop_bgr.size == 0:
        return None

    resized = cv2.resize(
        crop_bgr,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC
    )

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    return gray


def run_easyocr_original_only(region_name, crop_bgr):
    """
    OCR still runs only on ET and EN region crops.

    This version only improves EasyOCR by adding:
    - crop resizing x3
    - grayscale conversion
    - EasyOCR allowlist
    """

    if crop_bgr is None or crop_bgr.size == 0:
        return "", "", 0.0

    if region_name == "ET":
        cleaner = clean_english_text
        allowlist = ENGLISH_VALID_TEXT + ENGLISH_VALID_NUMBERS

    elif region_name == "EN":
        cleaner = clean_english_number
        allowlist = ENGLISH_VALID_NUMBERS + ENGLISH_VALID_TEXT

    else:
        return "", "", 0.0

    processed_crop = preprocess_for_easyocr(crop_bgr)

    if processed_crop is None:
        return "", "", 0.0

    results = reader_en.readtext(
        processed_crop,
        detail=1,
        paragraph=False,
        allowlist=allowlist
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
    The region model runs on the full original image.
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

    # Region model runs on the full image.
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
    # OCR on ET and EN crops only
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
                "crop_path": "",
                "processed_crop_path": ""
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

        processed_crop_name = f"{ACCIDENT_ID}_{source_type}_{base_name}_plate{plate_number}_{region_name}_processed.jpg"
        processed_crop_path = os.path.join(OCR_PROCESSED_DIR, processed_crop_name)

        if SAVE_OUTPUTS:
            cv2.imwrite(crop_path, region_crop)

            processed_crop = preprocess_for_easyocr(region_crop)
            if processed_crop is not None:
                cv2.imwrite(processed_crop_path, processed_crop)

        region_outputs[region_name] = {
            "raw_text": raw_text,
            "clean_text": clean_text,
            "det_conf": region["conf"],
            "ocr_conf": ocr_conf,
            "crop_path": crop_path,
            "processed_crop_path": processed_crop_path
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

        "ET_processed_crop_path": region_outputs.get("ET", {}).get("processed_crop_path", ""),
        "EN_processed_crop_path": region_outputs.get("EN", {}).get("processed_crop_path", ""),

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

        for region_name in ["ET", "EN"]:
            crop_path = region_outputs.get(region_name, {}).get("crop_path", "")
            processed_path = region_outputs.get(region_name, {}).get("processed_crop_path", "")

            if crop_path and os.path.exists(crop_path):
                crop_img = cv2.imread(crop_path)
                display_bgr(
                    crop_img,
                    title=f"{img_name} | OCR Crop {region_name}",
                    figsize=(5, 2)
                )

            if processed_path and os.path.exists(processed_path):
                processed_img = cv2.imread(processed_path, cv2.IMREAD_GRAYSCALE)
                display_gray_or_bgr(
                    processed_img,
                    title=f"{img_name} | Processed OCR Crop {region_name}",
                    figsize=(5, 2)
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
    print("Processed OCR crops:", OCR_PROCESSED_DIR)

    print("\nTop plate candidates:")

    if len(df_scan) > 0 and "plate_candidate_score" in df_scan.columns:
        plate_rows = df_scan[df_scan["plate_detected"] == True]

        if len(plate_rows) > 0:
            print(
                plate_rows
                .sort_values(by="plate_candidate_score", ascending=False)
                .head(10)
            )
        else:
            print("No plate candidates found.")

    print("\nFinal result:")
    print(df_final)
