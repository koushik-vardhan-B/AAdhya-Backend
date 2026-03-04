"""
OCR Module — Extract text from screenshot images
Uses EasyOCR with Indian language support + smart image preprocessing.

Supports: English, Hindi, Telugu, Tamil, Kannada, Bengali, Marathi
Handles: Dark mode screenshots, chat bubbles, low contrast
"""

import io
import numpy as np
import easyocr
from PIL import Image, ImageEnhance, ImageOps

# Initialize EasyOCR readers
# Telugu can only pair with English (EasyOCR limitation)
# So we use two readers: Telugu+English (primary) and Hindi+English (fallback)
print("📸 Loading OCR engine (English + Telugu)...")
reader_te = easyocr.Reader(["en", "te"], gpu=False, verbose=False)
print("✅ Telugu OCR ready!")

print("📸 Loading OCR engine (English + Hindi)...")
reader_hi = easyocr.Reader(["en", "hi"], gpu=False, verbose=False)
print("✅ Hindi OCR ready!")


def _preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Smart preprocessing for phone screenshots.
    Detects dark mode and inverts, enhances contrast for OCR.
    """
    img = Image.open(io.BytesIO(image_bytes))

    # Convert to RGB
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Scale up small images
    width, height = img.size
    if width < 800:
        scale = 800 / width
        img = img.resize((int(width * scale), int(height * scale)), Image.LANCZOS)

    # Detect dark mode: check average brightness of the image
    grayscale = img.convert("L")
    avg_brightness = np.array(grayscale).mean()

    if avg_brightness < 100:
        # Dark mode screenshot — invert colors (dark bg → light bg)
        img = ImageOps.invert(img)

    # Enhance for OCR
    img = ImageEnhance.Contrast(img).enhance(1.5)
    img = ImageEnhance.Sharpness(img).enhance(2.0)

    return np.array(img)


def _run_ocr(reader, image_data, label="") -> dict:
    """Run OCR with a specific reader and return structured result."""
    results = reader.readtext(image_data)

    if not results:
        return {"text": "", "confidence": 0.0, "lines": [], "success": False}

    lines = []
    total_confidence = 0.0

    for bbox, text, confidence in results:
        cleaned = text.strip()
        if cleaned and len(cleaned) > 1:
            lines.append({"text": cleaned, "confidence": round(confidence, 4)})
            total_confidence += confidence

    if not lines:
        return {"text": "", "confidence": 0.0, "lines": [], "success": False}

    full_text = " ".join(line["text"] for line in lines)
    avg_confidence = round(total_confidence / len(lines), 4)

    return {"text": full_text, "confidence": avg_confidence, "lines": lines, "success": True}


def extract_text_from_bytes(image_bytes: bytes) -> dict:
    """
    Extract text from image bytes with smart preprocessing.
    Tries both Telugu+English and Hindi+English readers, picks the best.
    """
    try:
        processed = _preprocess_image(image_bytes)

        # Try both readers, pick the one with higher confidence
        result_te = _run_ocr(reader_te, processed, "Telugu")
        result_hi = _run_ocr(reader_hi, processed, "Hindi")

        # Pick best result by confidence
        if result_te["success"] and result_hi["success"]:
            best = result_te if result_te["confidence"] >= result_hi["confidence"] else result_hi
            lang = "te" if result_te["confidence"] >= result_hi["confidence"] else "hi"
        elif result_te["success"]:
            best = result_te
            lang = "te"
        elif result_hi["success"]:
            best = result_hi
            lang = "hi"
        else:
            return {
                "text": "", "confidence": 0.0, "lines": [],
                "language_detected": "unknown", "success": False,
                "error": "No text found in image",
            }

        return {
            "text": best["text"],
            "confidence": best["confidence"],
            "lines": best["lines"],
            "language_detected": lang,
            "success": True,
            "error": None,
        }

    except Exception as e:
        return {
            "text": "", "confidence": 0.0, "lines": [],
            "language_detected": "unknown", "success": False,
            "error": str(e),
        }


def extract_text_from_image(image_path: str) -> dict:
    """Extract text from a file path (for testing)."""
    with open(image_path, "rb") as f:
        return extract_text_from_bytes(f.read())
