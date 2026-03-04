import sys
import os
import time
import logging
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel, field_validator
from database import (
    save_scan, get_recent_scans, get_scan_by_id,
    save_to_community, get_community_feed,
    upsert_keywords, get_top_keywords
)

# Add detection_layer to Python path so we can import layer1, layer2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "detection_layer"))
from layer1 import run_layer1, get_pattern_stats
from layer2 import run_layer2, _detect_language, EXPLANATIONS, PREVENTION_TIPS_I18N
from ocr import extract_text_from_bytes

# Setup logger for response time tracking
logger = logging.getLogger("aadhya")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / Response models with validation
# ---------------------------------------------------------------------------
class ScanResult(BaseModel):
    message: str
    language: str = "en"
    result: dict

class PredictRequest(BaseModel):
    message: str
    language: str = "en"

    @field_validator("message")
    @classmethod
    def message_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Message cannot be empty or whitespace only")
        if len(v) > 5000:
            raise ValueError("Message too long (max 5000 characters)")
        return v.strip()


# ---------------------------------------------------------------------------
# Prevention tips per fraud type
# ---------------------------------------------------------------------------
PREVENTION_TIPS = {
    "UPI Fraud": [
        "Never share OTP, PIN, or CVV with anyone.",
        "Banks never ask for UPI PIN via SMS or call.",
        "Always verify the sender before making any payment.",
        "Use official banking apps downloaded from Play Store/App Store.",
        "Report suspicious UPI requests to your bank immediately.",
    ],
    "Lottery Scam": [
        "You cannot win a lottery you never entered.",
        "Never pay 'processing fees' to claim a prize.",
        "Delete messages claiming you've won prizes from unknown sources.",
        "Legitimate lotteries never ask for advance payments.",
        "Report such messages to cybercrime.gov.in.",
    ],
    "Job Scam": [
        "Legitimate companies never charge registration fees.",
        "Be wary of unrealistic salary promises.",
        "Verify company details on official websites before applying.",
        "Never pay money to get a job offer.",
        "Report fake job offers to the Cyber Crime helpline 1930.",
    ],
    "Phishing": [
        "Never click on unknown or suspicious links.",
        "Always verify URLs before entering credentials.",
        "Banks and services never ask for passwords via SMS.",
        "Look for HTTPS and correct domain spelling in URLs.",
        "Enable two-factor authentication on all accounts.",
    ],
    "Others": [
        "Be cautious with unsolicited messages from unknown senders.",
        "Never share personal or financial information via SMS.",
        "When in doubt, contact the organization directly using official channels.",
        "Report suspicious messages to Cyber Crime helpline 1930.",
    ],
}


# ---------------------------------------------------------------------------
# 🔍 PREDICT — Main detection endpoint
# ---------------------------------------------------------------------------
@router.post("/predict")
def predict(payload: PredictRequest):
    """
    Analyze a message for fraud using Layer 1 + Layer 2 detection pipeline.
    Auto-saves results to Supabase.
    """
    pipeline_start = time.time()

    try:
        # Layer 1 — Is it spam?
        l1 = run_layer1(payload.message)

        # If safe, return early
        if not l1["proceed_to_layer2"]:
            total_ms = round((time.time() - pipeline_start) * 1000, 2)
            logger.info(f"✅ SAFE | {total_ms}ms | \"{payload.message[:50]}...\"")

            result = {
                "is_fraud": False,
                "scam_probability": l1["risk_score"],
                "risk_level": l1["risk_level"],
                "fraud_type": None,
                "suspicious_keywords": l1.get("matched_keywords", []),
                "explanation": EXPLANATIONS.get(_detect_language(payload.message), EXPLANATIONS["en"])["safe"],
                "prevention_tips": [],
                "helpline": None,
                "url_analysis": None,
            }
        else:
            # Layer 2 — What kind of fraud?
            l2 = run_layer2(payload.message, layer1_result=l1)
            fraud_type = l2["fraud_type"]
            total_ms = round((time.time() - pipeline_start) * 1000, 2)
            logger.info(f"🚨 {l2['risk_level'].upper()} | {fraud_type} | {l2['risk_score']}/100 | {total_ms}ms | \"{payload.message[:50]}...\"")

            result = {
                "is_fraud": True,
                "scam_probability": l2["risk_score"],
                "risk_level": l2["risk_level"],
                "fraud_type": fraud_type,
                "suspicious_keywords": l2["matched_keywords"],
                "explanation": l2["explanation"],
                "prevention_tips": l2.get("prevention_tips", []),
                "helpline": "📞 National Cyber Crime Helpline: 1930 | cybercrime.gov.in",
                "url_analysis": l2.get("url_analysis"),
                "detected_language": l2.get("detected_language", "en"),
            }

        # Timing info
        total_ms = round((time.time() - pipeline_start) * 1000, 2)
        result["processing_time"] = {
            "total_ms": total_ms,
            "layer1_ms": l1.get("processing_time_ms", 0),
            "layer2_ms": l2.get("processing_time_ms", 0) if l1["proceed_to_layer2"] else 0,
        }

        # Auto-save scan to Supabase
        try:
            save_scan(payload.message, result, payload.language)

            if result.get("risk_level") == "High Risk":
                save_to_community(
                    result.get("fraud_type"),
                    result.get("risk_level"),
                    payload.message[:100],
                )

            keywords = result.get("suspicious_keywords", [])
            if keywords:
                upsert_keywords(keywords, result.get("fraud_type"))
        except Exception:
            pass  # Don't fail prediction if DB save fails

        return {"message": payload.message, **result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# 📸 PREDICT IMAGE — Screenshot-based fraud detection
# ---------------------------------------------------------------------------
@router.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    """
    Upload a screenshot of a suspicious message.
    OCR extracts the text, then runs Layer 1 + Layer 2 detection.
    Perfect for rural users who can't copy-paste.
    """
    pipeline_start = time.time()

    # Validate file type
    allowed_types = {"image/png", "image/jpeg", "image/jpg", "image/webp"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Upload PNG, JPG, or WEBP."
        )

    # Validate file size (max 10MB)
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image too large (max 10MB)")

    # OCR — Extract text from image
    ocr_result = extract_text_from_bytes(contents)

    if not ocr_result["success"] or not ocr_result["text"].strip():
        return {
            "message": None,
            "extracted_text": ocr_result.get("text", ""),
            "ocr_confidence": ocr_result.get("confidence", 0),
            "ocr_lines": ocr_result.get("lines", []),
            "is_fraud": False,
            "error": ocr_result.get("error", "Could not extract text from image"),
            "processing_time": {
                "total_ms": round((time.time() - pipeline_start) * 1000, 2),
                "ocr_ms": round((time.time() - pipeline_start) * 1000, 2),
            },
        }

    extracted_text = ocr_result["text"]
    ocr_time = round((time.time() - pipeline_start) * 1000, 2)

    # Run detection pipeline on extracted text
    l1 = run_layer1(extracted_text)

    if not l1["proceed_to_layer2"]:
        total_ms = round((time.time() - pipeline_start) * 1000, 2)
        logger.info(f"📸 ✅ SAFE | {total_ms}ms | OCR: \"{extracted_text[:50]}...\"")

        result = {
            "is_fraud": False,
            "scam_probability": l1["risk_score"],
            "risk_level": l1["risk_level"],
            "fraud_type": None,
            "suspicious_keywords": l1.get("matched_keywords", []),
            "explanation": EXPLANATIONS.get(ocr_result.get("language_detected", "en"), EXPLANATIONS["en"])["safe"],
            "prevention_tips": [],
            "helpline": None,
            "url_analysis": None,
        }
    else:
        ocr_lang = ocr_result.get("language_detected", None)
        l2 = run_layer2(extracted_text, layer1_result=l1, language=ocr_lang)
        fraud_type = l2["fraud_type"]
        total_ms = round((time.time() - pipeline_start) * 1000, 2)
        logger.info(f"📸 🚨 {l2['risk_level'].upper()} | {fraud_type} | {total_ms}ms | OCR: \"{extracted_text[:50]}...\"")

        result = {
            "is_fraud": True,
            "scam_probability": l2["risk_score"],
            "risk_level": l2["risk_level"],
            "fraud_type": fraud_type,
            "suspicious_keywords": l2["matched_keywords"],
            "explanation": l2["explanation"],
            "prevention_tips": l2.get("prevention_tips", []),
            "helpline": "📞 National Cyber Crime Helpline: 1930 | cybercrime.gov.in",
            "url_analysis": l2.get("url_analysis"),
            "detected_language": l2.get("detected_language", "en"),
        }

    # Auto-save to Supabase
    try:
        save_scan(extracted_text, result, "auto")
        if result.get("risk_level") == "High Risk":
            save_to_community(result.get("fraud_type"), result.get("risk_level"), extracted_text[:100])
        keywords = result.get("suspicious_keywords", [])
        if keywords:
            upsert_keywords(keywords, result.get("fraud_type"))
    except Exception:
        pass

    total_ms = round((time.time() - pipeline_start) * 1000, 2)

    return {
        "message": extracted_text,
        "extracted_text": extracted_text,
        "ocr_confidence": ocr_result["confidence"],
        "ocr_lines": ocr_result["lines"],
        **result,
        "processing_time": {
            "total_ms": total_ms,
            "ocr_ms": ocr_time,
            "layer1_ms": l1.get("processing_time_ms", 0),
            "layer2_ms": l2.get("processing_time_ms", 0) if l1["proceed_to_layer2"] else 0,
        },
    }


# ---------------------------------------------------------------------------
# 📊 STATS — Real-time detection statistics (Great for demo!)
# ---------------------------------------------------------------------------
@router.get("/stats")
def detection_stats():
    """
    Get real-time detection statistics.
    Shows total scans, spam/ham counts, top patterns, and system health.
    """
    stats = get_pattern_stats()
    total = stats["scan_counter"].get("total", 0)
    spam = stats["scan_counter"].get("spam", 0)
    ham = stats["scan_counter"].get("ham", 0)

    return {
        "total_scans": total,
        "fraud_detected": spam,
        "safe_messages": ham,
        "fraud_rate": round(spam / total * 100, 1) if total > 0 else 0,
        "top_scam_patterns": [
            {"pattern": name, "count": count}
            for name, count in stats["top_patterns"]
        ],
        "system_status": "operational",
        "models_loaded": {
            "layer1": "distilbert-spam-detection",
            "layer2": "bert-finetuned-phishing",
        },
    }


# ---------------------------------------------------------------------------
# 📋 SCANS — Save and retrieve scan history
# ---------------------------------------------------------------------------
@router.post("/scans")
def create_scan(payload: ScanResult):
    """Your teammate calls this after getting Groq response"""
    try:
        save_scan(payload.message, payload.result, payload.language)

        if payload.result.get("risk_level") == "High Risk":
            save_to_community(
                payload.result.get("fraud_type"),
                payload.result.get("risk_level"),
                payload.message[:100]
            )

        keywords = payload.result.get("suspicious_keywords", [])
        if keywords:
            upsert_keywords(keywords, payload.result.get("fraud_type"))

        return {"status": "saved"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/scans")
def list_scans(limit: int = 20):
    return get_recent_scans(limit)


@router.get("/scans/{scan_id}")
def get_scan(scan_id: str):
    scan = get_scan_by_id(scan_id)
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")
    return scan


# ---------------------------------------------------------------------------
# 🌐 COMMUNITY — Feed of reported scams
# ---------------------------------------------------------------------------
@router.get("/community")
def community_feed(limit: int = 20, fraud_type: str = None):
    return get_community_feed(limit, fraud_type)


# ---------------------------------------------------------------------------
# 🔑 KEYWORDS — Top flagged keywords
# ---------------------------------------------------------------------------
@router.get("/keywords")
def top_keywords(limit: int = 10):
    return get_top_keywords(limit)