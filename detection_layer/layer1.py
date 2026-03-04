"""
Layer 1 — Fast Binary Spam/Ham Pre-filter with Indian Keyword Booster
Uses AventIQ-AI/distilBERT_spam_detection + keyword pattern matching
to detect both Western and Indian-specific scams.

Improvements:
    - Calibrated confidence scores (smooth, realistic)
    - Hinglish / Hindi keyword support
    - Scam pattern memory (tracks pattern frequency)
    - Better input validation
"""

import os
import re
import time
import torch
from collections import Counter
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_PATH = os.path.join(_SCRIPT_DIR, "models", "distilbert_spam")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Layer 1 using device: {device}")

tokenizer = DistilBertTokenizer.from_pretrained(_MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(_MODEL_PATH).to(device)
model.eval()

# ---------------------------------------------------------------------------
# Scam Pattern Memory — tracks which patterns are hit most
# ---------------------------------------------------------------------------
pattern_memory = Counter()   # { "upi_fraud": 12, "phishing": 8, ... }
scan_counter = {"total": 0, "spam": 0, "ham": 0}

# ---------------------------------------------------------------------------
# Indian-specific scam keyword patterns (English + Hinglish)
# Each pattern group has a weight (how strongly it indicates a scam)
# ---------------------------------------------------------------------------
SCAM_PATTERNS = {
    # UPI / Banking fraud
    "upi_fraud": {
        "weight": 0.35,
        "keywords": [
            r"upi\s*(id|pin|fraud|block)",
            r"(gpay|phonepe|paytm|bhim).*(block|suspend|verify|expire)",
            r"(bank|sbi|hdfc|icici|axis).*(suspend|block|verify|update\s*kyc)",
            r"(account|a/c).*(block|suspend|close|verify)",
            r"send\s*(money|amount|rs|₹)",
            # Hinglish
            r"(paisa|paise|rupay)\s*(bhej|transfer|de)",
            r"(khata|account)\s*(band|block|verify)",
            r"apn[ae]\s*(bank|upi|account)\s*(verify|update)",
        ],
    },
    # Lottery / Prize scam
    "lottery_scam": {
        "weight": 0.40,
        "keywords": [
            r"(won|win|winner|congratulat).*(lakh|crore|prize|reward|₹|rs\.?\s*\d)",
            r"(kbc|kaun\s*banega|lottery|lucky\s*draw)",
            r"claim\s*(now|your|prize|reward|amount)",
            r"(₹|rs\.?)\s*\d+.*(lakh|crore|thousand)",
            # Hinglish
            r"(jeet|jeeta|jeete|jeetiye).*(lakh|crore|inam|prize)",
            r"(inam|inaam)\s*(milega|mila|le)",
            r"badhai\s*ho",
        ],
    },
    # Job scam
    "job_scam": {
        "weight": 0.30,
        "keywords": [
            r"(earn|income|salary).*(₹|rs|month|daily|weekly)",
            r"work\s*from\s*home",
            r"(registration|joining)\s*(fee|charge|amount)",
            r"(₹|rs\.?)\s*\d+.*per\s*(month|day|hour)",
            r"part\s*time.*income",
            # Hinglish
            r"(ghar\s*baithe|ghar\s*se)\s*(kama|kamai|paise)",
            r"(naukri|job)\s*(chahiye|dilayenge|milegi)",
            r"(kamai|kamao|kamaye)\s*(₹|rs|lakh|hazar)",
        ],
    },
    # Phishing / Link scam
    "phishing": {
        "weight": 0.30,
        "keywords": [
            r"(click|tap)\s*(here|now|link|below)",
            r"(verify|update|confirm)\s*(now|your|account|kyc|pan|aadhaar)",
            r"http[s]?://\S+",  # any URL in SMS is suspicious
            r"bit\.ly|tinyurl|short\.link",
            r"(pan|aadhaar|aadhar)\s*(card|number|link|illegal)",
            # Hinglish
            r"(yahan|idhar|neeche)\s*(click|tap|dabaye)",
            r"(link|url)\s*(khole|kholo|open\s*karo)",
        ],
    },
    # Urgency / Pressure tactics
    "urgency": {
        "weight": 0.20,
        "keywords": [
            r"urgent",
            r"immediate(ly)?",
            r"(last|final)\s*(chance|warning|notice)",
            r"(expire|expir|within)\s*\d+\s*(hour|minute|hr|min)",
            r"act\s*now",
            r"limited\s*(time|offer|period)",
            # Hinglish
            r"(jaldi|turant|abhi)\s*(karo|kare|karein)",
            r"(aakhri|antim)\s*(mauka|chance|chetavni)",
        ],
    },
}

# Patterns that indicate a LEGITIMATE message (reduce false positives)
SAFE_PATTERNS = [
    r"your\s*otp\s*(is|:)\s*\d+.*do\s*not\s*share",  # Real OTP message
    r"(dear\s*customer|valued\s*customer).*otp",
    r"bill\s*(due|payment|of\s*₹|of\s*rs)",  # Utility bill reminders
    r"(appointment|meeting|schedule|reminder)\s*(at|on|for)",
    r"(happy\s*birthday|congratulations\s*on\s*your)",
    r"(delivery|order|shipment).*(arriving|dispatched|shipped)",
    # Hindi safe
    r"(otp|code).*kisi\s*ko\s*(mat|nahi)\s*(bataye|batayen|share)",
]


def _keyword_scan(message: str) -> dict:
    """
    Scan message against Indian scam keyword patterns.

    Returns:
        dict with:
            - keyword_score: float 0.0–1.0 (how scammy the keywords look)
            - matched_patterns: list of pattern group names that matched
            - matched_keywords: list of actual keyword matches found
            - is_safe_override: bool if safe patterns matched
    """
    text = message.lower()
    matched_patterns = []
    matched_keywords = []
    total_weight = 0.0

    # Check scam patterns
    for pattern_name, pattern_group in SCAM_PATTERNS.items():
        for regex in pattern_group["keywords"]:
            match = re.search(regex, text)
            if match:
                if pattern_name not in matched_patterns:
                    matched_patterns.append(pattern_name)
                    total_weight += pattern_group["weight"]
                matched_keywords.append(match.group())
                break  # One match per group is enough

    # Check safe patterns (override)
    is_safe = any(re.search(p, text) for p in SAFE_PATTERNS)

    # Cap keyword score at 1.0
    keyword_score = min(total_weight, 1.0)

    return {
        "keyword_score": keyword_score,
        "matched_patterns": matched_patterns,
        "matched_keywords": matched_keywords,
        "is_safe_override": is_safe,
    }


def _calibrate_score(model_prob: float, keyword_score: float, num_patterns: int) -> float:
    """
    Calibrate the combined score to produce realistic, well-distributed values.
    Avoids clustering at fixed thresholds like 45, 75, 90.
    """
    # Base: weighted blend (model is less reliable for Indian scams)
    base = (0.35 * model_prob) + (0.65 * keyword_score)

    # Pattern count bonus (diminishing returns)
    pattern_bonus = {0: 0.0, 1: 0.12, 2: 0.28, 3: 0.38, 4: 0.44, 5: 0.48}
    bonus = pattern_bonus.get(num_patterns, 0.50)
    base = max(base, bonus)

    # Add small variance based on model confidence to avoid identical scores
    # e.g., two "1 pattern" matches won't both score exactly 0.45
    variance = model_prob * 0.15
    calibrated = base + variance * (1 - base)  # Compress toward top

    # Smooth into 0–1 range
    calibrated = max(0.0, min(calibrated, 1.0))

    # Final floor: if any pattern matched, minimum 0.40
    if num_patterns >= 1:
        calibrated = max(calibrated, 0.40)

    return round(calibrated, 4)


def run_layer1(message: str) -> dict:
    """
    Analyze a message using DistilBERT model + Indian keyword booster.

    Args:
        message: The raw SMS/text message to analyze.

    Returns:
        dict with keys:
            - label: "spam" or "ham"
            - confidence: float 0.0–1.0
            - risk_score: int 0–100
            - risk_level: "Safe", "Suspicious", or "High Risk"
            - matched_patterns: list of scam pattern groups matched
            - matched_keywords: list of actual keywords found
            - proceed_to_layer2: bool
            - processing_time_ms: float
    """
    start_time = time.time()

    # --- Input validation ---
    if not message or not message.strip():
        return {
            "label": "ham",
            "confidence": 1.0,
            "risk_score": 0,
            "risk_level": "Safe",
            "matched_patterns": [],
            "matched_keywords": [],
            "proceed_to_layer2": False,
            "processing_time_ms": 0.0,
        }

    # Strip and limit message length
    message = message.strip()[:2000]

    # --- Model inference ---
    inputs = tokenizer(
        message,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=128,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        spam_prob = probs[0][1].item()  # Class 1 = spam probability

    # --- Keyword booster ---
    kw = _keyword_scan(message)

    # --- Safe pattern override ---
    if kw["is_safe_override"]:
        elapsed = round((time.time() - start_time) * 1000, 2)
        scan_counter["total"] += 1
        scan_counter["ham"] += 1
        return {
            "label": "ham",
            "confidence": round(max(1 - spam_prob, 0.85), 4),
            "risk_score": max(int(spam_prob * 15), 5),
            "risk_level": "Safe",
            "matched_patterns": [],
            "matched_keywords": [],
            "proceed_to_layer2": False,
            "processing_time_ms": elapsed,
        }

    # --- Calibrated scoring ---
    combined_score = _calibrate_score(spam_prob, kw["keyword_score"], len(kw["matched_patterns"]))

    # Final decision
    risk_score = int(combined_score * 100)

    if combined_score >= 0.60:
        label = "spam"
        risk_level = "High Risk" if combined_score >= 0.72 else "Suspicious"
    elif combined_score >= 0.35:
        label = "spam"
        risk_level = "Suspicious"
    else:
        label = "ham"
        risk_level = "Safe"

    elapsed = round((time.time() - start_time) * 1000, 2)

    # --- Update pattern memory ---
    scan_counter["total"] += 1
    if label == "spam":
        scan_counter["spam"] += 1
        for p in kw["matched_patterns"]:
            pattern_memory[p] += 1
    else:
        scan_counter["ham"] += 1

    return {
        "label": label,
        "confidence": round(combined_score, 4),
        "risk_score": risk_score,
        "risk_level": risk_level,
        "matched_patterns": kw["matched_patterns"],
        "matched_keywords": kw["matched_keywords"],
        "proceed_to_layer2": label == "spam" or risk_level == "Suspicious",
        "processing_time_ms": elapsed,
    }


def get_pattern_stats() -> dict:
    """Return scam pattern frequency stats for the /stats endpoint."""
    return {
        "scan_counter": dict(scan_counter),
        "top_patterns": pattern_memory.most_common(10),
    }
