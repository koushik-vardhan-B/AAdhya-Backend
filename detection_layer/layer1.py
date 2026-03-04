"""
Layer 1 — Fast Binary Spam/Ham Pre-filter with Indian Keyword Booster
Uses AventIQ-AI/distilBERT_spam_detection + keyword pattern matching
to detect both Western and Indian-specific scams.
"""

import os
import re
import torch
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
# Indian-specific scam keyword patterns
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


def run_layer1(message: str) -> dict:
    """
    Analyze a message using DistilBERT model + Indian keyword booster.

    The final score combines:
        - Model confidence (what the AI thinks)
        - Keyword score (Indian-specific scam pattern matches)

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
    """
    if not message or not message.strip():
        return {
            "label": "ham",
            "confidence": 1.0,
            "risk_score": 0,
            "risk_level": "Safe",
            "matched_patterns": [],
            "matched_keywords": [],
            "proceed_to_layer2": False,
        }

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
    # If safe patterns match, trust it as legitimate regardless of model
    # (OTP messages, bill reminders, etc. are very specific patterns)
    if kw["is_safe_override"]:
        return {
            "label": "ham",
            "confidence": round(max(1 - spam_prob, 0.85), 4),
            "risk_score": max(int(spam_prob * 15), 5),  # Low risk
            "risk_level": "Safe",
            "matched_patterns": [],
            "matched_keywords": [],
            "proceed_to_layer2": False,
        }

    # --- Combined scoring ---
    # Weighted combination: 40% model + 60% keywords (keywords are more
    # reliable for Indian scams since model was trained on Western data)
    combined_score = (0.4 * spam_prob) + (0.6 * kw["keyword_score"])

    # Minimum score floors based on keyword matches
    # Even a single strong pattern match should flag as at least suspicious
    if len(kw["matched_patterns"]) >= 1:
        combined_score = max(combined_score, 0.45)  # At least Suspicious
    if len(kw["matched_patterns"]) >= 2:
        combined_score = max(combined_score, 0.75)  # High Risk
    if len(kw["matched_patterns"]) >= 3:
        combined_score = max(combined_score, 0.90)  # Very High Risk

    # Final decision
    risk_score = int(combined_score * 100)

    if combined_score >= 0.60:
        label = "spam"
        risk_level = "High Risk" if combined_score >= 0.75 else "Suspicious"
    elif combined_score >= 0.35:
        label = "spam"
        risk_level = "Suspicious"
    else:
        label = "ham"
        risk_level = "Safe"

    return {
        "label": label,
        "confidence": round(combined_score, 4),
        "risk_score": risk_score,
        "risk_level": risk_level,
        "matched_patterns": kw["matched_patterns"],
        "matched_keywords": kw["matched_keywords"],
        "proceed_to_layer2": label == "spam" or risk_level == "Suspicious",
    }
