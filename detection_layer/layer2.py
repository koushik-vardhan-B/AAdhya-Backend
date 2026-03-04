"""
Layer 2 — Fraud Type Classifier
Uses ealvaradob/bert-finetuned-phishing for phishing detection +
keyword-based classification for Indian-specific fraud types.

This layer ONLY runs on messages flagged by Layer 1 (spam/suspicious).

Improvements:
    - URL Risk Detector (analyzes actual URL structure)
    - Calibrated scoring (smooth, realistic confidence)
    - Hinglish pattern support
    - Response time tracking

Output fraud types:
    - UPI Fraud
    - Job Scam
    - Lottery Scam
    - Phishing
    - Others
"""

import os
import re
import time
import torch
from urllib.parse import urlparse
from transformers import BertTokenizer, BertForSequenceClassification

# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_PATH = os.path.join(_SCRIPT_DIR, "models", "bert_phishing")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Layer 2 using device: {device}")

tokenizer = BertTokenizer.from_pretrained(_MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(_MODEL_PATH).to(device)
model.eval()

# ---------------------------------------------------------------------------
# URL Risk Detector
# ---------------------------------------------------------------------------
# Suspicious TLDs often used in phishing
SUSPICIOUS_TLDS = {
    ".xyz", ".top", ".buzz", ".club", ".info", ".tk", ".ml", ".ga",
    ".cf", ".gq", ".work", ".click", ".link", ".online", ".site",
    ".icu", ".pw", ".cc", ".ws", ".bid", ".stream", ".racing",
}

# Known legitimate domains (whitelist)
SAFE_DOMAINS = {
    "google.com", "facebook.com", "amazon.in", "flipkart.com", "paytm.com",
    "phonepe.com", "sbi.co.in", "hdfcbank.com", "icicibank.com",
    "axisbank.com", "rbi.org.in", "npci.org.in", "gov.in", "nic.in",
}

# Brand names attackers commonly mimic
SPOOFED_BRANDS = [
    "sbi", "hdfc", "icici", "axis", "pnb", "kotak", "canara",
    "paytm", "phonepe", "gpay", "google", "amazon", "flipkart",
    "whatsapp", "telegram", "facebook", "instagram", "gov", "aadhaar",
]


def _analyze_url(url_str: str) -> dict:
    """
    Analyze a URL for phishing signals.

    Returns:
        dict with:
            - is_suspicious: bool
            - risk_score: float 0.0 – 1.0
            - reasons: list of human-readable reasons
    """
    reasons = []
    score = 0.0

    try:
        parsed = urlparse(url_str if "://" in url_str else f"http://{url_str}")
        domain = parsed.netloc.lower()
        full_url = url_str.lower()

        # 1. Short URL services → always suspicious in SMS
        short_domains = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "short.link", "rb.gy", "is.gd"}
        if any(s in domain for s in short_domains):
            score += 0.50
            reasons.append("Shortened URL (hides real destination)")

        # 2. Suspicious TLD
        for tld in SUSPICIOUS_TLDS:
            if domain.endswith(tld):
                score += 0.35
                reasons.append(f"Suspicious TLD: {tld}")
                break

        # 3. IP address instead of domain
        if re.match(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", domain):
            score += 0.60
            reasons.append("IP address used instead of domain name")

        # 4. Brand spoofing (e.g., sbi-update.in, hdfc-verify.com)
        for brand in SPOOFED_BRANDS:
            if brand in domain and domain not in SAFE_DOMAINS:
                score += 0.45
                reasons.append(f"Possible brand impersonation: '{brand}' in URL")
                break

        # 5. Excessive hyphens (common in phishing: sbi-account-update-verify.com)
        if domain.count("-") >= 2:
            score += 0.25
            reasons.append("Multiple hyphens in domain (common phishing tactic)")

        # 6. Long subdomain chains (secure.login.verify.sbi-fake.xyz)
        if domain.count(".") >= 3:
            score += 0.20
            reasons.append("Multiple subdomains (suspicious domain structure)")

        # 7. HTTP without HTTPS
        if url_str.startswith("http://"):
            score += 0.15
            reasons.append("No HTTPS (insecure connection)")

    except Exception:
        score += 0.30
        reasons.append("Malformed URL")

    return {
        "is_suspicious": score > 0.25,
        "risk_score": min(score, 1.0),
        "reasons": reasons,
    }


def _extract_urls(message: str) -> list:
    """Extract all URLs from a message."""
    url_pattern = r"https?://\S+|(?:bit\.ly|tinyurl\.com|t\.co|goo\.gl)/\S+"
    return re.findall(url_pattern, message, re.IGNORECASE)


# ---------------------------------------------------------------------------
# Fraud type keyword patterns (more granular than Layer 1)
# ---------------------------------------------------------------------------
FRAUD_TYPE_PATTERNS = {
    "UPI Fraud": {
        "score": 0,
        "keywords": [
            r"upi\s*(id|pin|fraud|block|debit|credit)",
            r"(gpay|google\s*pay|phonepe|paytm|bhim|amazon\s*pay)",
            r"(bank|sbi|hdfc|icici|axis|pnb|boi|canara|kotak).*(suspend|block|verify|update|freeze|close)",
            r"(account|a/c).*(block|suspend|close|verify|freeze|debit)",
            r"(send|transfer|pay)\s*(₹|rs\.?\s*\d|money|amount)",
            r"(refund|cashback).*(₹|rs|fail|process|pending)",
            r"(otp|pin|cvv|password)\s*(share|send|enter|verify)",
            r"(kyc|know\s*your\s*customer)\s*(update|verify|expire|pending)",
            r"(card|debit\s*card|credit\s*card).*(block|expire|suspend)",
            # Hinglish
            r"(paisa|paise|rupay)\s*(bhej|de|wapas)",
            r"(khata|account)\s*(band|block|freeze)",
        ],
    },
    "Lottery Scam": {
        "score": 0,
        "keywords": [
            r"(won|win|winner|congratulat).*(lakh|crore|prize|reward|₹|rs|dollar|\$|£)",
            r"(kbc|kaun\s*banega|lottery|lucky\s*draw|mega\s*draw)",
            r"(prize|reward|gift).*(₹|rs|money|amount|cash)",
            r"(claim|collect|redeem)\s*(now|your|prize|reward|amount|gift)",
            r"(₹|rs\.?)\s*\d+.*(lakh|crore|thousand|million)",
            r"(lucky|selected|chosen)\s*(customer|number|winner|user)",
            r"(bumper|jackpot|grand)\s*(prize|offer|win)",
            # Hinglish
            r"(jeet|jeeta|jeete).*(inam|prize|lakh|crore)",
            r"badhai\s*ho.*(jeet|prize|inam)",
        ],
    },
    "Job Scam": {
        "score": 0,
        "keywords": [
            r"(earn|income|salary|make)\s*(₹|rs|money|upto).*\d",
            r"work\s*from\s*home",
            r"(registration|joining|processing)\s*(fee|charge|amount|cost)",
            r"(₹|rs\.?)\s*\d+.*per\s*(month|day|hour|week|task)",
            r"part\s*time.*(job|income|work|earn)",
            r"(data\s*entry|typing|copy\s*paste|ad\s*posting)\s*(job|work)",
            r"(no\s*experience|no\s*qualification|anyone\s*can)",
            r"(whatsapp|telegram|contact).*\d{10}",
            r"(vacancy|hiring|recruit).*(urgent|immediate)",
            r"(daily\s*payment|weekly\s*payment|instant\s*payment)",
            # Hinglish
            r"(ghar\s*baithe|ghar\s*se)\s*(kama|paise)",
            r"(naukri|kaam)\s*(chahiye|milegi|dilayenge)",
        ],
    },
    "Phishing": {
        "score": 0,
        "keywords": [
            r"(click|tap)\s*(here|now|link|below|button)",
            r"(verify|update|confirm|validate)\s*(now|your|account|identity|email)",
            r"http[s]?://\S+",
            r"(bit\.ly|tinyurl|short\.link|goo\.gl|t\.co)",
            r"(pan|aadhaar|aadhar)\s*(card|number|link|update|verify|illegal)",
            r"(login|log\s*in|sign\s*in)\s*(here|now|to\s*verify)",
            r"(suspend|restrict|limit|disable).*(account|access|service)",
            r"(security|suspicious)\s*(alert|activity|login|access)",
            r"(reset|change)\s*(password|pin|credential)",
            # Hinglish
            r"(yahan|neeche)\s*(click|tap|dabaye)",
            r"(link|url)\s*(khole|kholo|open)",
        ],
    },
}


def _classify_fraud_type(message: str) -> dict:
    """Classify the specific fraud type using keyword pattern matching."""
    text = message.lower()
    scores = {}
    all_matched = {}

    for fraud_type, pattern_group in FRAUD_TYPE_PATTERNS.items():
        matched = []
        for regex in pattern_group["keywords"]:
            match = re.search(regex, text)
            if match:
                matched.append(match.group())

        total_patterns = len(pattern_group["keywords"])
        score = len(matched) / total_patterns if total_patterns > 0 else 0

        # Boost if multiple keywords match
        if len(matched) >= 3:
            score = min(score * 1.3, 1.0)
        if len(matched) >= 2:
            score = min(score * 1.15, 1.0)

        scores[fraud_type] = round(score, 4)
        all_matched[fraud_type] = matched

    # Find the best matching fraud type
    if not any(scores.values()):
        return {
            "fraud_type": "Others",
            "type_confidence": 0.5,
            "matched_keywords": [],
            "all_scores": scores,
        }

    best_type = max(scores, key=scores.get)
    best_score = scores[best_type]

    return {
        "fraud_type": best_type,
        "type_confidence": round(best_score, 4),
        "matched_keywords": all_matched[best_type],
        "all_scores": scores,
    }


def run_layer2(message: str, layer1_result: dict = None) -> dict:
    """
    Classify a flagged message into specific fraud type.

    Combines:
        - BERT phishing model confidence
        - Keyword-based fraud type classification
        - URL risk analysis

    Returns dict with fraud_type, risk_score, explanation, url_analysis, etc.
    """
    start_time = time.time()

    if not message or not message.strip():
        return {
            "fraud_type": "Others",
            "phishing_confidence": 0.0,
            "type_confidence": 0.0,
            "risk_score": 0,
            "risk_level": "Safe",
            "matched_keywords": [],
            "explanation": "Empty message received.",
            "url_analysis": None,
            "processing_time_ms": 0.0,
        }

    message = message.strip()[:2000]

    # --- BERT phishing model inference ---
    inputs = tokenizer(
        message,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
    ).to(device)

    with torch.inference_mode():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        phishing_prob = probs[0][1].item()

    # --- URL risk analysis ---
    urls = _extract_urls(message)
    url_results = [_analyze_url(u) for u in urls]
    url_risk = max((r["risk_score"] for r in url_results), default=0.0)
    url_reasons = []
    for r in url_results:
        url_reasons.extend(r["reasons"])

    # --- Keyword-based fraud type classification ---
    classification = _classify_fraud_type(message)

    # --- Combine scores with calibration ---
    # If BERT says phishing with high confidence AND keywords match phishing, boost
    if phishing_prob > 0.7 and classification["fraud_type"] == "Phishing":
        classification["type_confidence"] = max(classification["type_confidence"], 0.85)

    # Factor in URL risk — if URLs are suspicious, boost risk
    if url_risk > 0.3:
        phishing_boost = url_risk * 0.4
        classification["type_confidence"] = max(classification["type_confidence"], phishing_boost)

    # Calibrated combination of all signals
    signals = [phishing_prob, classification["type_confidence"], url_risk]
    max_signal = max(signals)

    if layer1_result and "risk_score" in layer1_result:
        l1_score = layer1_result["risk_score"] / 100
        # Weighted: L1 context + BERT + keywords + URL
        combined = (0.20 * l1_score) + (0.25 * phishing_prob) + (0.35 * classification["type_confidence"]) + (0.20 * url_risk)
        # Don't let combined score be lower than the strongest signal
        combined = max(combined, max_signal * 0.75)
        combined = max(combined, 0.45)  # Already flagged, minimum suspicious
    else:
        combined = max_signal

    # Smooth variance to avoid identical scores
    variance = (phishing_prob * 0.08) + (url_risk * 0.05)
    combined = min(combined + variance * (1 - combined), 1.0)

    risk_score = int(combined * 100)
    risk_level = "High Risk" if risk_score >= 65 else "Suspicious"

    # --- Generate explanation ---
    explanation = _generate_explanation(
        classification["fraud_type"],
        classification["matched_keywords"],
        phishing_prob,
        risk_score,
        url_reasons,
    )

    elapsed = round((time.time() - start_time) * 1000, 2)

    return {
        "fraud_type": classification["fraud_type"],
        "phishing_confidence": round(phishing_prob, 4),
        "type_confidence": classification["type_confidence"],
        "risk_score": risk_score,
        "risk_level": risk_level,
        "matched_keywords": classification["matched_keywords"],
        "explanation": explanation,
        "url_analysis": {
            "urls_found": urls,
            "url_risk_score": round(url_risk, 4),
            "url_warnings": url_reasons,
        } if urls else None,
        "processing_time_ms": elapsed,
    }


def _generate_explanation(fraud_type: str, keywords: list, phishing_prob: float, risk_score: int, url_reasons: list = None) -> str:
    """Generate a human-readable explanation of the detection."""

    explanations = {
        "UPI Fraud": (
            "This message appears to be a UPI/banking fraud attempt. "
            "It tries to trick you into sharing banking details, OTP, or making payments. "
            "Legitimate banks NEVER ask for OTP, PIN, or CVV via SMS."
        ),
        "Lottery Scam": (
            "This message claims you've won a prize or lottery. "
            "This is a classic fraud — you cannot win a lottery you never entered. "
            "NEVER pay any 'processing fee' or share personal details to claim prizes."
        ),
        "Job Scam": (
            "This message promotes a suspicious job or earning opportunity. "
            "Legitimate jobs NEVER ask for registration fees upfront. "
            "Be wary of unrealistic salary promises and 'work from home' offers."
        ),
        "Phishing": (
            "This message contains suspicious links or asks you to verify personal information. "
            "NEVER click unknown links or enter credentials on unfamiliar websites. "
            "Always verify URLs carefully before clicking."
        ),
        "Others": (
            "This message has been flagged as potentially fraudulent. "
            "Exercise caution and verify the sender's identity before responding."
        ),
    }

    base = explanations.get(fraud_type, explanations["Others"])

    if keywords:
        base += f" Suspicious keywords detected: {', '.join(keywords[:5])}."

    if url_reasons:
        base += f" URL risks: {'; '.join(url_reasons[:3])}."

    if risk_score >= 80:
        base += " ⚠️ HIGH RISK — Do NOT respond to this message."
    elif risk_score >= 60:
        base += " ⚠️ Exercise extreme caution with this message."

    return base
