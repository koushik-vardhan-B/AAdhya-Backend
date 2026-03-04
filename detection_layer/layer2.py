"""
Layer 2 — Fraud Type Classifier
Uses ealvaradob/bert-finetuned-phishing for phishing detection +
keyword-based classification for Indian-specific fraud types.

This layer ONLY runs on messages flagged by Layer 1 (spam/suspicious).
It answers: "What KIND of fraud is this?"

Output fraud types:
    - UPI Fraud
    - Job Scam
    - Lottery Scam
    - Phishing
    - Others
"""

import os
import re
import torch
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
# Fraud type keyword patterns (more granular than Layer 1)
# These determine the SPECIFIC type of fraud
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
        ],
    },
}


def _classify_fraud_type(message: str) -> dict:
    """
    Classify the specific fraud type using keyword pattern matching.

    Returns:
        dict with:
            - fraud_type: str (UPI Fraud, Lottery Scam, Job Scam, Phishing, Others)
            - type_confidence: float 0.0–1.0
            - matched_keywords: list of matched keyword strings
            - all_scores: dict of scores per fraud type (for transparency)
    """
    text = message.lower()
    scores = {}
    all_matched = {}

    for fraud_type, pattern_group in FRAUD_TYPE_PATTERNS.items():
        matched = []
        for regex in pattern_group["keywords"]:
            match = re.search(regex, text)
            if match:
                matched.append(match.group())

        # Score = number of matched patterns / total patterns in group
        # More matches = higher confidence in this fraud type
        total_patterns = len(pattern_group["keywords"])
        score = len(matched) / total_patterns if total_patterns > 0 else 0

        # Boost if multiple keywords match (strong signal)
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

    Args:
        message: The raw SMS/text message (already flagged by Layer 1).
        layer1_result: Optional dict from Layer 1 for context.

    Returns:
        dict with keys:
            - fraud_type: "UPI Fraud" | "Lottery Scam" | "Job Scam" | "Phishing" | "Others"
            - phishing_confidence: float 0.0–1.0 (BERT model score)
            - type_confidence: float 0.0–1.0 (keyword classification score)
            - risk_score: int 0–100 (final combined score)
            - risk_level: "Suspicious" or "High Risk"
            - matched_keywords: list of keywords that matched
            - explanation: str (human-readable reason)
    """
    if not message or not message.strip():
        return {
            "fraud_type": "Others",
            "phishing_confidence": 0.0,
            "type_confidence": 0.0,
            "risk_score": 0,
            "risk_level": "Safe",
            "matched_keywords": [],
            "explanation": "Empty message received.",
        }

    # --- BERT phishing model inference ---
    inputs = tokenizer(
        message,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        phishing_prob = probs[0][1].item()  # Class 1 = phishing

    # --- Keyword-based fraud type classification ---
    classification = _classify_fraud_type(message)

    # --- Combine scores ---
    # If BERT says phishing with high confidence AND keywords match phishing,
    # boost the phishing type score
    if phishing_prob > 0.7 and classification["fraud_type"] == "Phishing":
        classification["type_confidence"] = max(classification["type_confidence"], 0.85)

    # If BERT says phishing but keywords say something more specific
    # (like UPI Fraud), trust the keywords for the TYPE but use BERT
    # confidence for the overall risk
    combined_risk = max(phishing_prob, classification["type_confidence"])

    # Factor in Layer 1 confidence if available
    if layer1_result and "risk_score" in layer1_result:
        l1_score = layer1_result["risk_score"] / 100
        combined_risk = (0.3 * l1_score) + (0.3 * phishing_prob) + (0.4 * classification["type_confidence"])
        combined_risk = max(combined_risk, 0.45)  # Already flagged, minimum suspicious

    risk_score = int(combined_risk * 100)
    risk_level = "High Risk" if risk_score >= 65 else "Suspicious"

    # --- Generate explanation ---
    explanation = _generate_explanation(
        classification["fraud_type"],
        classification["matched_keywords"],
        phishing_prob,
        risk_score,
    )

    return {
        "fraud_type": classification["fraud_type"],
        "phishing_confidence": round(phishing_prob, 4),
        "type_confidence": classification["type_confidence"],
        "risk_score": risk_score,
        "risk_level": risk_level,
        "matched_keywords": classification["matched_keywords"],
        "explanation": explanation,
    }


def _generate_explanation(fraud_type: str, keywords: list, phishing_prob: float, risk_score: int) -> str:
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

    if risk_score >= 80:
        base += " ⚠️ HIGH RISK — Do NOT respond to this message."
    elif risk_score >= 60:
        base += " ⚠️ Exercise extreme caution with this message."

    return base
