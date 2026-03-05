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
            # Hindi (Devanagari)
            r"(बैंक|खाता|अकाउंट).*(बंद|ब्लॉक|सस्पेंड|फ्रीज़|वेरिफ़ाई|अपडेट)",
            r"(kyc|केवाईसी).*(अपडेट|वेरिफ़ाई|एक्सपायर|पेंडिंग)",
            r"(otp|ओटीपी|पिन|cvv).*(शेयर|भेजे|बताए|दें)",
            r"(पैसा|पैसे|रुपये).*(भेज|ट्रांसफर|वापस)",
            # Telugu
            r"(బ్యాంకు|ఖాతా|అకౌంట్).*(బ్లాక్|సస్పెండ్|వెరిఫై|అప్‌డేట్)",
            r"(kyc|కేవైసీ).*(అప్‌డేట్|వెరిఫై|ఎక్స్‌పైర్)",
            r"(otp|ఓటీపీ|పిన్).*(చెప్పకండి|షేర్|పంపండి)",
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
            # Hindi (Devanagari)
            r"(जीत|जीता|जीते).*(इनाम|लाख|करोड़|प्राइज)",
            r"(बधाई|मुबारक).*(जीत|इनाम|प्राइज)",
            r"(इनाम|पुरस्कार|बहुमतি).*(प्राप्त|पाएं|मिल|क्लेम)",
            r"(लॉटरी|लकी\s*ड्रॉ|मेगा\s*ड्रॉ)",
            r"(iPhone|आईफोन|सैमसंग|गैलेक्सी).*(जीत|गेलुचु|గెలుచు|won|win)",
            # Telugu
            r"(అభినందనలు|బహుమతి|ఇనాం).*(గెలుచు|గెలిచారు|పొందండి)",
            r"(లాటరీ|లక్కీ\s*డ్రా)",
            r"(iPhone|ఐఫోన్).*(గెలుచు|గెలిచారు)",
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
            # Hindi (Devanagari)
            r"(घर\s*बैठे|घर\s*से)\s*(कमा|कमाई|पैसे)",
            r"(नौकरी|जॉब).*(दिलाएंगे|मिलेगी|चाहिए)",
            r"(रजिस्ट्रेशन|ज्वाइनिंग)\s*(फीस|शुल्क)",
            # Telugu
            r"(ఇంట్లో|ఇంటి\s*నుండి)\s*(సంపాదించ|డబ్బు)",
            r"(ఉద్యోగం|జాబ్).*(రిజిస్ట్రేషన్|ఫీజు)",
        ],
    },
    # Investment / Trading scam
    "investment_scam": {
        "weight": 0.35,
        "keywords": [
            r"(earn|make|get)\s*(₹|rs\.?|र)?\s*\d[\d,\.]*\s*(daily|per\s*day|per\s*month|monthly)",
            r"(mcx|nse|bse|forex|crypto|bitcoin|share\s*market|stock\s*market|commodity)",
            r"(trading|trade)\s*.*(profit|signal|tip|call)",
            r"(signal|tip)s?\s*.*(trading|market|stock|gold|silver|crude)",
            r"(leverage|margin)\s*\d+\s*x",
            r"\d+\s*x\s*(leverage|return|margin)",
            r"(lowest|zero|0|no)\s*(brokerage|tax|fee|commission)",
            r"(guaranteed|assured|fixed)\s*(return|income|profit|earning)",
            r"(gold|silver|crude|nifty|sensex).*(gain|profit|moving|rally|bull)",
            r"(invest|deposit)\s*(₹|rs\.?)\s*\d+.*(return|profit|income|double)",
            r"(mutual\s*fund|sip|demat|portfolio).*(fraud|fake|scam|guaranteed)",
            # Hinglish
            r"(paisa|paise)\s*(double|triple|kamao)",
            r"(share|stock)\s*(market|bazaar)\s*(tip|signal|call)",
            # Hindi (Devanagari)
            r"(शेयर|स्टॉक|मार्केट|बाजार).*(टिप|सिग्नल|कॉल|मुनाफा)",
            r"(निवेश|इन्वेस्ट).*(रिटर्न|मुनाफा|डबल|गारंटी)",
            r"(ट्रेडिंग|ट्रेड).*(प्रॉफिट|सिग्नल|टिप)",
            # Telugu
            r"(షేర్|స్టాక్|మార్కెట్).*(టిప్|సిగ్నల్|లాభం)",
            r"(పెట్టుబడి|ఇన్వెస్ట్).*(రిటర్న్|లాభం|డబుల్)",
        ],
    },
    # Phishing / Link scam
    "phishing": {
        "weight": 0.30,
        "keywords": [
            r"(click|tap)\s*(here|now|link|below)",
            r"(verify|update|confirm)\s*(now|your|account|kyc|pan|aadhaar)",
            r"https?://\S+",  # any URL in SMS is suspicious
            r"bit\.ly|tinyurl|short\.link",
            r"(pan|aadhaar|aadhar)\s*(card|number|link|illegal)",
            # Bare domains with suspicious TLDs
            r"(?:[a-zA-Z0-9\-]+\.)+(?:xyz|top|buzz|club|info|tk|ml|ga|cf|gq|work|click|link|online|site|icu|pw|win|bid|stream|racing)",
            # Hinglish
            r"(yahan|idhar|neeche)\s*(click|tap|dabaye)",
            r"(link|url)\s*(khole|kholo|open\s*karo)",
            # Hindi (Devanagari)
            r"(क्लिक|टैप|दबाएं).*(यहां|नीचे|लिंक|अभी)",
            r"(लिंक|यूआरएल).*(खोलें|ओपन|क्लिक)",
            r"(वेरिफ़ाई|अपडेट|कन्फर्म)\s*(करें|करो|कीजिए)",
            r"(तुरंत|अभी|जल्दी)\s*(अपडेट|वेरिफ़ाई|क्लिक)",
            # Telugu
            r"(క్లిక్|ట్యాప్)\s*(చేయండి|చేసి|ఇక్కడ)",
            r"(లింక్|యూఆర్ఎల్).*(ఓపెన్|క్లిక్)",
            r"(వెరిఫై|అప్‌డేట్)\s*(చేయండి|చేసి)",
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
            # Hindi (Devanagari)
            r"(तुरंत|जल्दी|अभी|फौरन)",
            r"(आखिरी|अंतिम)\s*(मौका|चेतावनी|चांस)",
            r"\d+\s*(घंटे|मिनट|दिन)\s*(में|के\s*अंदर)",
            r"(सिर्फ|केवल)\s*(आज|अभी)",
            # Telugu
            r"(వెంటనే|ఇప్పుడే|త్వరగా)",
            r"(ఈరోజే|ఈ\s*రోజే)",
            r"(చివరి|ఆఖరి)\s*(అవకాశం|ఛాన్స్)",
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
    r"(ओटीपी|otp).*(किसी|कभी).*(शेयर|बताए).*नहीं",
    # Telugu safe
    r"(otp|ఓటీపీ).*(ఎవరికీ|ఎప్పుడూ).*(చెప్పకండి|షేర్)",
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
