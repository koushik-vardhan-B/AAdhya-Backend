"""
Translation Module — Translate fraud explanations and tips to Telugu/Hindi
Uses Groq LLM for high-quality regional language translations.
"""

import os
from typing import Optional
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Initialize Groq client
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# Language codes
SUPPORTED_LANGUAGES = {
    "en": "English",
    "te": "Telugu",
    "hi": "Hindi",
    "ta": "Tamil",
    "kn": "Kannada",
    "bn": "Bengali",
    "mr": "Marathi",
}


def translate_text(text: str, target_language: str) -> str:
    """
    Translate text to target language using Groq.
    
    Args:
        text: The text to translate
        target_language: Language code (te, hi, ta, kn, bn, mr)
    
    Returns:
        Translated text, or original if translation fails
    """
    if not client:
        return text  # No API key, return original
    
    if target_language == "en" or target_language not in SUPPORTED_LANGUAGES:
        return text
    
    lang_name = SUPPORTED_LANGUAGES[target_language]
    
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # Fast and good for translation
            messages=[
                {
                    "role": "system",
                    "content": f"""You are a translator. Translate the following text to {lang_name}.
Rules:
- Keep technical terms like 'OTP', 'UPI', 'PIN', 'CVV', 'KYC' in English
- Keep phone numbers and URLs as-is
- Use simple, easy-to-understand language
- Only output the translation, nothing else"""
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            temperature=0.3,
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Translation error: {e}")
        return text  # Return original on error


def translate_explanation(explanation: str, target_language: str) -> str:
    """Translate fraud explanation to target language."""
    return translate_text(explanation, target_language)


def translate_prevention_tips(tips: list, target_language: str) -> list:
    """
    Translate list of prevention tips to target language.
    
    Args:
        tips: List of prevention tip strings
        target_language: Language code (te, hi, etc.)
    
    Returns:
        List of translated tips
    """
    if not tips or target_language == "en":
        return tips
    
    if not client:
        return tips
    
    # Translate all tips in one API call for efficiency
    combined = "\n".join([f"{i+1}. {tip}" for i, tip in enumerate(tips)])
    
    try:
        translated = translate_text(combined, target_language)
        # Parse back into list
        lines = [line.strip() for line in translated.split("\n") if line.strip()]
        # Remove numbering if present
        cleaned = []
        for line in lines:
            # Remove "1. ", "2. " etc.
            if line and line[0].isdigit() and ". " in line[:4]:
                cleaned.append(line.split(". ", 1)[1])
            else:
                cleaned.append(line)
        return cleaned if cleaned else tips
    except Exception:
        return tips


def translate_response(result: dict, target_language: str) -> dict:
    """
    Translate explanation and prevention_tips in a prediction result.
    
    Args:
        result: The prediction result dict
        target_language: Language code (te, hi, etc.)
    
    Returns:
        Result dict with translated fields + original fields preserved
    """
    if target_language == "en" or target_language not in SUPPORTED_LANGUAGES:
        return result
    
    translated = result.copy()
    
    # Translate explanation
    if result.get("explanation"):
        translated["explanation"] = translate_explanation(
            result["explanation"], target_language
        )
        translated["explanation_original"] = result["explanation"]
    
    # Translate prevention tips
    if result.get("prevention_tips"):
        translated["prevention_tips"] = translate_prevention_tips(
            result["prevention_tips"], target_language
        )
        translated["prevention_tips_original"] = result["prevention_tips"]
    
    translated["translated_to"] = SUPPORTED_LANGUAGES[target_language]
    
    return translated


# Pre-translated common responses for faster response (no API call needed)
PRESET_TRANSLATIONS = {
    "te": {
        "safe_message": "ఈ సందేశం సురక్షితంగా కనిపిస్తోంది. మోసం సూచనలు కనుగొనబడలేదు.",
        "high_risk_warning": "⚠️ అధిక ప్రమాదం — ఈ సందేశానికి స్పందించవద్దు.",
        "helpline": "📞 జాతీయ సైబర్ క్రైమ్ హెల్ప్‌లైన్: 1930 | cybercrime.gov.in",
    },
    "hi": {
        "safe_message": "यह संदेश सुरक्षित प्रतीत होता है। कोई धोखाधड़ी संकेत नहीं मिले।",
        "high_risk_warning": "⚠️ उच्च जोखिम — इस संदेश का जवाब न दें।",
        "helpline": "📞 राष्ट्रीय साइबर क्राइम हेल्पलाइन: 1930 | cybercrime.gov.in",
    },
}


def get_preset_translation(key: str, language: str) -> Optional[str]:
    """Get a preset translation if available."""
    if language in PRESET_TRANSLATIONS:
        return PRESET_TRANSLATIONS[language].get(key)
    return None
