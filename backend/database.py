from supabase import create_client
from dotenv import load_dotenv
from typing import Optional
import os

load_dotenv()

SUPABASE_PROJECT_URL = os.getenv("SUPABASE_PROJECT_URL")
SUPABASE_API_KEY = os.getenv("SUPABASE_API_KEY")

if not SUPABASE_PROJECT_URL or not SUPABASE_API_KEY:
    import logging
    logging.warning("⚠️ SUPABASE_PROJECT_URL or SUPABASE_API_KEY not set. DB features disabled.")
    supabase = None
else:
    supabase = create_client(SUPABASE_PROJECT_URL, SUPABASE_API_KEY)


# scan the input
def save_scan(message: str, result: dict, language: str = "en"):
    if not supabase:
        return None
    data = {
        "message_preview": message[:100],
        "full_message": message,
        "scam_probability": result.get("scam_probability"),
        "risk_level": result.get("risk_level"),
        "fraud_type": result.get("fraud_type"),
        "suspicious_keywords": result.get("suspicious_keywords", []),
        "explanation": result.get("explanation"),
        "prevention_tips": result.get("prevention_tips", []),
        "language": language,
    }
    return supabase.table("scans").insert(data).execute()


def get_recent_scans(limit: int = 20):
    if not supabase:
        return []
    return (
        supabase.table("scans")
        .select("id, created_at, message_preview, scam_probability, risk_level, fraud_type, language")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
        .data
    )


def get_scan_by_id(scan_id: str):
    if not supabase:
        return None
    return supabase.table("scans").select("*").eq("id", scan_id).single().execute().data


## Community feed
def save_to_community(fraud_type: str, risk_level: str, message_preview: str):
    """Save only High Risk scams to community feed."""
    if not supabase:
        return None
    if (risk_level or "").lower() != "high":
        return None
    return supabase.table("community_reports").insert(
        {
            "fraud_type": fraud_type,
            "risk_level": risk_level,
            "message_preview": message_preview,
        }
    ).execute()


def get_community_feed(limit: int = 20, fraud_type: Optional[str] = None):
    if not supabase:
        return []
    query = supabase.table("community_reports").select("*").order("created_at", desc=True).limit(limit)
    if fraud_type:
        query = query.eq("fraud_type", fraud_type)
    return query.execute().data


## Flagged Keywords
def upsert_keywords(keywords: list, fraud_type: str):
    """Increment frequency if keyword exists, insert if new."""
    if not supabase:
        return
    for kw in keywords:
        k = (kw or "").strip().lower()
        if not k:
            continue

        existing = supabase.table("flagged_keywords").select("id, frequency").eq("keyword", k).execute().data

        if existing:
            supabase.table("flagged_keywords").update({"frequency": existing[0]["frequency"] + 1}).eq(
                "keyword", k
            ).execute()
        else:
            supabase.table("flagged_keywords").insert({"keyword": k, "fraud_type": fraud_type}).execute()


def get_top_keywords(limit: int = 10):
    if not supabase:
        return []
    return (
        supabase.table("flagged_keywords")
        .select("keyword, fraud_type, frequency")
        .order("frequency", desc=True)
        .limit(limit)
        .execute()
        .data
    )