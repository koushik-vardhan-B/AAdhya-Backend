from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from database import (
    save_scan, get_recent_scans, get_scan_by_id,
    save_to_community, get_community_feed,
    upsert_keywords, get_top_keywords
)

router = APIRouter()

class ScanResult(BaseModel):
    message: str
    language: str = "en"
    result: dict  # the full JSON from groq layer


## scans

@router.post("/scans")
def create_scan(payload: ScanResult):
    """Your teammate calls this after getting Groq response"""
    try:
        save_scan(payload.message, payload.result, payload.language)

        # Auto-save High Risk to community feed
        if payload.result.get("risk_level") == "High Risk":
            save_to_community(
                payload.result.get("fraud_type"),
                payload.result.get("risk_level"),
                payload.message[:100]
            )

        # Track keywords
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


## community feed 

@router.get("/community")
def community_feed(limit: int = 20, fraud_type: str = None):
    return get_community_feed(limit, fraud_type)


## keywords

@router.get("/keywords")
def top_keywords(limit: int = 10):
    return get_top_keywords(limit)