from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import router
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="AADHYA 3.0 - Fraud Detection API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

app.include_router(router, prefix="/api")

@app.get("/health")
def health():
    return {"status": "ok"}