import json
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from routes import router
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Middleware: fix copy-paste issues (smart quotes, newlines, control chars)
# Makes the API fully ChatGPT / Word / Google Docs copy-paste friendly.
# ---------------------------------------------------------------------------
class SanitizeBodyMiddleware(BaseHTTPMiddleware):
    """
    Fixes two common copy-paste issues that break JSON parsing:
      1. Smart/curly quotes  →  straight ASCII quotes
      2. Literal newlines & control chars inside JSON strings
    """
    QUOTE_MAP = str.maketrans({
        "\u201c": '"',  # "
        "\u201d": '"',  # "
        "\u2018": "'",  # '
        "\u2019": "'",  # '
        "\u2033": '"',  # ″
        "\u2032": "'",  # ′
    })

    async def dispatch(self, request: Request, call_next):
        if request.method in ("POST", "PUT", "PATCH"):
            content_type = request.headers.get("content-type", "")
            if "json" in content_type:
                body = await request.body()
                text = body.decode("utf-8", errors="replace")

                # Step 1: Fix smart/curly quotes
                text = text.translate(self.QUOTE_MAP)

                # Step 2: Fix literal newlines & control chars in JSON strings
                # json.loads(strict=False) tolerates control characters,
                # then json.dumps re-encodes them with proper \n escapes.
                try:
                    parsed = json.loads(text, strict=False)
                    text = json.dumps(parsed, ensure_ascii=False)
                except (json.JSONDecodeError, ValueError):
                    pass  # If it still fails, let FastAPI return the normal error

                request._body = text.encode("utf-8")
        return await call_next(request)


app = FastAPI(title="AADHYA 3.0 - Fraud Detection API")
app.add_middleware(SanitizeBodyMiddleware)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

app.include_router(router, prefix="/api")


@app.get("/health")
def health():
    return {"status": "ok"}