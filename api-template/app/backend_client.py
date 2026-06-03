"""Client for the internal Featurify backend.

This is where BACKEND_KEY is used — server-side only. The browser never sees it.

The backend's /api/response endpoint expects:
  - a form field `payload` containing JSON: {"message": str, "context": [...]}
  - an optional `file` upload (CSV)
  - a header `BACKEND_KEY`
and returns the Featurify JSONOutput.
"""
import json
import logging
from typing import Optional

import httpx
from fastapi import HTTPException

from .config import settings

logger = logging.getLogger("featurify.backend")


async def call_featurify(
    message: str,
    context: list[dict],
    file_bytes: Optional[bytes] = None,
    filename: Optional[str] = None,
) -> dict:
    payload = json.dumps({"message": message, "context": context})
    data = {"payload": payload}
    files = None
    if file_bytes is not None and filename:
        files = {"file": (filename, file_bytes, "text/csv")}

    url = f"{settings.BACKEND_URL.rstrip('/')}/api/response"
    headers = {"BACKEND_KEY": settings.BACKEND_KEY}

    try:
        async with httpx.AsyncClient(timeout=120.0) as http:
            resp = await http.post(url, data=data, files=files, headers=headers)
    except httpx.RequestError as e:
        logger.exception("Cannot reach ML backend at %s", url)
        raise HTTPException(status_code=502, detail=f"Cannot reach ML backend: {e}")

    if resp.status_code != 200:
        # Surface the backend's own error message when possible.
        detail = resp.text
        try:
            detail = resp.json().get("detail", detail)
        except Exception:
            pass
        logger.warning("Backend %s -> %s: %s", url, resp.status_code, detail)
        raise HTTPException(status_code=resp.status_code, detail=f"Backend error: {detail}")

    return resp.json()
