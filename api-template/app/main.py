import time
import traceback

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import InfoResponse, RunRequest, RunResponse
from app.service import get_service

app = FastAPI(
    title="API Template",
    description="Unified API wrapper template for student projects.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TimingMiddleware(BaseHTTPMiddleware):
    """Middleware that logs request processing time."""

    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.time()
        response = await call_next(request)
        elapsed = time.time() - start
        print(f"[timing] {request.method} {request.url.path} — {elapsed:.3f}s")
        return response


app.add_middleware(TimingMiddleware)

service = get_service()


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/info", response_model=InfoResponse)
def info() -> InfoResponse:
    return service.get_info()


@app.post("/run", response_model=RunResponse)
def run(request: RunRequest) -> RunResponse:
    try:
        return service.run(request)
    except Exception:
        return RunResponse(
            status="error",
            error=traceback.format_exc(),
        )
