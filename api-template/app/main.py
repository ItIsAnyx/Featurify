"""Featurify public API + static frontend.

The browser talks ONLY to this service. It:
  - serves the single-page chat UI (static/)
  - handles login (access key -> JWT)
  - stores chats/messages in the database
  - forwards each turn to the internal Featurify backend (with BACKEND_KEY)
"""
import os
import time
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import settings
from .database import init_db
from .logging_config import setup_logging
from .routes import auth_routes, chat_routes

# Configure logging as early as possible.
logger = setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()  # re-assert after uvicorn has configured its own logging
    init_db()
    logger.info("Featurify web layer started")
    yield
    logger.info("Featurify web layer shutting down")


app = FastAPI(title=f"{settings.APP_NAME} Web", lifespan=lifespan)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - start) * 1000
    path = request.url.path
    if response.status_code >= 500:
        logger.error("%s %s -> %s (%.0f ms)", request.method, path, response.status_code, elapsed)
    elif path.startswith("/api") or path.startswith("/auth"):
        logger.info("%s %s -> %s (%.0f ms)", request.method, path, response.status_code, elapsed)
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    # Logs the full traceback to latest.log; returns a generic 500 to the client.
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


# API routers are registered BEFORE the static mount so they take precedence.
app.include_router(auth_routes.router)
app.include_router(chat_routes.router)

# Serve the SPA. html=True makes "/" return index.html.
_static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
app.mount("/", StaticFiles(directory=_static_dir, html=True), name="static")
