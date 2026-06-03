"""Logging setup: writes to stdout AND a rotating file at /data/logs/latest.log.

The file lives on the persistent data volume so logs survive restarts. View it
with:
    docker compose logs -f api-template                  # stdout
    docker compose exec api-template tail -f /data/logs/latest.log
"""
import os
import logging
from logging.handlers import RotatingFileHandler

LOG_DIR = os.environ.get("LOG_DIR", "/data/logs")
LOG_FILE = os.path.join(LOG_DIR, "latest.log")

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure root + uvicorn loggers. Idempotent (safe to call twice)."""
    os.makedirs(LOG_DIR, exist_ok=True)
    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()          # avoid duplicate lines if called again
    root.addHandler(file_handler)
    root.addHandler(stream_handler)

    # Route uvicorn's loggers through our handlers (so its access/error logs
    # also land in latest.log instead of disappearing).
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.propagate = True

    return logging.getLogger("featurify")
