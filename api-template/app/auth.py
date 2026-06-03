"""Authentication (access-key -> JWT) and a simple in-memory rate limiter.

Flow:
  1. Browser POSTs an access key to /auth/login.
  2. We look the key up in ACCESS_KEYS. If found, we issue a signed JWT whose
     subject is the username.
  3. Every protected endpoint requires `Authorization: Bearer <token>`.
"""
import time
import threading
from collections import defaultdict, deque
from datetime import datetime, timezone, timedelta

import jwt
from fastapi import Depends, HTTPException, Header

from .config import settings


# --------------------------------------------------------------------------- #
# JWT
# --------------------------------------------------------------------------- #
def create_access_token(username: str) -> tuple[str, int]:
    expires_seconds = settings.JWT_EXPIRE_HOURS * 3600
    now = datetime.now(timezone.utc)
    payload = {
        "sub": username,
        "iat": now,
        "exp": now + timedelta(seconds=expires_seconds),
    }
    token = jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)
    return token, expires_seconds


def authenticate_key(key: str) -> str:
    """Return the username for a valid access key, else raise 401."""
    username = settings.access_key_map.get(key.strip())
    if not username:
        raise HTTPException(status_code=401, detail="Invalid access key")
    return username


def get_current_user(authorization: str = Header(None)) -> str:
    """FastAPI dependency. Validates the Bearer token and returns the username."""
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.split(" ", 1)[1].strip()
    try:
        payload = jwt.decode(
            token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM]
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Session expired, please log in again")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

    username = payload.get("sub")
    if not username:
        raise HTTPException(status_code=401, detail="Invalid token")
    return username


# --------------------------------------------------------------------------- #
# Rate limiting (per username)
# --------------------------------------------------------------------------- #
class RateLimiter:
    def __init__(self, per_minute: int, per_day: int):
        self.per_minute = per_minute
        self.per_day = per_day
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, username: str) -> None:
        now = time.time()
        minute_ago = now - 60
        day_ago = now - 86400
        with self._lock:
            hits = self._hits[username]
            # Drop entries older than a day.
            while hits and hits[0] < day_ago:
                hits.popleft()
            last_minute = sum(1 for t in hits if t >= minute_ago)
            if last_minute >= self.per_minute:
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit: too many requests per minute. Slow down a moment.",
                )
            if len(hits) >= self.per_day:
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit: daily request quota reached for this key.",
                )
            hits.append(now)


rate_limiter = RateLimiter(
    per_minute=settings.RATE_LIMIT_PER_MINUTE,
    per_day=settings.RATE_LIMIT_PER_DAY,
)


def enforce_rate_limit(username: str = Depends(get_current_user)) -> str:
    rate_limiter.check(username)
    return username
