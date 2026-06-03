"""
Configuration for the public-facing api-template layer.

This service is the ONLY thing the browser talks to. It holds the secrets
(BACKEND_KEY, JWT secret, access keys) server-side and forwards work to the
internal Featurify `backend` service.
"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "Featurify"

    # --- Talking to the internal Featurify backend ---
    # Inside the docker-compose network the backend service is reachable by its
    # service name. The backend listens on port 8000 *inside* its container.
    BACKEND_URL: str = "http://backend:8000"
    BACKEND_KEY: str  # MUST match the backend's BACKEND_KEY (same .env)

    # --- Auth ---
    # Comma-separated "username=key" pairs. Example:
    #   ACCESS_KEYS=alice=demo-key-alice-7f3a,bob=demo-key-bob-9c1d
    # Each key represents one demo "user". Revoke access by removing a pair.
    ACCESS_KEYS: str = ""
    JWT_SECRET: str  # long random string; sign-out everyone by changing it
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_HOURS: int = 12

    # --- Database ---
    # SQLite by default (single file, no extra container). To move to Postgres
    # later set APP_DATABASE_URL, e.g. postgresql+psycopg://user:pass@host/db
    # NOTE: named APP_DATABASE_URL (not DATABASE_URL) on purpose — the shared
    # .env already defines DATABASE_URL for litellm/langfuse, and a plain
    # DATABASE_URL field would inherit that Postgres value via env_file.
    APP_DATABASE_URL: str = "sqlite:////data/featurify.db"
    DATASET_DIR: str = "/data/datasets"

    # --- Request limits (protect the token budget) ---
    MAX_MESSAGE_LENGTH: int = 2000          # mirrors backend MAX_REQUEST_LENGTH
    MAX_FILE_SIZE_MB: int = 10
    RATE_LIMIT_PER_MINUTE: int = 8
    RATE_LIMIT_PER_DAY: int = 120

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # ignore the many backend-only vars in the shared .env

    @property
    def access_key_map(self) -> dict[str, str]:
        """Returns {key: username}, parsed from ACCESS_KEYS."""
        mapping: dict[str, str] = {}
        for pair in self.ACCESS_KEYS.split(","):
            pair = pair.strip()
            if not pair or "=" not in pair:
                continue
            username, key = pair.split("=", 1)
            username, key = username.strip(), key.strip()
            if username and key:
                mapping[key] = username
        return mapping


settings = Settings()
