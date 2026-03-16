from pydantic_settings import BaseSettings
from fastapi import HTTPException

class Settings(BaseSettings):
    APP_NAME: str = "Featurify"
    BACKEND_KEY: str
    AI_API_KEY: str

    class Config():
        env_file = ".env"
        case_sensitive = True

settings = Settings()

def validate_key(backend_key: str):
    if not backend_key:
        raise HTTPException(status_code=500, detail="Server misconfiguration: AI_API_KEY is not set")
    if backend_key != settings.BACKEND_KEY:
        raise HTTPException(status_code=403, detail="Forbidden: Backend key is invalid")