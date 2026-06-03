"""Request / response schemas exchanged with the browser."""
from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel


class LoginRequest(BaseModel):
    key: str


class LoginResponse(BaseModel):
    token: str
    username: str
    expires_in: int  # seconds


class ChatCreate(BaseModel):
    title: Optional[str] = None


class ChatOut(BaseModel):
    id: str
    title: str
    dataset_filename: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    data: Optional[dict[str, Any]] = None
    created_at: datetime
