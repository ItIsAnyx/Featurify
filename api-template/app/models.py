"""Database models: Chat and Message.

We deliberately do NOT store a `users` table. Users are defined by the
ACCESS_KEYS env var; a chat simply records its `owner` (the username encoded
in the JWT). This keeps the demo free of user-management code while still
isolating each demo user's chats.
"""
import uuid
from datetime import datetime, timezone
from typing import Optional
from sqlmodel import SQLModel, Field


def _uuid() -> str:
    return uuid.uuid4().hex


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Chat(SQLModel, table=True):
    id: str = Field(default_factory=_uuid, primary_key=True)
    owner: str = Field(index=True)            # username from the JWT
    title: str = "New chat"

    # A CSV stays "sticky" to a chat: once uploaded it is re-sent to the
    # backend on every following turn so the dataset remains in context.
    dataset_filename: Optional[str] = None
    dataset_path: Optional[str] = None

    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


class Message(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    chat_id: str = Field(index=True, foreign_key="chat.id")
    role: str                                 # "user" | "assistant"
    content: str                              # user text, or assistant analysis

    # For assistant turns we keep the structured Featurify output (feature
    # lists, recommended models, token usage) as a JSON string for rendering.
    data: Optional[str] = None

    created_at: datetime = Field(default_factory=_now)
