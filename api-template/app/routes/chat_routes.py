"""Chat and message routes.

All routes require a valid JWT. A user can only see and modify their own chats.
"""
import os
import json
from datetime import datetime, timezone

from fastapi import (
    APIRouter, Depends, HTTPException, UploadFile, File, Form,
)
from sqlmodel import Session, select

from ..config import settings
from ..database import get_session
from ..models import Chat, Message
from ..schemas import ChatCreate, ChatOut, MessageOut
from ..auth import get_current_user, enforce_rate_limit
from ..backend_client import call_featurify

router = APIRouter(prefix="/api", tags=["chats"])

# Assistant response fields we persist (everything except the analysis text,
# which is stored in Message.content).
# Assistant response fields we persist (everything except the analysis text,
# which is stored in Message.content). Token counts are intentionally NOT
# stored or returned — they aren't useful to end users and reveal prompt size.
_STRUCTURED_FIELDS = (
    "remove_features",
    "transform_features",
    "create_features",
    "recommended_models",
)


def _get_owned_chat(chat_id: str, username: str, session: Session) -> Chat:
    chat = session.get(Chat, chat_id)
    if not chat or chat.owner != username:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat


def _message_to_out(m: Message) -> MessageOut:
    data = json.loads(m.data) if m.data else None
    return MessageOut(
        id=m.id, role=m.role, content=m.content, data=data, created_at=m.created_at
    )


# --------------------------------------------------------------------------- #
# Chat CRUD
# --------------------------------------------------------------------------- #
@router.get("/chats", response_model=list[ChatOut])
def list_chats(
    username: str = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    chats = session.exec(
        select(Chat).where(Chat.owner == username).order_by(Chat.updated_at.desc())
    ).all()
    return [
        ChatOut(
            id=c.id, title=c.title, dataset_filename=c.dataset_filename,
            created_at=c.created_at, updated_at=c.updated_at,
        )
        for c in chats
    ]


@router.post("/chats", response_model=ChatOut)
def create_chat(
    body: ChatCreate,
    username: str = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    chat = Chat(owner=username, title=(body.title or "New chat").strip()[:120] or "New chat")
    session.add(chat)
    session.commit()
    session.refresh(chat)
    return ChatOut(
        id=chat.id, title=chat.title, dataset_filename=chat.dataset_filename,
        created_at=chat.created_at, updated_at=chat.updated_at,
    )


@router.delete("/chats/{chat_id}")
def delete_chat(
    chat_id: str,
    username: str = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    chat = _get_owned_chat(chat_id, username, session)

    # Remove messages and the sticky dataset file.
    msgs = session.exec(select(Message).where(Message.chat_id == chat_id)).all()
    for m in msgs:
        session.delete(m)
    if chat.dataset_path and os.path.exists(chat.dataset_path):
        try:
            os.remove(chat.dataset_path)
        except OSError:
            pass
    session.delete(chat)
    session.commit()
    return {"deleted": chat_id}


@router.get("/chats/{chat_id}/messages", response_model=list[MessageOut])
def get_messages(
    chat_id: str,
    username: str = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    _get_owned_chat(chat_id, username, session)
    msgs = session.exec(
        select(Message).where(Message.chat_id == chat_id).order_by(Message.id)
    ).all()
    return [_message_to_out(m) for m in msgs]


# --------------------------------------------------------------------------- #
# Send a message (the core endpoint)
# --------------------------------------------------------------------------- #
@router.post("/chats/{chat_id}/messages", response_model=MessageOut)
async def send_message(
    chat_id: str,
    message: str = Form(...),
    file: UploadFile = File(None),
    username: str = Depends(enforce_rate_limit),
    session: Session = Depends(get_session),
):
    chat = _get_owned_chat(chat_id, username, session)

    message = message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message is empty")
    if len(message) > settings.MAX_MESSAGE_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Message too long (max {settings.MAX_MESSAGE_LENGTH} chars)",
        )

    # --- Handle an optional CSV upload (stays sticky to the chat) ---
    file_bytes = None
    filename = None
    if file is not None and file.filename:
        if not file.filename.lower().endswith(".csv"):
            raise HTTPException(status_code=400, detail="Only CSV files are allowed")
        raw = await file.read()
        if len(raw) > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail=f"File too large (max {settings.MAX_FILE_SIZE_MB} MB)",
            )
        os.makedirs(settings.DATASET_DIR, exist_ok=True)
        path = os.path.join(settings.DATASET_DIR, f"{chat.id}.csv")
        with open(path, "wb") as f:
            f.write(raw)
        chat.dataset_filename = file.filename
        chat.dataset_path = path
        file_bytes, filename = raw, file.filename
    elif chat.dataset_path and os.path.exists(chat.dataset_path):
        # Re-send the previously uploaded dataset so it stays in context.
        with open(chat.dataset_path, "rb") as f:
            file_bytes = f.read()
        filename = chat.dataset_filename

    # --- Rebuild conversation context from stored messages ---
    # We keep only user/assistant text turns. This is robust and keeps the
    # dataset/system prompt fully under the backend's control each turn.
    prior = session.exec(
        select(Message).where(Message.chat_id == chat_id).order_by(Message.id)
    ).all()
    context = [{"role": m.role, "content": m.content} for m in prior]

    # --- Call the Featurify backend ---
    result = await call_featurify(
        message=message, context=context, file_bytes=file_bytes, filename=filename
    )

    analysis = result.get("analysis", "")
    structured = {k: result.get(k) for k in _STRUCTURED_FIELDS}

    # --- Persist both turns ---
    user_msg = Message(chat_id=chat.id, role="user", content=message)
    assistant_msg = Message(
        chat_id=chat.id, role="assistant", content=analysis, data=json.dumps(structured)
    )
    session.add(user_msg)
    session.add(assistant_msg)

    # Auto-title the chat from the first user message.
    if chat.title == "New chat" and not prior:
        chat.title = (message[:60] + "…") if len(message) > 60 else message
    chat.updated_at = datetime.now(timezone.utc)
    session.add(chat)

    session.commit()
    session.refresh(assistant_msg)
    return _message_to_out(assistant_msg)
