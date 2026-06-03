"""Database engine + session helpers (SQLModel / SQLAlchemy)."""
import os
from sqlmodel import SQLModel, Session, create_engine
from .config import settings

# SQLite needs check_same_thread=False when used across FastAPI's threadpool.
connect_args = (
    {"check_same_thread": False}
    if settings.APP_DATABASE_URL.startswith("sqlite")
    else {}
)

engine = create_engine(settings.APP_DATABASE_URL, echo=False, connect_args=connect_args)


def init_db() -> None:
    """Create tables and the dataset directory on startup."""
    os.makedirs(settings.DATASET_DIR, exist_ok=True)
    # Import models so SQLModel registers the tables before create_all.
    from . import models  # noqa: F401
    SQLModel.metadata.create_all(engine)


def get_session():
    """FastAPI dependency: yields a DB session."""
    with Session(engine) as session:
        yield session
