"""
Shared test fixtures. Sets up environment before any app imports.
All tests use an in-memory SQLite database that is created fresh per test
and never touches the real production database.
"""

import os
import sys

os.environ.setdefault("EDUKAAI_ALLOW_REMOTE", "true")
os.environ.setdefault("EDUKAAI_ENV", "testing")

backend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
backend_dir = os.path.normpath(backend_dir)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

import pytest
from unittest.mock import patch
from sqlalchemy import create_engine, event, StaticPool
from sqlalchemy.orm import sessionmaker, Session
from app.models import Base, get_db


def _make_test_db():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        echo=False,
        poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.close()

    Base.metadata.create_all(bind=engine)
    TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return engine, TestSessionLocal


@pytest.fixture(autouse=True)
def _use_test_db():
    import app.models as models
    from app.config import Settings
    from app.main import app as fastapi_app

    engine, TestSessionLocal = _make_test_db()

    saved_engine = models._engine
    saved_session = models._SessionLocal
    saved_init_db = models.init_db

    models._engine = engine
    models._SessionLocal = TestSessionLocal

    def _override_get_db():
        db = TestSessionLocal()
        try:
            yield db
        finally:
            db.close()

    def _test_init_db(database_url=None, force_recreate=False):
        models._engine = engine
        models._SessionLocal = TestSessionLocal
        fastapi_app.dependency_overrides[get_db] = _override_get_db
        return engine

    models.init_db = _test_init_db
    fastapi_app.dependency_overrides[get_db] = _override_get_db

    test_settings = Settings(
        allow_remote=True,
        debug=True,
        secret_key="test-secret-key-for-pytest",
    )

    with patch("app.main.get_settings", return_value=test_settings):
        yield

    fastapi_app.dependency_overrides.pop(get_db, None)
    models.init_db = saved_init_db
    models._engine = saved_engine
    models._SessionLocal = saved_session
    engine.dispose()


@pytest.fixture()
def test_db_session() -> Session:
    """Get a session for the in-memory test database."""
    import app.models as models
    db = models._SessionLocal()
    try:
        yield db
    finally:
        db.close()
