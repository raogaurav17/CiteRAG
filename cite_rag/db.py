"""
PostgreSQL helpers for authentication, authorization, and audit logging.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from typing import Any, Optional

import psycopg2
from psycopg2.extras import RealDictCursor


def _get_postgres_config(cfg: Optional[Any] = None) -> dict[str, Any]:
    postgres_cfg = getattr(cfg, "postgres", None)

    return {
        "database_url": os.getenv("DATABASE_URL") or getattr(postgres_cfg, "database_url", ""),
        "host": os.getenv("POSTGRES_HOST") or getattr(postgres_cfg, "host", "localhost"),
        "port": int(os.getenv("POSTGRES_PORT") or getattr(postgres_cfg, "port", 5432)),
        "dbname": os.getenv("POSTGRES_DB") or getattr(postgres_cfg, "dbname", "citerag"),
        "user": os.getenv("POSTGRES_USER") or getattr(postgres_cfg, "user", "citerag"),
        "password": os.getenv("POSTGRES_PASSWORD") or getattr(postgres_cfg, "password", "citerag"),
        "sslmode": os.getenv("POSTGRES_SSLMODE") or getattr(postgres_cfg, "sslmode", "prefer"),
    }


def get_connection(cfg: Optional[Any] = None):
    """Create a new PostgreSQL connection."""

    postgres = _get_postgres_config(cfg)
    if postgres["database_url"]:
        return psycopg2.connect(postgres["database_url"], cursor_factory=RealDictCursor)

    return psycopg2.connect(
        host=postgres["host"],
        port=postgres["port"],
        dbname=postgres["dbname"],
        user=postgres["user"],
        password=postgres["password"],
        sslmode=postgres["sslmode"],
        cursor_factory=RealDictCursor,
    )


@contextmanager
def db_cursor(cfg: Optional[Any] = None):
    """Yield a cursor with automatic commit/rollback handling."""

    conn = get_connection(cfg)
    try:
        with conn:
            with conn.cursor() as cursor:
                yield cursor
    finally:
        conn.close()


def initialize_schema(cfg: Optional[Any] = None) -> None:
    """Create the PostgreSQL tables needed for auth and audit logging."""

    with db_cursor(cfg) as cursor:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id UUID PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                display_name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('admin', 'editor', 'viewer')),
                is_active BOOLEAN NOT NULL DEFAULT TRUE,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                last_login_at TIMESTAMPTZ
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS auth_events (
                id BIGSERIAL PRIMARY KEY,
                user_id UUID REFERENCES users(id) ON DELETE SET NULL,
                event_type TEXT NOT NULL,
                event_details JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id UUID PRIMARY KEY,
                owner_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                title TEXT NOT NULL,
                source_label TEXT,
                chunk_count INTEGER NOT NULL DEFAULT 0,
                content_hash TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_auth_events_user_id ON auth_events(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_owner_id ON documents(owner_id)")


def record_event(cfg: Optional[Any], user_id: Optional[str], event_type: str, details: Optional[dict[str, Any]] = None) -> None:
    """Write a user action to the audit log."""

    with db_cursor(cfg) as cursor:
        cursor.execute(
            """
            INSERT INTO auth_events (user_id, event_type, event_details)
            VALUES (%s, %s, %s::jsonb)
            """,
            (user_id, event_type, json.dumps(details or {})),
        )