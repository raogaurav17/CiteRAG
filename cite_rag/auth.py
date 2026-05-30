"""
Authentication helpers backed by PostgreSQL.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import uuid
from typing import Any, Optional

from cite_rag.db import db_cursor, initialize_schema, record_event

PASSWORD_ITERATIONS = 210_000
HASH_SCHEME = "pbkdf2_sha256"


def ensure_auth_storage(cfg: Optional[Any] = None) -> None:
    initialize_schema(cfg)


def hash_password(password: str) -> str:
    """Return a salted PBKDF2 password hash."""

    salt = secrets.token_bytes(16)
    password_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PASSWORD_ITERATIONS,
    )
    salt_b64 = base64.urlsafe_b64encode(salt).decode("ascii")
    hash_b64 = base64.urlsafe_b64encode(password_hash).decode("ascii")
    return f"{HASH_SCHEME}${PASSWORD_ITERATIONS}${salt_b64}${hash_b64}"


def verify_password(password: str, stored_hash: str) -> bool:
    """Verify a password against the stored PBKDF2 hash."""

    try:
        scheme, iterations, salt_b64, hash_b64 = stored_hash.split("$", 3)
        if scheme != HASH_SCHEME:
            return False

        salt = base64.urlsafe_b64decode(salt_b64.encode("ascii"))
        expected_hash = base64.urlsafe_b64decode(hash_b64.encode("ascii"))
        candidate_hash = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            int(iterations),
        )
        return hmac.compare_digest(candidate_hash, expected_hash)
    except Exception:
        return False


def _row_to_user(row: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if not row:
        return None

    return {
        "id": str(row["id"]),
        "email": row["email"],
        "display_name": row["display_name"],
        "role": row["role"],
        "is_active": row["is_active"],
        "created_at": row.get("created_at"),
        "last_login_at": row.get("last_login_at"),
    }


def get_user_by_email(cfg: Optional[Any], email: str) -> Optional[dict[str, Any]]:
    normalized_email = email.strip().lower()
    with db_cursor(cfg) as cursor:
        cursor.execute(
            """
            SELECT id, email, display_name, password_hash, role, is_active, created_at, last_login_at
            FROM users
            WHERE lower(email) = lower(%s)
            """,
            (normalized_email,),
        )
        return cursor.fetchone()


def count_users(cfg: Optional[Any]) -> int:
    with db_cursor(cfg) as cursor:
        cursor.execute("SELECT COUNT(*) AS count FROM users")
        return int(cursor.fetchone()["count"])


def create_user(cfg: Optional[Any], display_name: str, email: str, password: str) -> dict[str, Any]:
    existing_user = get_user_by_email(cfg, email)
    if existing_user:
        raise ValueError("An account with this email already exists.")

    user_count = count_users(cfg)
    auth_cfg = getattr(cfg, "auth", None)
    default_role = getattr(auth_cfg, "default_role", "viewer")
    first_user_role = getattr(auth_cfg, "first_user_role", "admin")
    role = first_user_role if user_count == 0 else default_role

    user_id = str(uuid.uuid4())
    normalized_email = email.strip().lower()

    with db_cursor(cfg) as cursor:
        cursor.execute(
            """
            INSERT INTO users (id, email, display_name, password_hash, role)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id, email, display_name, role, is_active, created_at, last_login_at
            """,
            (user_id, normalized_email, display_name.strip(), hash_password(password), role),
        )
        user = cursor.fetchone()

    record_event(cfg, user_id, "signup", {"role": role})
    return _row_to_user(user)


def authenticate_user(cfg: Optional[Any], email: str, password: str) -> Optional[dict[str, Any]]:
    user = get_user_by_email(cfg, email)
    if not user or not user.get("is_active"):
        return None

    if not verify_password(password, user["password_hash"]):
        return None

    with db_cursor(cfg) as cursor:
        cursor.execute(
            """
            UPDATE users
            SET last_login_at = NOW(), updated_at = NOW()
            WHERE id = %s
            RETURNING id, email, display_name, role, is_active, created_at, last_login_at
            """,
            (user["id"],),
        )
        authenticated_user = cursor.fetchone()

    record_event(cfg, str(user["id"]), "login")
    return _row_to_user(authenticated_user)


def is_admin(user: Optional[dict[str, Any]]) -> bool:
    return bool(user and user.get("role") == "admin")