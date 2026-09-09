"""
Thin wrapper around the Supabase client.

Uses the service-role key server-side (bypasses RLS), so this module must
NEVER be imported or used on the frontend. Secrets come from env vars only.
"""

from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

_sb = None


def get_db():
    """
    Lazily build and return the shared Supabase client.

    If credentials are missing (e.g. local dev without a configured
    project) this returns None and callers are expected to handle it.
    """
    global _sb
    if _sb is not None:
        return _sb

    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        return None

    from supabase import create_client

    _sb = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    return _sb


def is_configured() -> bool:
    return bool(SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY)


def ensure_bucket(name: str = "loanwise-documents") -> None:
    """Create the private document bucket if it does not exist (idempotent)."""
    db = get_db()
    if db is None:
        return
    try:
        db.storage.get_bucket(name)
    except Exception:
        db.storage.create_bucket(name, options={"public": False})
