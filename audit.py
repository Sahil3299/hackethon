"""
Append-only immutable audit log.

Every meaningful state change writes a row to the `audit_log` table
(application created, risk computed, officer decision, document uploaded).
There is intentionally NO update or delete endpoint for audit rows.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from database import get_db


def log_audit(
    *,
    action: str,
    application_id: Optional[str] = None,
    actor: Optional[str] = None,
    actor_label: str = "system",
    before: Optional[Dict[str, Any]] = None,
    after: Optional[Dict[str, Any]] = None,
) -> None:
    db = get_db()
    if db is None:
        return
    row = {
        "application_id": application_id,
        "actor": actor,
        "actor_label": actor_label,
        "action": action,
        "before": _safe_json(before),
        "after": _safe_json(after),
    }
    try:
        db.from_("audit_log").insert(row).execute()
    except Exception as exc:
        _ = exc


def _safe_json(value: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Best-effort make a snapshot JSON-serializable (Supabase stores it as jsonb)."""
    if value is None:
        return None
    try:
        return json.loads(json.dumps(value, default=str))
    except Exception:
        return {"_serialization_error": str(value)}


def list_audit(
    *,
    application_id: Optional[str] = None,
    page: int = 1,
    page_size: int = 50,
) -> Dict[str, Any]:
    db = get_db()
    if db is None:
        return {"items": [], "page": page, "page_size": page_size, "total": 0}

    query = db.from_("audit_log").select("*", count="exact").order("created_at", desc=True)
    if application_id:
        query = query.eq("application_id", application_id)
    query = query.range((page - 1) * page_size, page * page_size - 1)

    try:
        resp = query.execute()
        items = resp.data or []
        total = resp.count if hasattr(resp, "count") else len(items)
        return {"items": items, "page": page, "page_size": page_size, "total": total}
    except Exception as exc:
        _ = exc
        return {"items": [], "page": page, "page_size": page_size, "total": 0}
