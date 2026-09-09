"""
Document management over Supabase Storage.

Files are stored in the private `loanwise-documents` bucket under
`<application_id>/<uuid>.<ext>` so applicants cannot list or fetch each
other's documents (ownership is enforced both by path scope and by the
application-layer checks in api_server.py).
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from database import ensure_bucket, get_db

BUCKET = "loanwise-documents"

# Allowed extensions and their content types + size cap (5 MB).
ALLOWED_TYPES = {
    "application/pdf": ".pdf",
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "application/msword": ".doc",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
}
MAX_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB


def validate_upload(content_type: str, size: int) -> Optional[str]:
    """Return an error string if the upload should be rejected, else None."""
    if content_type not in ALLOWED_TYPES:
        return f"Unsupported file type: {content_type}. Allowed: {', '.join(sorted(ALLOWED_TYPES))}."
    if size <= 0:
        return "Uploaded file is empty."
    if size > MAX_SIZE_BYTES:
        return f"File exceeds the {MAX_SIZE_BYTES // (1024 * 1024)} MB size limit."
    return None


def upload_document(
    *,
    application_id: str,
    user_id: str,
    filename: str,
    content_type: str,
    data: bytes,
) -> Dict[str, Any]:
    db = get_db()
    if db is None:
        raise RuntimeError("Supabase is not configured.")

    ensure_bucket(BUCKET)

    extension = ALLOWED_TYPES.get(content_type, "")
    storage_path = f"{application_id}/{uuid.uuid4().hex}{extension}"
    size = len(data)

    db.storage.from_(BUCKET).upload(storage_path, data, {"content-type": content_type})

    row = {
        "application_id": application_id,
        "user_id": user_id,
        "filename": filename,
        "content_type": content_type,
        "size_bytes": size,
        "storage_path": storage_path,
    }
    resp = db.from_("documents").insert(row).execute()
    inserted = (resp.data or [{}])[0]
    return inserted


def list_documents(application_id: str) -> List[Dict[str, Any]]:
    db = get_db()
    if db is None:
        return []
    try:
        resp = (
            db.from_("documents")
            .select("id, application_id, filename, content_type, size_bytes, storage_path, uploaded_at")
            .eq("application_id", application_id)
            .order("uploaded_at", desc=True)
            .execute()
        )
        rows = resp.data or []
        result = []
        for row in rows:
            result.append({
                "id": row.get("id"),
                "application_id": row.get("application_id"),
                "filename": row.get("filename"),
                "content_type": row.get("content_type"),
                "size_bytes": row.get("size_bytes"),
                "uploaded_at": row.get("uploaded_at"),
            })
        return result
    except Exception as exc:
        _ = exc
        return []


def create_signed_url(application_id: str, row_id: str) -> Optional[str]:
    """Return a short-lived download URL scoped to the given document."""
    db = get_db()
    if db is None:
        return None
    try:
        resp = (
            db.from_("documents")
            .select("storage_path, application_id")
            .eq("id", row_id)
            .eq("application_id", application_id)
            .limit(1)
            .execute()
        )
        rows = resp.data or []
        if not rows:
            return None
        signed = db.storage.from_(BUCKET).create_signed_url(rows[0]["storage_path"], 3600)
        return (signed or {}).get("signedURL")
    except Exception as exc:
        _ = exc
        return None