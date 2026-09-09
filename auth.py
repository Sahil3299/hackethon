"""
Supabase Auth integration for FastAPI.

Verifies the Supabase-issued JWT on every protected request, loads the
user's role from the `profiles` table, and injects the current user into
the request.

Supabase signs access tokens with HS256 using the project's JWT secret
(the `SUPABASE_JWT_SECRET`). We verify locally with PyJWT rather than
round-tripping to the Auth server on every request.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from dotenv import load_dotenv

from database import get_db

load_dotenv()

_scheme = HTTPBearer(auto_error=False)

JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET") or os.getenv("SUPABASE_SECRET_KEY")


class CurrentUser:
    """The authenticated user injected into request handlers."""

    def __init__(self, *, id: str, email: Optional[str], role: str, metadata: Dict[str, Any]):
        self.id = id
        self.email = email
        self.role = role
        self.metadata = metadata

    @property
    def is_loan_officer(self) -> bool:
        return self.role == "loan_officer"

    @property
    def is_applicant(self) -> bool:
        return self.role == "applicant"


def _decode_token(token: str) -> Dict[str, Any]:
    if not JWT_SECRET:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Supabase JWT secret is not configured on the server.",
        )
    try:
        return jwt.decode(
            token,
            JWT_SECRET,
            algorithms=["HS256"],
            options={"verify_aud": False, "verify_exp": True, "verify_iat": True},
        )
    except jwt.ExpiredSignatureError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Session token has expired."
        ) from exc
    except jwt.InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication token."
        ) from exc


def _load_role(db, user_id: str) -> Optional[str]:
    try:
        resp = db.from_("profiles").select("role").eq("id", user_id).limit(1).execute()
        rows = resp.data or []
        if rows:
            return rows[0].get("role")
    except Exception as exc:  # pragma: no cover - defensive
        _ = exc
    return None


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_scheme),
) -> CurrentUser:
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = _decode_token(credentials.credentials)

    user_id = str(payload.get("sub") or "")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Token missing subject."
        )

    email = payload.get("email")
    user_metadata = payload.get("user_metadata") or payload.get("app_metadata") or {}
    role = user_metadata.get("role")

    # Fall back to the profiles table if the role is not embedded in the token.
    if not role:
        db = get_db()
        role = _load_role(db, user_id) or "applicant"

    return CurrentUser(
        id=user_id,
        email=email,
        role=role if role in ("applicant", "loan_officer") else "applicant",
        metadata=user_metadata,
    )


async def require_loan_officer(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
    if not user.is_loan_officer:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This operation requires a loan officer role.",
        )
    return user


async def require_applicant(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
    if not user.is_applicant:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This operation requires an applicant role.",
        )
    return user
