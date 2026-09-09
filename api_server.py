"""
LoanWise Recommendation API — production configuration.

Thin REST layer over the existing loan agents with Supabase Auth (JWT
verification), Supabase Postgres persistence, approval workflow, audit
trail, document storage, and model metrics.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from fastapi import Depends, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from pydantic import BaseModel, Field
from dotenv import load_dotenv


from audit import log_audit, list_audit
from auth import CurrentUser, get_current_user, require_loan_officer
from database import get_db, is_configured
from documents_store import (
    create_signed_url,
    list_documents,
    upload_document,
    validate_upload,
)
from pipeline_service import pipeline
from adverse_action import build_adverse_action
from guardrail import validate_explanation

load_dotenv()

FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")

app = FastAPI(
    title="LoanWise Recommendation API",
    description="HTTP facade for the existing Personalized Loan Recommendation agents.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        FRONTEND_ORIGIN,
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / response models ─────────────────────────────────────────

class IntakeRequest(BaseModel):
    customer_id: Optional[str] = None
    full_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    employer: Optional[str] = None
    employment_type: Optional[str] = None
    loan_purpose: Optional[str] = None
    monthly_income: float = Field(..., gt=0)
    age: int = Field(..., ge=21, le=70)
    gender: str
    house_rent: float = Field(0, ge=0)
    existing_emi: float = Field(0, ge=0)
    cibil_score: int = Field(..., ge=300, le=850)
    requested_loan: float = Field(..., gt=0)
    requested_tenure: int = Field(..., ge=6, le=84)
    early_closure_month: Optional[int] = Field(18, ge=1)
    include_explanation: bool = True


class SimulateRequest(BaseModel):
    application_id: Optional[str] = None
    loan_amount: float = Field(..., gt=0)
    tenure_months: int = Field(..., ge=6, le=84)
    early_closure_month: Optional[int] = 18
    interest_rate: Optional[float] = None
    monthly_income: Optional[float] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    house_rent: Optional[float] = 0
    existing_emi: Optional[float] = 0
    cibil_score: Optional[int] = None
    customer_id: Optional[str] = None


class ApprovalDecisionRequest(BaseModel):
    decision: str = Field(..., pattern=r"^(APPROVED|REJECTED|COUNTER_OFFERED|WITHDRAWN)$")
    note: Optional[str] = Field(None, max_length=2000)


# ── Health endpoints ──────────────────────────────────────────────────

@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "loanwise-api",
        "version": "2.0.0",
        "agents": [
            "CustomerProfilingAgentLLM",
            "AffordabilityAgent",
            "RiskMLAgent",
            "PolicyRAGAgent",
            "OfferDiscountAgentLLM",
            "LoanSimulatorAgent",
            "ComplianceExplanationAgentLLM",
        ],
        "auth": "supabase_jwt",
        "database": "supabase" if is_configured() else "in_memory",
        "max_foir": 0.60,
        "base_rate": 11.5,
    }


@app.get("/api/health")
def api_health() -> dict[str, Any]:
    return health()


# ── Recommendations ───────────────────────────────────────────────────

@app.post("/api/recommendations")
def create_recommendation(
    body: IntakeRequest,
    user: CurrentUser = Depends(get_current_user),
) -> dict[str, Any]:
    try:
        record = pipeline.evaluate(
            body.model_dump(),
            include_explanation=body.include_explanation,
            user_id=user.id,
            status="PENDING_REVIEW",
        )
        adverse = None
        decision_type = (record.get("affordability") or {}).get("decision_type", "")
        if decision_type in ("REJECTED", "COUNTER_OFFER"):
            adverse = build_adverse_action(record)
        if body.include_explanation and record.get("explanation"):
            guard = validate_explanation(record.get("explanation"), record)
            if not guard["pass"]:
                record["explanation"] = None
                record["explanation_error"] = "AI explanation contained numerical mismatches. Falling back to rule-based rationale."
                log_audit(
                    action="explanation_guardrail_mismatch",
                    application_id=record["application_id"],
                    actor=user.id,
                    after={"mismatches": guard["mismatches"]},
                )
        log_audit(
            action="application_created",
            application_id=record["application_id"],
            actor=user.id,
            actor_label=user.email or user.id,
            after=record,
        )
        result = dict(record)
        if adverse:
            result["adverse_action"] = adverse
        return result
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Unable to generate recommendation. The recommendation service encountered an error.",
        ) from exc


@app.get("/api/recommendations/{application_id}")
def get_recommendation(
    application_id: str,
    user: CurrentUser = Depends(get_current_user),
) -> dict[str, Any]:
    record = pipeline.get_application(application_id)
    if not record:
        raise HTTPException(status_code=404, detail="Application not found.")
    _check_access(user, record)
    return _with_adverse_action(record)


def _with_adverse_action(record: dict[str, Any]) -> dict[str, Any]:
    decision_type = (record.get("affordability") or {}).get("decision_type", "")
    result = dict(record)
    if decision_type in ("REJECTED", "COUNTER_OFFER") and "adverse_action" not in result:
        result["adverse_action"] = build_adverse_action(record)
    return result


@app.get("/api/applications")
def list_applications(
    user: CurrentUser = Depends(get_current_user),
) -> dict[str, Any]:
    items = pipeline.list_applications(user_id=user.id, officer=user.is_loan_officer)
    return {"items": items}


@app.get("/api/customers")
def list_customers(
    user: CurrentUser = Depends(get_current_user),
) -> dict[str, Any]:
    items = pipeline.list_applications(user_id=user.id, officer=user.is_loan_officer)
    return {"items": items}


@app.get("/api/customers/{application_id}")
def get_customer(
    application_id: str,
    user: CurrentUser = Depends(get_current_user),
) -> dict[str, Any]:
    record = pipeline.get_application(application_id)
    if not record:
        raise HTTPException(status_code=404, detail="Customer application not found.")
    _check_access(user, record)
    return _with_adverse_action(record)


# ── Simulator / Foreclosure ──────────────────────────────────────────

@app.post("/api/simulator")
def simulate(
    body: SimulateRequest,
    user: CurrentUser = Depends(get_current_user),
) -> dict[str, Any]:
    payload = body.model_dump()
    if not payload.get("application_id"):
        required = ["monthly_income", "age", "gender", "cibil_score"]
        missing = [key for key in required if payload.get(key) is None]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Provide application_id or these fields: {', '.join(missing)}",
            )
        payload["requested_loan"] = payload["loan_amount"]
        payload["requested_tenure"] = payload["tenure_months"]
    try:
        return pipeline.simulate(payload)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Unable to run loan simulation.") from exc


@app.post("/api/foreclosure")
def foreclosure(
    body: SimulateRequest,
    user: CurrentUser = Depends(get_current_user),
) -> dict[str, Any]:
    payload = body.model_dump()
    if not payload.get("application_id"):
        required = ["monthly_income", "age", "gender", "cibil_score"]
        missing = [key for key in required if payload.get(key) is None]
        if missing:
            raise HTTPException(
                status_code=400,
                detail="Provide application_id or a complete borrower profile.",
            )
        payload["requested_loan"] = payload["loan_amount"]
        payload["requested_tenure"] = payload["tenure_months"]
    try:
        return pipeline.foreclosure(payload)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Unable to calculate early closure.") from exc


# ── Explainability / Advisor ─────────────────────────────────────────

@app.get("/api/explainability/{application_id}")
def explainability(
    application_id: str,
    user: CurrentUser = Depends(get_current_user),
) -> dict[str, Any]:
    record = pipeline.get_application(application_id)
    if not record:
        raise HTTPException(status_code=404, detail="Application not found.")
    _check_access(user, record)
    return {
        "application_id": application_id,
        "policy_fit": record.get("policy_fit"),
        "risk": record.get("risk"),
        "affordability": {
            "decision_type": (record.get("affordability") or {}).get("decision_type"),
            "reason": (record.get("affordability") or {}).get("reason"),
            "eligibility_message": (record.get("affordability") or {}).get("eligibility_message"),
            "foir": (record.get("affordability") or {}).get("foir"),
            "requested_foir": (record.get("affordability") or {}).get("requested_foir"),
        },
        "explanation": record.get("explanation"),
        "concessions": record.get("concessions"),
    }


@app.get("/api/advisor/{application_id}")
def advisor(
    application_id: str,
    user: CurrentUser = Depends(get_current_user),
) -> dict[str, Any]:
    record = pipeline.get_application(application_id)
    if not record:
        raise HTTPException(status_code=404, detail="Application not found.")
    _check_access(user, record)
    adverse = None
    decision_type = (record.get("affordability") or {}).get("decision_type", "")
    if decision_type in ("REJECTED", "COUNTER_OFFER"):
        adverse = build_adverse_action(record)
    return {
        "application_id": application_id,
        "explanation": record.get("explanation"),
        "explanation_error": record.get("explanation_error"),
        "disclaimer": "AI-generated guidance. It does not make the final lending decision.",
        "adverse_action": adverse,
    }


# ── Approval workflow ────────────────────────────────────────────────

@app.get("/api/approvals")
def approvals(
    user: CurrentUser = Depends(require_loan_officer),
    status: str = Query("PENDING_REVIEW", alias="status"),
) -> dict[str, Any]:
    db = get_db()
    if db is None:
        items = pipeline.list_applications(officer=True)
        filtered = [i for i in items if i.get("status") == status]
        return {"supported": True, "items": filtered}
    try:
        resp = db.from_("applications").select("*").eq("status", status).order("created_at", desc=True).execute()
        rows = resp.data or []
        items = []
        for row in rows:
            record = row.get("record") or {}
            record.setdefault("application_id", row.get("application_id"))
            items.append(pipeline._summary_from(record, row))
        return {"supported": True, "items": items}
    except Exception as exc:
        _ = exc
        raise HTTPException(status_code=500, detail="Unable to load approvals queue.")


@app.post("/api/approvals/{application_id}/decision")
def make_decision(
    application_id: str,
    body: ApprovalDecisionRequest,
    user: CurrentUser = Depends(require_loan_officer),
) -> dict[str, Any]:
    summary = pipeline.make_decision(
        application_id=application_id,
        decision=body.decision,
        note=body.note,
        decided_by=user.id,
    )
    if summary is None:
        raise HTTPException(status_code=404, detail="Application not found.")
    log_audit(
        action=f"officer_decision:{body.decision}",
        application_id=application_id,
        actor=user.id,
        actor_label=user.email or user.id,
        before={"status": summary["previous_status"]},
        after={"status": body.decision, "note": body.note or ""},
    )
    return {
        "application_id": application_id,
        "previous_status": summary["previous_status"],
        "new_status": summary["new_status"],
        "decision_note": body.note or "",
        "decided_by": user.email or user.id,
    }


# ── Audit trail ──────────────────────────────────────────────────────

@app.get("/api/audit")
def audit(
    user: CurrentUser = Depends(require_loan_officer),
    application_id: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
) -> dict[str, Any]:
    result = list_audit(application_id=application_id, page=page, page_size=page_size)
    return result


# ── Document management ──────────────────────────────────────────────

@app.get("/api/documents")
def documents_list(
    user: CurrentUser = Depends(get_current_user),
) -> dict[str, Any]:
    return {
        "supported": True,
        "items": [],
        "message": "Use GET /api/documents/{application_id} for application-scoped documents.",
    }


@app.get("/api/documents/{application_id}")
def documents_list_application(
    application_id: str,
    user: CurrentUser = Depends(get_current_user),
) -> dict[str, Any]:
    if not is_configured():
        return {"supported": False, "items": [], "message": "Document storage is not configured."}
    record = pipeline.get_application(application_id)
    if not record:
        raise HTTPException(status_code=404, detail="Application not found.")
    _check_access(user, record)
    items = list_documents(application_id)
    return {"supported": True, "items": items}


@app.post("/api/documents/{application_id}")
async def document_upload(
    application_id: str,
    file: UploadFile = File(...),
    user: CurrentUser = Depends(get_current_user),
) -> dict[str, Any]:
    if not is_configured():
        raise HTTPException(status_code=503, detail="Document storage is not configured.")
    record = pipeline.get_application(application_id)
    if not record:
        raise HTTPException(status_code=404, detail="Application not found.")
    _check_access(user, record)
    content_type = file.content_type or "application/octet-stream"
    data = await file.read()
    error = validate_upload(content_type, len(data))
    if error:
        raise HTTPException(status_code=400, detail=error)
    result = upload_document(
        application_id=application_id,
        user_id=user.id,
        filename=file.filename or "unnamed",
        content_type=content_type,
        data=data,
    )
    log_audit(
        action="document_uploaded",
        application_id=application_id,
        actor=user.id,
        actor_label=user.email or user.id,
        after={"filename": file.filename, "size_bytes": len(data), "content_type": content_type},
    )
    return {"ok": True, "document": result}


@app.get("/api/documents/{application_id}/{doc_id}/download")
def document_download(
    application_id: str,
    doc_id: str,
    user: CurrentUser = Depends(get_current_user),
) -> dict[str, Any]:
    if not is_configured():
        raise HTTPException(status_code=503, detail="Document storage is not configured.")
    url = create_signed_url(application_id, doc_id)
    if not url:
        raise HTTPException(status_code=404, detail="Document not found.")
    return {"url": url}


# ── Model metrics (staff only) ──────────────────────────────────────

@app.get("/api/model/metrics")
def model_metrics(
    user: CurrentUser = Depends(require_loan_officer),
) -> dict[str, Any]:
    return pipeline.risk_engine.get_metrics()


@app.post("/api/model/retrain")
def model_retrain(
    user: CurrentUser = Depends(require_loan_officer),
) -> dict[str, Any]:
    try:
        pipeline.risk_engine.retrain()
        return {"ok": True, "metrics": pipeline.risk_engine.get_metrics()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Retrain failed: {exc}")


# ── Access-control helper ────────────────────────────────────────────

def _check_access(user: CurrentUser, record: dict[str, Any]) -> None:
    if user.is_loan_officer:
        return
    app_user_id = record.get("user_id")
    if app_user_id and app_user_id != user.id:
        raise HTTPException(status_code=403, detail="Access denied.")


# ── Entry point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
