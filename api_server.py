"""
Thin REST API over the existing loan agents.

Recommendation, FOIR, risk, discounts, EMI, and foreclosure math
are delegated to the original Python agents.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from pipeline_service import pipeline

load_dotenv()

FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")

app = FastAPI(
    title="LoanWise Recommendation API",
    description="HTTP facade for the existing Personalized Loan Recommendation agents.",
    version="1.0.0",
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


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "loanwise-api",
        "agents": [
            "CustomerProfilingAgentLLM",
            "AffordabilityAgent",
            "RiskMLAgent",
            "PolicyRAGAgent",
            "OfferDiscountAgentLLM",
            "LoanSimulatorAgent",
            "ComplianceExplanationAgentLLM",
        ],
        "auth": "none",
        "max_foir": 0.60,
        "base_rate": 11.5,
    }


@app.get("/api/health")
def api_health() -> dict[str, Any]:
    return health()


@app.post("/api/recommendations")
def create_recommendation(body: IntakeRequest) -> dict[str, Any]:
    try:
        return pipeline.evaluate(body.model_dump(), include_explanation=body.include_explanation)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Unable to generate recommendation. The recommendation service encountered an error.",
        ) from exc


@app.get("/api/recommendations/{application_id}")
def get_recommendation(application_id: str) -> dict[str, Any]:
    record = pipeline.get_application(application_id)
    if not record:
        raise HTTPException(status_code=404, detail="Application not found. Evaluate a customer first.")
    return record


@app.get("/api/applications")
def list_applications() -> dict[str, Any]:
    return {"items": pipeline.list_applications()}


@app.get("/api/customers")
def list_customers() -> dict[str, Any]:
    return {"items": pipeline.list_applications()}


@app.get("/api/customers/{application_id}")
def get_customer(application_id: str) -> dict[str, Any]:
    record = pipeline.get_application(application_id)
    if not record:
        raise HTTPException(status_code=404, detail="Customer application not found.")
    return record


@app.post("/api/simulator")
def simulate(body: SimulateRequest) -> dict[str, Any]:
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
def foreclosure(body: SimulateRequest) -> dict[str, Any]:
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


@app.get("/api/explainability/{application_id}")
def explainability(application_id: str) -> dict[str, Any]:
    record = pipeline.get_application(application_id)
    if not record:
        raise HTTPException(status_code=404, detail="Application not found.")
    return {
        "application_id": application_id,
        "policy_fit": record["policy_fit"],
        "risk": record["risk"],
        "affordability": {
            "decision_type": record["affordability"].get("decision_type"),
            "reason": record["affordability"].get("reason"),
            "eligibility_message": record["affordability"].get("eligibility_message"),
            "foir": record["affordability"].get("foir"),
            "requested_foir": record["affordability"].get("requested_foir"),
        },
        "explanation": record.get("explanation"),
        "concessions": record.get("concessions"),
    }


@app.get("/api/advisor/{application_id}")
def advisor(application_id: str) -> dict[str, Any]:
    record = pipeline.get_application(application_id)
    if not record:
        raise HTTPException(status_code=404, detail="Application not found.")
    return {
        "application_id": application_id,
        "explanation": record.get("explanation"),
        "explanation_error": record.get("explanation_error"),
        "disclaimer": "AI-generated guidance. It does not make the final lending decision.",
    }


@app.get("/api/audit")
def audit() -> dict[str, Any]:
    return {
        "supported": False,
        "items": [],
        "message": "The existing backend does not persist audit events.",
    }


@app.get("/api/approvals")
def approvals() -> dict[str, Any]:
    review = [
        item
        for item in pipeline.list_applications()
        if item["decision_type"] in ("COUNTER_OFFER", "REJECTED") or item["risk_band"] in ("HIGH", "MEDIUM")
    ]
    return {
        "supported": False,
        "note": "No approval mutation API exists. This list is derived from in-memory evaluations only.",
        "items": review,
    }


@app.get("/api/documents")
def documents() -> dict[str, Any]:
    return {
        "supported": False,
        "items": [],
        "message": "Document storage is not implemented in the existing backend.",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)

