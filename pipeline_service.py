"""
Orchestration layer for the existing loan agents.

Does not reimplement FOIR, EMI, risk, discounts, or simulation.
Those calculations remain in the original agent modules.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from affordability_agent import AffordabilityAgent
from compliance_explanation_agent import ComplianceExplanationAgentLLM
from customer_profiling_agent import CustomerProfilingAgentLLM, ExtractedProfile
from database import get_db
from loan_simulator_agent import LoanSimulatorAgent
from offer_discount_agent import OfferDiscountAgentLLM, PolicyEvaluationResult
from policy_rag_agent import PolicyRAGAgent
from risk_ml_agent import RiskMLAgent


FEATURE_LABELS = {
    "monthly_income": "Monthly income",
    "age": "Borrower age",
    "house_rent": "Housing cost",
    "existing_emi": "Existing EMIs",
    "cibil_score": "Credit score (CIBIL)",
    "loan_amount": "Loan amount",
    "tenure_months": "Loan tenure",
    "foir": "Obligation ratio (FOIR)",
}


class LoanPipelineService:
    def __init__(self) -> None:
        self.profiler = CustomerProfilingAgentLLM()
        self.affordability = AffordabilityAgent(max_foir=0.60, benchmark_rate=11.5)
        self.risk_engine = RiskMLAgent("loan_risk_dataset.csv")
        self.rag_agent = PolicyRAGAgent()
        self.offer_engine = OfferDiscountAgentLLM(self.rag_agent, base_rate=11.5)
        self.simulator = LoanSimulatorAgent()
        self.compliance = ComplianceExplanationAgentLLM()
        self.applications: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Persistence helpers. Fall back to the in-memory dict when Supabase
    # is not configured so the app still runs locally.
    # ------------------------------------------------------------------
    def _db(self):
        return get_db()

    def _own_applications(self, user_id: str) -> List[Dict[str, Any]]:
        """Load application summary rows visible to a user."""
        db = self._db()
        if db is None:
            return [
                r for r in self.applications.values()
                if r.get("user_id") == user_id or not r.get("user_id")
            ]
        try:
            resp = db.from_("applications").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
            return resp.data or []
        except Exception as exc:
            _ = exc
            return []

    def _all_applications(self) -> List[Dict[str, Any]]:
        """Load all application rows (loan-officer scope)."""
        db = self._db()
        if db is None:
            return list(self.applications.values())
        try:
            resp = db.from_("applications").select("*").order("created_at", desc=True).execute()
            return resp.data or []
        except Exception as exc:
            _ = exc
            return []

    def _get_application_record(self, application_id: str) -> Optional[Dict[str, Any]]:
        db = self._db()
        if db is None:
            return self.applications.get(application_id)
        try:
            resp = db.from_("applications").select("*").eq("application_id", application_id).limit(1).execute()
            rows = resp.data or []
            if not rows:
                return None
            row = rows[0]
            record = dict(row.get("record") or {})
            record["application_id"] = record.get("application_id") or application_id
            record["user_id"] = row.get("user_id")
            record["status"] = row.get("status")
            record["company_status"] = row.get("status")
            record["human_review_required"] = row.get("human_review_required", False)
            if row.get("decision_note"):
                record["decision_note"] = row.get("decision_note")
            return record
        except Exception as exc:
            _ = exc
            return self.applications.get(application_id)

    def _save_application(
        self,
        record: Dict[str, Any],
        user_id: Optional[str] = None,
        status: str = "PENDING_REVIEW",
        human_review_required: Optional[bool] = None,
    ) -> None:
        db = self._db()
        application_id = record["application_id"]
        record["user_id"] = user_id
        record["status"] = status
        self.applications[application_id] = dict(record)
        if db is None:
            return
        review = (human_review_required
                  if human_review_required is not None
                  else bool((record.get("trust") or {}).get("human_review_required")))
        payload = {
            "application_id": application_id,
            "user_id": user_id,
            "status": status,
            "human_review_required": review,
            "record": record,
        }
        try:
            existing = db.from_("applications").select("application_id").eq("application_id", application_id).limit(1).execute()
            if existing.data:
                db.from_("applications").update(payload).eq("application_id", application_id).execute()
            else:
                db.from_("applications").insert(payload).execute()
        except Exception as exc:
            _ = exc

    def create_profile(self, payload: Dict[str, Any]) -> ExtractedProfile:
        agent_input = {
            "customer_id": payload.get("customer_id") or f"APP-{uuid.uuid4().hex[:8].upper()}",
            "monthly_income": payload["monthly_income"],
            "age": payload["age"],
            "gender": payload["gender"],
            "house_rent": payload.get("house_rent", 0.0),
            "existing_emi": payload.get("existing_emi", 0.0),
            "cibil_score": payload["cibil_score"],
            "requested_loan": payload["requested_loan"],
            "requested_tenure": payload["requested_tenure"],
        }
        return self.profiler.process(agent_input)

    def evaluate(
        self,
        payload: Dict[str, Any],
        include_explanation: bool = True,
        user_id: Optional[str] = None,
        status: str = "PENDING_REVIEW",
    ) -> Dict[str, Any]:
        profile = self.create_profile(payload)
        aff = self.affordability.assess(profile)
        rate, discounts = self.offer_engine.compute_discounted_rate(profile)

        is_eligible = bool(aff.get("is_eligible", False))
        adjusted_loan = float(aff.get("adjusted_loan", 0.0))
        adjusted_tenure = int(aff.get("adjusted_tenure", 0))
        final_foir = float(aff.get("foir", 0.0))
        closure_month = int(payload.get("early_closure_month") or min(18, max(profile.requested_tenure, 1)))

        if not is_eligible:
            sim = self.simulator.generate_schedule_and_closure(0, rate, 0, closure_month)
        else:
            sim = self.simulator.generate_schedule_and_closure(
                adjusted_loan,
                rate,
                adjusted_tenure,
                min(closure_month, adjusted_tenure),
            )

        risk = self.risk_engine.evaluate_risk(profile, adjusted_loan, adjusted_tenure, final_foir)

        explanation = None
        explanation_error = None
        if include_explanation:
            try:
                report = self.compliance.format_final_decision(
                    profile, aff, risk, rate, discounts, sim
                )
                explanation = {
                    "executive_summary": report.executive_summary,
                    "affordability_rationale": report.affordability_rationale,
                    "risk_and_pricing_rationale": report.risk_and_pricing_rationale,
                    "smart_prepayment_strategy": report.smart_prepayment_strategy,
                }
            except Exception as exc:
                explanation_error = "Advisor explanation could not be generated. Numerical results below are unchanged."
                explanation = None
                _ = exc

        application_id = profile.customer_id
        record = self._assemble_record(
            application_id=application_id,
            officer_fields=payload,
            profile=profile,
            aff=aff,
            rate=rate,
            discounts=discounts,
            sim=sim,
            risk=risk,
            explanation=explanation,
            explanation_error=explanation_error,
            status=status,
        )
        self._save_application(record, user_id=user_id, status=status)
        return record

    def simulate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Re-run EMI, FOIR, risk, and early-closure using existing agents."""
        application_id = payload.get("application_id")
        base = self._get_application_record(application_id) if application_id else None

        if application_id and not base:
            raise KeyError(f"Application not found: {application_id}")

        if base:
            profile_data = base["profile"]
            profile = ExtractedProfile(
                customer_id=profile_data["customer_id"],
                monthly_income=profile_data["monthly_income"],
                age=profile_data["age"],
                gender=profile_data["gender"],
                house_rent=profile_data["house_rent"],
                existing_emi=profile_data["existing_emi"],
                cibil_score=profile_data["cibil_score"],
                requested_loan=float(payload.get("loan_amount", profile_data["requested_loan"])),
                requested_tenure=int(payload.get("tenure_months", profile_data["requested_tenure"])),
            )
        else:
            profile = self.create_profile({
                **payload,
                "requested_loan": payload["loan_amount"],
                "requested_tenure": payload["tenure_months"],
            })

        loan_amount = float(payload["loan_amount"])
        tenure_months = int(payload["tenure_months"])
        closure_month = int(payload.get("early_closure_month") or min(18, tenure_months))

        rate, discounts = self.offer_engine.compute_discounted_rate(profile)
        if "interest_rate" in payload and payload["interest_rate"] is not None:
            rate = float(payload["interest_rate"])

        sim = self.simulator.generate_schedule_and_closure(
            loan_amount,
            rate,
            tenure_months,
            min(closure_month, tenure_months),
        )

        emi = float(sim.get("emi", 0.0))
        existing = profile.existing_emi + profile.house_rent
        foir = (existing + emi) / profile.monthly_income if profile.monthly_income else 0.0
        max_foir = self.affordability.max_foir
        is_within = foir <= max_foir

        risk = self.risk_engine.evaluate_risk(profile, loan_amount, tenure_months, foir)

        foir_breakdown = self._foir_breakdown(
            income=profile.monthly_income,
            existing_emi=profile.existing_emi,
            house_rent=profile.house_rent,
            proposed_emi=emi,
            max_foir=max_foir,
        )

        return {
            "application_id": profile.customer_id,
            "loan_amount": loan_amount,
            "tenure_months": tenure_months,
            "interest_rate": rate,
            "concessions": self._serialize_discounts(discounts),
            "simulation": sim,
            "foir": foir,
            "foir_percent": round(foir * 100, 2),
            "max_foir": max_foir,
            "within_policy_limit": is_within,
            "foir_status": foir_breakdown["status"],
            "foir_breakdown": foir_breakdown,
            "risk": self._serialize_risk(risk),
            "policy_fit": self._policy_fit(
                decision_type="APPROVED_AS_REQUESTED" if is_within else "COUNTER_OFFER",
                is_eligible=is_within,
                foir=foir,
                requested_foir=foir,
                risk=risk,
            ),
        }

    def foreclosure(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        result = self.simulate(payload)
        early = result["simulation"].get("early_closure", {})
        policies = self.rag_agent.retrieve("foreclosure early closure penalty fee", n_results=2)
        return {
            **result,
            "foreclosure": {
                "closure_month": early.get("closure_month", 0),
                "outstanding_principal": early.get("outstanding_principal", 0.0),
                "interest_paid_until_closure": early.get("interest_paid_until_closure", 0.0),
                "interest_saved": early.get("interest_saved", 0.0),
                "total_repaid": early.get("total_repaid", 0.0),
                "foreclosure_charge": None,
                "foreclosure_charge_note": (
                    "This backend does not calculate a foreclosure penalty amount. "
                    "Interest saved and outstanding principal come from the loan simulator."
                ),
                "related_policies": policies,
            },
        }

    def list_applications(self, user_id: Optional[str] = None, officer: bool = False) -> List[Dict[str, Any]]:
        if officer:
            rows_source = self._all_applications()
        elif user_id:
            rows_source = self._own_applications(user_id)
        else:
            rows_source = self._all_applications()

        rows = []
        for item in rows_source:
            if isinstance(item, dict) and "record" in item and isinstance(item.get("record"), dict):
                record = item["record"]
                record.setdefault("application_id", item.get("application_id"))
                rows.append(self._summary_from(record, item))
            else:
                record = item
                rows.append(self._summary_from(record))
        rows.sort(key=lambda item: item.get("created_at", "") or "", reverse=True)
        return rows

    def _summary_from(self, record: Dict[str, Any], row: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {
            "application_id": record.get("application_id") or (row or {}).get("application_id"),
            "display_name": (record.get("officer_fields") or {}).get("full_name")
                            or (record.get("application_id") or ""),
            "requested_loan": (record.get("profile") or {}).get("requested_loan"),
            "recommended_loan": (record.get("affordability") or {}).get("adjusted_loan"),
            "decision_type": (record.get("affordability") or {}).get("decision_type"),
            "is_eligible": (record.get("affordability") or {}).get("is_eligible"),
            "foir_percent": ((record.get("foir_breakdown") or {}).get("committed_percent")),
            "policy_fit_score": (record.get("policy_fit") or {}).get("score"),
            "risk_band": (record.get("risk") or {}).get("risk_band"),
            "status": (row or {}).get("status") or record.get("status") or "PENDING_REVIEW",
            "human_review_required": (row or {}).get("human_review_required",
                                        (record.get("trust") or {}).get("human_review_required", False)),
            "created_at": (row or {}).get("created_at") or record.get("created_at"),
        }

    def get_application(self, application_id: str) -> Optional[Dict[str, Any]]:
        return self._get_application_record(application_id)

    def make_decision(
        self,
        application_id: str,
        decision: str,
        note: Optional[str],
        decided_by: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Record a loan-officer decision. Persists to Supabase when configured;
        otherwise falls back to the in-memory cache so the app still runs
        end to end in local dev.

        Returns a summary dict, or None if the application does not exist.
        """
        db = self._db()
        if db is not None:
            resp = db.from_("applications").select("*").eq("application_id", application_id).limit(1).execute()
            rows = resp.data or []
            if not rows:
                return None
            existing = rows[0]
            old_status = existing.get("status")
            db.from_("applications").update({
                "status": decision,
                "decision_note": note or "",
                "decision_by": decided_by,
                "decided_at": "now()",
            }).eq("application_id", application_id).execute()
            return {
                "application_id": application_id,
                "previous_status": old_status,
                "new_status": decision,
            }

        record = self.applications.get(application_id)
        if not record:
            return None
        old_status = record.get("status") or "PENDING_REVIEW"
        record["status"] = decision
        record["decision_note"] = note or ""
        record["decision_by"] = decided_by
        record["decided_at"] = datetime.now(timezone.utc).isoformat()
        self.applications[application_id] = dict(record)
        return {
            "application_id": application_id,
            "previous_status": old_status,
            "new_status": decision,
        }

    def _assemble_record(
        self,
        application_id: str,
        officer_fields: Dict[str, Any],
        profile: ExtractedProfile,
        aff: Dict[str, Any],
        rate: float,
        discounts: PolicyEvaluationResult,
        sim: Dict[str, Any],
        risk: Dict[str, Any],
        explanation: Optional[Dict[str, str]],
        explanation_error: Optional[str],
        status: str = "PENDING_REVIEW",
    ) -> Dict[str, Any]:
        emi = float(sim.get("emi", aff.get("estimated_emi", 0.0)) or 0.0)
        foir_breakdown = self._foir_breakdown(
            income=profile.monthly_income,
            existing_emi=profile.existing_emi,
            house_rent=profile.house_rent,
            proposed_emi=emi,
            max_foir=self.affordability.max_foir,
        )
        policy_fit = self._policy_fit(
            decision_type=str(aff.get("decision_type", "UNKNOWN")),
            is_eligible=bool(aff.get("is_eligible", False)),
            foir=float(aff.get("foir", 0.0)),
            requested_foir=float(aff.get("requested_foir", aff.get("foir", 0.0))),
            risk=risk,
        )

        requested_sim = self.simulator.generate_schedule_and_closure(
            profile.requested_loan,
            rate,
            profile.requested_tenure,
            min(int(officer_fields.get("early_closure_month") or 18), profile.requested_tenure),
        )

        alternatives = self._tenure_alternatives(profile, rate, discounts)

        return {
            "application_id": application_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "illustrative_data": True,
            "officer_fields": {
                "full_name": officer_fields.get("full_name"),
                "email": officer_fields.get("email"),
                "phone": officer_fields.get("phone"),
                "employer": officer_fields.get("employer"),
                "employment_type": officer_fields.get("employment_type"),
                "loan_purpose": officer_fields.get("loan_purpose"),
            },
            "profile": profile.model_dump(),
            "affordability": aff,
            "interest_rate": rate,
            "base_rate": self.offer_engine.base_rate,
            "concessions": self._serialize_discounts(discounts),
            "simulation": sim,
            "requested_simulation": requested_sim,
            "risk": self._serialize_risk(risk),
            "foir_breakdown": foir_breakdown,
            "policy_fit": policy_fit,
            "alternatives": alternatives,
            "explanation": explanation,
            "explanation_error": explanation_error,
            "financial_health": None,
            "thin_file": None,
            "trust": {
                "data_completeness_note": "Score uses fields required by the existing agents only.",
                "policy_checks": "FOIR assessed against 60% maximum used by AffordabilityAgent.",
                "human_review_required": policy_fit["label"] in ("Review", "Poor Match") or not aff.get("is_eligible"),
                "synthetic_training_data": True,
                "model": "XGBoost default-risk model trained on loan_risk_dataset.csv",
            },
        }

    def _tenure_alternatives(
        self,
        profile: ExtractedProfile,
        rate: float,
        discounts: PolicyEvaluationResult,
    ) -> List[Dict[str, Any]]:
        """Compare requested tenure options using the real simulator + FOIR formula."""
        tenures = []
        for months in (24, 36, 48, 60, 84):
            if months == profile.requested_tenure:
                continue
            sim = self.simulator.generate_schedule_and_closure(
                profile.requested_loan, rate, months, min(18, months)
            )
            emi = float(sim.get("emi", 0.0))
            foir = (
                (profile.existing_emi + profile.house_rent + emi) / profile.monthly_income
                if profile.monthly_income
                else 0.0
            )
            tenures.append({
                "label": f"{months}-month tenure",
                "loan_amount": profile.requested_loan,
                "tenure_months": months,
                "interest_rate": rate,
                "emi": emi,
                "total_interest": sim.get("total_interest", 0.0),
                "total_cost": sim.get("total_cost", 0.0),
                "foir": foir,
                "foir_percent": round(foir * 100, 2),
                "within_limit": foir <= self.affordability.max_foir,
            })
        tenures.sort(key=lambda row: row["foir"])
        return tenures[:4]

    def _foir_breakdown(
        self,
        income: float,
        existing_emi: float,
        house_rent: float,
        proposed_emi: float,
        max_foir: float,
    ) -> Dict[str, Any]:
        if income <= 0:
            committed = 0.0
        else:
            committed = (existing_emi + house_rent + proposed_emi) / income
        available = max(0.0, max_foir - committed)
        if committed > max_foir:
            status = "EXCEEDS_LIMIT"
        elif committed >= max_foir - 0.05:
            status = "NEAR_LIMIT"
        else:
            status = "WITHIN_LIMIT"
        return {
            "monthly_income": income,
            "existing_emi": existing_emi,
            "house_rent": house_rent,
            "other_obligations": 0.0,
            "proposed_emi": proposed_emi,
            "committed_ratio": committed,
            "committed_percent": round(committed * 100, 2),
            "policy_limit": max_foir,
            "policy_limit_percent": round(max_foir * 100, 1),
            "available_ratio": available,
            "available_percent": round(available * 100, 2),
            "status": status,
        }

    def _policy_fit(
        self,
        decision_type: str,
        is_eligible: bool,
        foir: float,
        requested_foir: float,
        risk: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Presentation index derived from existing agent outputs.
        This is NOT a separate ML suitability model.
        """
        max_foir = self.affordability.max_foir
        utilization = min(requested_foir / max_foir, 2.0) if max_foir else 1.0
        affordability_points = max(0, 55 - int((utilization - 1) * 40)) if not is_eligible else int(55 + (1 - min(foir / max_foir, 1)) * 25)
        risk_band = risk.get("risk_band", "NOT_EVALUATED")
        risk_points = {"LOW": 20, "MEDIUM": 10, "HIGH": 0, "NOT_EVALUATED": 5}.get(risk_band, 5)
        if decision_type == "REJECTED":
            score = max(8, min(38, affordability_points + risk_points - 20))
            label = "Poor Match"
        elif decision_type == "COUNTER_OFFER":
            score = max(45, min(74, affordability_points + risk_points))
            label = "Review"
        else:
            score = max(70, min(96, affordability_points + risk_points))
            if score >= 85:
                label = "Strong Match"
            else:
                label = "Good Match"
        return {
            "score": int(score),
            "label": label,
            "source": "derived_from_foir_eligibility_and_risk",
            "note": "Composite of AffordabilityAgent decision, FOIR utilization, and RiskMLAgent band. Not a standalone model score.",
        }

    def _serialize_discounts(self, discounts: PolicyEvaluationResult) -> Dict[str, Any]:
        return {
            "total_discount": discounts.total_discount,
            "concessions": [
                {
                    "policy_name": item.policy_name,
                    "discount_percentage": item.discount_percentage,
                    "justification": item.justification,
                }
                for item in discounts.concessions
            ],
        }

    def _serialize_risk(self, risk: Dict[str, Any]) -> Dict[str, Any]:
        factors = []
        for item in risk.get("top_shap_factors") or []:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                feature, impact = item[0], float(item[1])
            else:
                continue
            factors.append({
                "feature": feature,
                "label": FEATURE_LABELS.get(str(feature), str(feature).replace("_", " ")),
                "impact": round(impact, 4),
                "direction": "increases_risk" if impact > 0 else "decreases_risk",
            })
        return {
            "default_probability": risk.get("default_probability", 0.0),
            "risk_band": risk.get("risk_band", "NOT_EVALUATED"),
            "message": risk.get("message"),
            "factors": factors,
        }


pipeline = LoanPipelineService()
