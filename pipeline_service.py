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
    "gender_encoded": "Applicant profile attribute",
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
        )
        self.applications[application_id] = record
        return record

    def simulate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Re-run EMI, FOIR, risk, and early-closure using existing agents."""
        application_id = payload.get("application_id")
        base = self.applications.get(application_id) if application_id else None

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

    def list_applications(self) -> List[Dict[str, Any]]:
        rows = []
        for record in self.applications.values():
            rows.append({
                "application_id": record["application_id"],
                "display_name": record["officer_fields"].get("full_name") or record["application_id"],
                "requested_loan": record["profile"]["requested_loan"],
                "recommended_loan": record["affordability"]["adjusted_loan"],
                "decision_type": record["affordability"]["decision_type"],
                "is_eligible": record["affordability"]["is_eligible"],
                "foir_percent": record["foir_breakdown"]["committed_percent"],
                "policy_fit_score": record["policy_fit"]["score"],
                "risk_band": record["risk"]["risk_band"],
                "created_at": record["created_at"],
            })
        rows.sort(key=lambda item: item["created_at"], reverse=True)
        return rows

    def get_application(self, application_id: str) -> Optional[Dict[str, Any]]:
        return self.applications.get(application_id)

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
