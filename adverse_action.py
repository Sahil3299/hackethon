"""
Adverse action notices — rule-based, defensible statements.

These are the OFFICIAL reasons for a rejection or counter-offer, sourced
directly from the affordability/risk output, NOT from the Gemini narrative.
The Gemini explanation is only ever presented alongside these and is
validated against the real numbers before display (see validate_explanation).
"""

from __future__ import annotations

from typing import Any, Dict, List


def build_adverse_action(record: Dict[str, Any]) -> Dict[str, Any]:
    aff = record.get("affordability") or {}
    risk = record.get("risk") or {}
    decision_type = str(aff.get("decision_type", "UNKNOWN")).upper()
    foir = float(aff.get("foir") or 0.0)
    requested_foir = float(aff.get("requested_foir") or 0.0)
    max_foir = 0.60
    profile = record.get("profile") or {}

    reasons: List[str] = []

    # FOIR-based affordability reason.
    if requested_foir > max_foir:
        reasons.append(
            f"The total monthly debt obligation of {requested_foir * 100:.2f}% of your "
            f"monthly income exceeds the maximum permitted limit of {max_foir * 100:.0f}%."
        )
    if foir > max_foir:
        reasons.append(
            f"Adding the proposed loan would bring your obligation ratio to "
            f"{foir * 100:.2f}%, above the {max_foir * 100:.0f}% ceiling."
        )

    # Risk-band reason (only if the model actually evaluated risk).
    risk_band = str(risk.get("risk_band", "NOT_EVALUATED")).upper()
    if risk_band in ("HIGH", "MEDIUM"):
        prob = risk.get("default_probability")
        reason = f"The assessed credit risk band is {risk_band.title()}."
        if isinstance(prob, (int, float)):
            reason += f" Estimated default probability is {prob * 100:.2f}%."
        reasons.append(reason)

    # Explicit statement for each decision class.
    if decision_type == "REJECTED":
        headline = (
            "Adverse Action Notice: Your loan application was not approved "
            "based on the following reason(s):"
        )
    elif decision_type == "COUNTER_OFFER":
        headline = (
            "Counter-Offer Notice: The requested loan amount could not be approved. "
            "We may approve a lower amount. Reason(s):"
        )
    else:
        headline = "Application review summary:"
        reasons.append("No affordability or risk-based adverse criteria were triggered.")

    if not reasons:
        reasons.append("No specific adverse reason was triggered by the affordability or risk assessment.")

    official_reason = " ".join(reasons)

    return {
        "notice_type": "ADVERSE_ACTION" if decision_type == "REJECTED" else "COUNTER_OFFER",
        "decision_type": decision_type,
        "headline": headline,
        "reasons": reasons,
        "official_reason": official_reason,
        "foir": round(foir, 4),
        "requested_foir": round(requested_foir, 4),
        "max_foir": max_foir,
        "risk_band": risk_band,
        "disclosure": (
            "This notice is generated from rule-based affordability and credit-risk "
            "outputs. You may request a manual review by contacting our loan cell."
        ),
    }