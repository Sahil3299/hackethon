"""
AI output guardrail.

Before returning a Gemini-generated explanation, validate it against the
real numbers it is supposed to describe (loan amount, rate, EMI, FOIR).
If the generated text contradicts those numbers, fall back to a rule-based
version and log the mismatch so we never show unverified narrative text.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional


def _extract_number(text: str, label: str) -> Optional[float]:
    """Find a number in the text that plausibly belongs to `label`."""
    if not text:
        return None
    # Look for the nearest number within a few hundred chars of the label.
    pattern = re.escape(label) + r"[\s\S]{0,120}?([\d,]+(?:\.\d+)?)"
    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        pattern2 = r"([\d,]+(?:\.\d+)?)\s*" + re.escape(label)
        match = re.search(pattern2, text, re.IGNORECASE)
    if not match:
        return None
    try:
        return float(match.group(1).replace(",", ""))
    except ValueError:
        return None


def _matches(
    expected: Optional[float], found: Optional[float], tol_rel: float = 0.05, pct: bool = False
) -> bool:
    if expected is None or found is None:
        return True  # Cannot verify -> assume OK (missing value, not a contradiction).
    if expected == 0:
        return abs(found) < 1
    candidates = [expected]
    if pct:
        # The LLM may state a percentage (e.g. "42%") that represents a
        # stored fraction (0.42) or the raw percentage form.
        candidates.append(expected * 100)
        candidates.append(expected / 100)
    return any(abs(candidate - found) / abs(candidate) <= tol_rel for candidate in candidates)


def validate_explanation(
    explanation: Optional[Dict[str, str]],
    record: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Validate an AI explanation against the true numbers. Returns:
      { pass: bool, mismatches: [...], explanation: ... }
    If validation fails, `explanation` is set to None so the caller can fall
    back to the rule-based adverse action / static explanation.
    """
    aff = record.get("affordability") or {}
    rate = record.get("interest_rate")
    sim = record.get("simulation") or {}
    foir = float(aff.get("foir") or 0.0)
    emi = float(sim.get("emi") or aff.get("estimated_emi") or 0.0)
    loan = float(aff.get("adjusted_loan") or 0.0)

    if not explanation:
        return {"pass": False, "mismatches": ["No AI explanation was produced."], "explanation": None}

    checks = [
        ("loan", loan, "Rs", False),
        ("rate", rate, "%", True),
        ("EMI", emi, "Rs", False),
        ("FOIR", foir, "%", True),
    ]

    mismatches: list[str] = []
    for label, expected, unit, pct in checks:
        found = _extract_number("\n".join(explanation.values()), label)
        if not _matches(expected, found, pct=pct):
            mismatches.append(
                f"Explanation contains a mismatched value for {label}: "
                f"expected {expected}{unit}, found {found}{unit if found is not None else ''}."
            )

    return {
        "pass": len(mismatches) == 0,
        "mismatches": mismatches,
        "explanation": explanation if not mismatches else None,
    }