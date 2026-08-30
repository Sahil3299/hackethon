from typing import Dict, Any

from customer_profiling_agent import ExtractedProfile


# =============================================================================
# AFFORDABILITY / FOIR AGENT
# =============================================================================

class AffordabilityAgent:

    """
    Calculates loan affordability using FOIR.

    FOIR =
        (Existing EMI + House Rent + New Loan EMI)
        / Monthly Income

    Maximum allowed FOIR = 60%.

    Results:

    1. APPROVED_AS_REQUESTED
       Requested loan fits within 60%.

    2. COUNTER_OFFER
       Requested loan exceeds 60%, but a smaller affordable
       loan can be calculated.

    3. REJECTED
       No meaningful affordable loan is possible.
    """

    def __init__(
        self,
        max_foir: float = 0.60,
        benchmark_rate: float = 11.5,
        max_age: int = 60,
        max_tenure: int = 84
    ):

        self.max_foir = max_foir
        self.benchmark_rate = benchmark_rate
        self.max_age = max_age
        self.max_tenure = max_tenure

    # =========================================================================
    # EMI CALCULATION
    # =========================================================================

    def calculate_emi(
        self,
        principal: float,
        annual_rate: float,
        tenure_months: int
    ) -> float:

        if principal <= 0 or tenure_months <= 0:
            return 0.0

        monthly_rate = annual_rate / 12 / 100

        if monthly_rate == 0:
            return principal / tenure_months

        emi = (
            principal
            * monthly_rate
            * (1 + monthly_rate) ** tenure_months
            /
            (
                (1 + monthly_rate) ** tenure_months - 1
            )
        )

        return float(emi)

    # =========================================================================
    # MAXIMUM LOAN CALCULATION
    # =========================================================================

    def calculate_max_loan(
        self,
        allowed_emi: float,
        annual_rate: float,
        tenure_months: int
    ) -> float:

        if allowed_emi <= 0 or tenure_months <= 0:
            return 0.0

        monthly_rate = annual_rate / 12 / 100

        if monthly_rate == 0:
            return allowed_emi * tenure_months

        max_loan = (
            allowed_emi
            *
            (
                (1 + monthly_rate) ** tenure_months - 1
            )
            /
            (
                monthly_rate
                *
                (1 + monthly_rate) ** tenure_months
            )
        )

        return float(max_loan)

    # =========================================================================
    # AFFORDABILITY ASSESSMENT
    # =========================================================================

    def assess(
        self,
        profile: ExtractedProfile
    ) -> Dict[str, Any]:

        # ---------------------------------------------------------------------
        # INVALID INCOME
        # ---------------------------------------------------------------------

        if profile.monthly_income <= 0:

            return self._rejected_result(
                foir=999.0,
                reason="Monthly income must be greater than zero."
            )

        # ---------------------------------------------------------------------
        # REQUESTED LOAN EMI
        # ---------------------------------------------------------------------

        requested_emi = self.calculate_emi(
            profile.requested_loan,
            self.benchmark_rate,
            profile.requested_tenure
        )

        existing_obligations = (
            profile.existing_emi
            + profile.house_rent
        )

        total_obligation = (
            existing_obligations
            + requested_emi
        )

        requested_foir = (
            total_obligation
            / profile.monthly_income
        )

        # ---------------------------------------------------------------------
        # CASE 1: REQUESTED LOAN IS ELIGIBLE
        # ---------------------------------------------------------------------

        if requested_foir <= self.max_foir:

            return {
                "is_eligible": True,

                "decision_type":
                    "APPROVED_AS_REQUESTED",

                "foir":
                    requested_foir,

                "requested_foir":
                    requested_foir,

                "adjusted_loan":
                    profile.requested_loan,

                "adjusted_tenure":
                    profile.requested_tenure,

                "estimated_emi":
                    requested_emi,

                "allowed_emi":
                    requested_emi,

                "action_taken":
                    "APPROVED_AS_REQUESTED",

                "reason":
                    "Requested loan fits within the maximum 60% FOIR threshold.",

                "eligibility_message":
                    "The requested loan is eligible based on the information provided."
            }

        # ---------------------------------------------------------------------
        # CASE 2: REQUESTED LOAN EXCEEDS 60% FOIR
        # ---------------------------------------------------------------------

        allowed_emi = (
            profile.monthly_income * self.max_foir
            - existing_obligations
        )

        # ---------------------------------------------------------------------
        # EXISTING OBLIGATIONS ALREADY EXCEED 60%
        # ---------------------------------------------------------------------

        if allowed_emi <= 0:

            return self._rejected_result(
                foir=requested_foir,

                reason=(
                    "The requested loan is not eligible because "
                    "existing EMI and house rent already consume "
                    "the maximum permitted 60% of monthly income."
                ),

                requested_foir=requested_foir
            )

        # ---------------------------------------------------------------------
        # AGE-BASED MAXIMUM TENURE
        # ---------------------------------------------------------------------

        remaining_months_to_age_60 = (
            self.max_age - profile.age
        ) * 12

        if remaining_months_to_age_60 <= 0:

            max_possible_tenure = 1

        else:

            max_possible_tenure = min(
                self.max_tenure,
                remaining_months_to_age_60
            )

        # ---------------------------------------------------------------------
        # MAXIMUM LOAN AT REQUESTED TENURE
        # ---------------------------------------------------------------------

        max_loan_requested_tenure = self.calculate_max_loan(
            allowed_emi,
            self.benchmark_rate,
            profile.requested_tenure
        )

        # ---------------------------------------------------------------------
        # MAXIMUM LOAN AT EXTENDED TENURE
        # ---------------------------------------------------------------------

        extended_tenure = max(
            profile.requested_tenure,
            max_possible_tenure
        )

        extended_tenure = min(
            extended_tenure,
            self.max_tenure
        )

        max_loan_extended = self.calculate_max_loan(
            allowed_emi,
            self.benchmark_rate,
            extended_tenure
        )

        # ---------------------------------------------------------------------
        # CHOOSE BEST AFFORDABLE LOAN
        # ---------------------------------------------------------------------

        if max_loan_extended > max_loan_requested_tenure:

            recommended_loan = max_loan_extended
            recommended_tenure = extended_tenure

        else:

            recommended_loan = max_loan_requested_tenure
            recommended_tenure = profile.requested_tenure

        # Round down to nearest ₹1,000
        recommended_loan = (
            int(recommended_loan // 1000) * 1000
        )

        # ---------------------------------------------------------------------
        # NO AFFORDABLE LOAN
        # ---------------------------------------------------------------------

        if recommended_loan < 1000:

            return self._rejected_result(
                foir=requested_foir,

                reason=(
                    f"The requested loan produces "
                    f"{requested_foir * 100:.2f}% FOIR, "
                    f"which exceeds the maximum allowed "
                    f"{self.max_foir * 100:.0f}%. "
                    "No meaningful affordable loan amount "
                    "could be calculated."
                ),

                requested_foir=requested_foir
            )

        # ---------------------------------------------------------------------
        # COUNTER OFFER
        # ---------------------------------------------------------------------

        recommended_emi = self.calculate_emi(
            recommended_loan,
            self.benchmark_rate,
            recommended_tenure
        )

        final_foir = (
            existing_obligations
            + recommended_emi
        ) / profile.monthly_income

        reduction = (
            profile.requested_loan
            - recommended_loan
        )

        return {

            "is_eligible": True,

            "decision_type":
                "COUNTER_OFFER",

            "foir":
                final_foir,

            "requested_foir":
                requested_foir,

            "adjusted_loan":
                recommended_loan,

            "adjusted_tenure":
                recommended_tenure,

            "estimated_emi":
                recommended_emi,

            "allowed_emi":
                allowed_emi,

            "action_taken":
                "COUNTER_OFFER_PRINCIPAL_REDUCED",

            "reason":
                (
                    f"Requested loan produces "
                    f"{requested_foir * 100:.2f}% FOIR, "
                    f"which exceeds the maximum allowed "
                    f"{self.max_foir * 100:.0f}%. "
                    f"The maximum affordable loan is "
                    f"approximately ₹{recommended_loan:,.0f}."
                ),

            "eligibility_message":
                (
                    "The requested loan is not eligible based "
                    "on the information provided. "
                    f"A lower loan amount of approximately "
                    f"₹{recommended_loan:,.0f} can fit within "
                    "the 60% FOIR limit."
                ),

            "loan_reduction":
                reduction
        }

    # =========================================================================
    # REJECTION RESULT
    # =========================================================================

    def _rejected_result(
        self,
        foir: float,
        reason: str,
        requested_foir: float | None = None
    ) -> Dict[str, Any]:

        return {

            "is_eligible": False,

            "decision_type":
                "REJECTED",

            "foir":
                foir,

            "requested_foir":
                requested_foir
                if requested_foir is not None
                else foir,

            "adjusted_loan":
                0.0,

            "adjusted_tenure":
                0,

            "estimated_emi":
                0.0,

            "allowed_emi":
                0.0,

            "action_taken":
                "REJECTED_FOIR_EXCEEDED",

            "reason":
                reason,

            "eligibility_message":
                (
                    "The requested loan is not eligible based on "
                    "the information provided because the "
                    "affordability limit has been exceeded."
                ),

            "loan_reduction":
                0.0
        }