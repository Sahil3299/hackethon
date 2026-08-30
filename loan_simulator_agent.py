from typing import Dict, Any


# =============================================================================
# LOAN SIMULATOR AGENT
# =============================================================================

class LoanSimulatorAgent:

    """
    EMI and early-closure simulator.

    IMPORTANT:

    Even if the loan is rejected, generate_schedule()
    returns a complete dictionary structure.

    Therefore:

        sim["early_closure"]["outstanding_principal"]

    will always exist.
    """

    # =========================================================================
    # EMI
    # =========================================================================

    def calculate_emi(
        self,
        principal: float,
        annual_rate: float,
        tenure_months: int
    ) -> float:

        if principal <= 0 or tenure_months <= 0:
            return 0.0

        monthly_rate = (
            annual_rate / 12 / 100
        )

        if monthly_rate == 0:
            return principal / tenure_months

        return (
            principal
            * monthly_rate
            * (1 + monthly_rate) ** tenure_months
            /
            (
                (1 + monthly_rate) ** tenure_months - 1
            )
        )

    # =========================================================================
    # GENERATE SCHEDULE
    # =========================================================================

    def generate_schedule(
        self,
        principal: float,
        rate: float,
        tenure_months: int,
        early_closure_month: int = 18
    ) -> Dict[str, Any]:

        # =====================================================================
        # REJECTED / INVALID LOAN
        # =====================================================================

        if principal <= 0 or tenure_months <= 0:

            return {

                "is_simulatable":
                    False,

                "emi":
                    0.0,

                "total_interest":
                    0.0,

                "total_cost":
                    0.0,

                "early_closure": {

                    "closure_month":
                        0,

                    "outstanding_principal":
                        0.0,

                    "interest_paid_until_closure":
                        0.0,

                    "interest_saved":
                        0.0,

                    "total_repaid":
                        0.0
                },

                "message":
                    "Loan simulation unavailable because "
                    "the requested loan is not eligible."
            }

        # =====================================================================
        # VALID LOAN
        # =====================================================================

        closure_month = min(
            max(1, early_closure_month),
            tenure_months
        )

        monthly_rate = (
            rate / 12 / 100
        )

        emi = self.calculate_emi(
            principal,
            rate,
            tenure_months
        )

        balance = float(principal)

        total_interest = 0.0

        balance_at_closure = float(principal)

        interest_paid_until_closure = 0.0

        payments_until_closure = 0.0

        # =====================================================================
        # MONTHLY CALCULATION
        # =====================================================================

        for month in range(
            1,
            tenure_months + 1
        ):

            interest = (
                balance
                * monthly_rate
            )

            principal_component = (
                emi
                - interest
            )

            principal_component = min(
                principal_component,
                balance
            )

            balance -= principal_component

            balance = max(
                0.0,
                balance
            )

            total_interest += interest

            if month == closure_month:

                balance_at_closure = balance

                interest_paid_until_closure = (
                    total_interest
                )

                payments_until_closure = (
                    emi * closure_month
                )

        # =====================================================================
        # INTEREST SAVING
        # =====================================================================

        remaining_interest = max(
            0.0,
            total_interest
            - interest_paid_until_closure
        )

        interest_saved = remaining_interest

        early_closure_total = (
            payments_until_closure
            + balance_at_closure
        )

        # =====================================================================
        # RETURN
        # =====================================================================

        return {

            "is_simulatable":
                True,

            "emi":
                round(
                    emi,
                    2
                ),

            "total_interest":
                round(
                    total_interest,
                    2
                ),

            "total_cost":
                round(
                    principal + total_interest,
                    2
                ),

            "early_closure": {

                "closure_month":
                    closure_month,

                "outstanding_principal":
                    round(
                        balance_at_closure,
                        2
                    ),

                "interest_paid_until_closure":
                    round(
                        interest_paid_until_closure,
                        2
                    ),

                "interest_saved":
                    round(
                        interest_saved,
                        2
                    ),

                "total_repaid":
                    round(
                        early_closure_total,
                        2
                    )
            },

            "message":
                "Early closure simulation calculated successfully."
        }

    # =========================================================================
    # PUBLIC METHOD
    # =========================================================================

    def generate_schedule_and_closure(
        self,
        principal: float,
        rate: float,
        tenure_months: int,
        early_closure_month: int = 18
    ) -> Dict[str, Any]:

        return self.generate_schedule(
            principal,
            rate,
            tenure_months,
            early_closure_month
        )