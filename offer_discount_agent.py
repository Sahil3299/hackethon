from typing import Any, List

from pydantic import BaseModel, Field


# =============================================================================
# CONCESSION SCHEMAS
# =============================================================================

class AppliedConcession(BaseModel):

    policy_name: str

    discount_percentage: float

    justification: str


class PolicyEvaluationResult(BaseModel):

    concessions: List[AppliedConcession] = Field(
        default_factory=list
    )

    total_discount: float = 0.0


# =============================================================================
# OFFER / DISCOUNT AGENT
# =============================================================================

class OfferDiscountAgentLLM:

    def __init__(
        self,
              rag_agent: Any,
        base_rate: float = 11.5
    ):

        self.rag = rag_agent

        self.base_rate = base_rate

    # =========================================================================
    # EVALUATE OFFERS
    # =========================================================================

    def evaluate_offers(
        self,
        profile
    ) -> tuple[float, PolicyEvaluationResult]:

        # ---------------------------------------------------------------------
        # RAG POLICY RETRIEVAL
        # ---------------------------------------------------------------------

        query = (
            f"loan discount policies for "
            f"{profile.gender} borrower "
            f"age {profile.age} "
            f"CIBIL {profile.cibil_score}"
        )

        self.rag.retrieve(
            query,
            n_results=5
        )

        concessions = []

        # ---------------------------------------------------------------------
        # WOMEN BORROWER
        # ---------------------------------------------------------------------

        if profile.gender.strip().lower() == "female":

            concessions.append(
                AppliedConcession(
                    policy_name="Women Borrowers Scheme",

                    discount_percentage=0.10,

                    justification=(
                        "Customer is a female primary applicant."
                    )
                )
            )

        # ---------------------------------------------------------------------
        # YOUTH
        # ---------------------------------------------------------------------

        if (
            21 <= profile.age <= 29
            and profile.monthly_income > 40000
        ):

            concessions.append(
                AppliedConcession(
                    policy_name=
                        "Youth Career Starter Program",

                    discount_percentage=0.20,

                    justification=(
                        "Customer is between 21 and 29 years old "
                        "and earns above ₹40,000 per month."
                    )
                )
            )

        # ---------------------------------------------------------------------
        # SUPER PRIME CIBIL
        # ---------------------------------------------------------------------

        if profile.cibil_score >= 780:

            concessions.append(
                AppliedConcession(
                    policy_name=
                        "Super Prime Credit Tier",

                    discount_percentage=0.35,

                    justification=(
                        "CIBIL score is at least 780."
                    )
                )
            )

        # ---------------------------------------------------------------------
        # TOTAL DISCOUNT
        # ---------------------------------------------------------------------

        total_discount = sum(
            c.discount_percentage
            for c in concessions
        )

        # Minimum final rate = 7.5%
        total_discount = min(
            total_discount,
            self.base_rate - 7.5
        )

        final_rate = (
            self.base_rate
            - total_discount
        )

        result = PolicyEvaluationResult(
            concessions=concessions,

            total_discount=round(
                total_discount,
                2
            )
        )

        return (
            round(final_rate, 2),
            result
        )

    # =========================================================================
    # PUBLIC METHOD
    # =========================================================================

    def compute_discounted_rate(
        self,
        profile
    ) -> tuple[float, PolicyEvaluationResult]:

        return self.evaluate_offers(profile)