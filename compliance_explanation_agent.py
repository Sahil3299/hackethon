import os
from pathlib import Path

from pydantic import BaseModel

from dotenv import load_dotenv

try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:  # pragma: no cover - optional for light deploys
    ChatPromptTemplate = None
    ChatGoogleGenerativeAI = None


# =============================================================================
# ENVIRONMENT + GEMINI
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"

load_dotenv(ENV_FILE)

api_key = (
    os.getenv("GEMINI_API_KEY")
    or
    os.getenv("GOOGLE_API_KEY")
)

if api_key and ChatGoogleGenerativeAI is not None:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.1,
        google_api_key=api_key,
    )
else:
    llm = None


# =============================================================================
# EXPLANATION SCHEMA
# =============================================================================

class ComplianceExplanationResult(BaseModel):

    executive_summary: str

    affordability_rationale: str

    risk_and_pricing_rationale: str

    smart_prepayment_strategy: str


# =============================================================================
# COMPLIANCE EXPLANATION AGENT
# =============================================================================

class ComplianceExplanationAgentLLM:

    def __init__(self, model=llm):

        self.structured_llm = None
        self.prompt = None
        self.chain = None

        if model is None or ChatPromptTemplate is None:
            return

        self.structured_llm = (
            model.with_structured_output(
                ComplianceExplanationResult
            )
        )

        self.prompt = ChatPromptTemplate.from_messages([

            (
                "system",
                """
You are the Senior Loan Advisory AI.

Explain the calculated loan recommendation clearly.

IMPORTANT RULES:

1. Never change numerical values.

2. Never invent bank policies.

3. Never claim guaranteed loan approval.

4. Explain FOIR using supplied calculations.

5. If eligibility is FALSE, clearly state:

"The requested loan is not eligible based on the information provided."

6. If decision type is COUNTER_OFFER:
   - Clearly state that the requested amount is not eligible.
   - Explain that a smaller amount is being recommended.
   - Clearly state that this is a counter-offer,
     NOT approval of the requested amount.

7. If loan is REJECTED:
   - Do not discuss EMI savings as if the loan exists.
   - Explain why affordability failed.

8. Explain risk only when risk assessment exists.

9. Explain applicable concessions.

10. Explain early closure only when simulation exists.

11. Keep the explanation concise and customer-friendly.
"""
            ),

            (
                "human",
                """
CUSTOMER

ID:
{customer_id}

Income:
₹{income:,.2f}

Rent:
₹{rent:,.2f}

Existing EMI:
₹{existing_emi:,.2f}

CIBIL:
{cibil}


AFFORDABILITY

Decision:
{decision_type}

Action:
{action}

Requested Loan:
₹{req_loan:,.2f}

Recommended Loan:
₹{sanctioned_loan:,.2f}

Requested Tenure:
{req_tenure} months

Recommended Tenure:
{sanctioned_tenure} months

Requested FOIR:
{requested_foir_pct:.2f}%

Final FOIR:
{foir_pct:.2f}%

Maximum FOIR:
60%

Eligibility Message:
{eligibility_message}

Reason:
{reason}


RISK

Default Probability:
{prob:.2f}%

Risk Band:
{risk_band}

SHAP Factors:
{shap_factors}


PRICING

Base Rate:
11.50%

Final Rate:
{final_rate:.2f}%

Concessions:
{concessions}


EARLY CLOSURE

Monthly EMI:
₹{emi:,.2f}

Closure Month:
{closure_m}

Outstanding Principal:
₹{outstanding:,.2f}

Estimated Interest Saved:
₹{savings:,.2f}


Generate a concise customer-friendly explanation.
"""
            )
        ])

        self.chain = (
            self.prompt
            |
            self.structured_llm
        )

    # =========================================================================
    # GENERATE REPORT
    # =========================================================================

    def generate_report(
        self,
        profile,
        aff,
        risk,
        rate,
        concessions,
        sim
    ):

        # ---------------------------------------------------------------------
        # SHAP
        # ---------------------------------------------------------------------

        if risk.get("top_shap_factors"):

            shap_str = ", ".join(
                [
                    f"{key} (impact: {value:+.3f})"
                    for key, value
                    in risk["top_shap_factors"]
                ]
            )

        else:

            shap_str = (
                "Risk assessment not performed because "
                "the requested loan is not eligible."
            )

        # ---------------------------------------------------------------------
        # CONCESSIONS
        # ---------------------------------------------------------------------

        if concessions.concessions:

            concession_str = ", ".join(
                [
                    f"{c.policy_name} "
                    f"(-{c.discount_percentage:.2f}%)"
                    for c in concessions.concessions
                ]
            )

        else:

            concession_str = "None"

        # ---------------------------------------------------------------------
        # SAFE SIMULATION EXTRACTION
        # ---------------------------------------------------------------------

        early = sim.get(
            "early_closure",
            {}
        )

        closure_month = early.get(
            "closure_month",
            0
        )

        outstanding = early.get(
            "outstanding_principal",
            0.0
        )

        savings = early.get(
            "interest_saved",
            0.0
        )

        # ---------------------------------------------------------------------
        # GEMINI
        # ---------------------------------------------------------------------

        if self.chain is None:
            raise RuntimeError(
                "Gemini LLM is not configured. Set GEMINI_API_KEY or GOOGLE_API_KEY in the Render environment."
            )

        return self.chain.invoke({

            "customer_id":
                profile.customer_id,

            "income":
                profile.monthly_income,

            "rent":
                profile.house_rent,

            "existing_emi":
                profile.existing_emi,

            "cibil":
                profile.cibil_score,

            "decision_type":
                aff.get(
                    "decision_type",
                    "UNKNOWN"
                ),

            "action":
                aff.get(
                    "action_taken",
                    "UNKNOWN"
                ),

            "sanctioned_loan":
                aff.get(
                    "adjusted_loan",
                    0.0
                ),

            "req_loan":
                profile.requested_loan,

            "sanctioned_tenure":
                aff.get(
                    "adjusted_tenure",
                    0
                ),

            "req_tenure":
                profile.requested_tenure,

            "requested_foir_pct":
                aff.get(
                    "requested_foir",
                    0.0
                ) * 100,

            "foir_pct":
                aff.get(
                    "foir",
                    0.0
                ) * 100,

            "eligibility_message":
                aff.get(
                    "eligibility_message",
                    ""
                ),

            "reason":
                aff.get(
                    "reason",
                    ""
                ),

            "prob":
                risk.get(
                    "default_probability",
                    0.0
                ) * 100,

            "risk_band":
                risk.get(
                    "risk_band",
                    "NOT_EVALUATED"
                ),

            "shap_factors":
                shap_str,

            "final_rate":
                rate,

            "concessions":
                concession_str,

            "emi":
                sim.get(
                    "emi",
                    0.0
                ),

            "closure_m":
                closure_month,

            "outstanding":
                outstanding,

            "savings":
                savings
        })

    # =========================================================================
    # PUBLIC METHOD
    # =========================================================================

    def format_final_decision(
        self,
        profile,
        aff,
        risk,
        rate,
        concessions,
        sim
    ):

        return self.generate_report(
            profile,
            aff,
            risk,
            rate,
            concessions,
            sim
        )