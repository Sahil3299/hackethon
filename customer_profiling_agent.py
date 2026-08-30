import os
from pathlib import Path
from typing import Dict, Any

from pydantic import BaseModel, Field
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

api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

if api_key and ChatGoogleGenerativeAI is not None:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.1,
        google_api_key=api_key,
    )
else:
    llm = None


# =============================================================================
# PROFILE SCHEMA
# =============================================================================

class ExtractedProfile(BaseModel):

    customer_id: str = Field(
        default="CUST-AUTO"
    )

    monthly_income: float = Field(
        ...,
        description="Net monthly in-hand salary"
    )

    age: int = Field(
        ...,
        description="Borrower age"
    )

    gender: str = Field(
        ...,
        description="Female, Male or Other"
    )

    house_rent: float = Field(
        default=0.0
    )

    existing_emi: float = Field(
        default=0.0
    )

    cibil_score: int = Field(
        ...,
        description="CIBIL score between 300 and 850"
    )

    requested_loan: float = Field(
        ...,
        description="Requested loan amount"
    )

    requested_tenure: int = Field(
        ...,
        description="Requested tenure in months"
    )


# =============================================================================
# CUSTOMER PROFILING AGENT
# =============================================================================

class CustomerProfilingAgentLLM:

    def __init__(self, model=llm):

        self.structured_llm = None
        self.prompt = None
        self.chain = None

        if model is None or ChatPromptTemplate is None:
            return

        self.structured_llm = model.with_structured_output(
            ExtractedProfile
        )

        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
You are an expert banking customer-profile extraction agent.

Extract only information present in the customer statement.

Rules:
- Never invent financial values.
- Missing rent = 0.
- Missing existing EMI = 0.
- Gender must be Female, Male or Other.
- CIBIL must be between 300 and 850.
- Loan amount must be positive.
- Tenure must be in months.
"""
            ),
            (
                "human",
                """
Extract the borrower profile from:

{user_input}
"""
            )
        ])

        self.chain = self.prompt | self.structured_llm

    def parse(
        self,
        raw_input: str | Dict[str, Any]
    ) -> ExtractedProfile:

        if isinstance(raw_input, dict):
            return ExtractedProfile(**raw_input)

        if self.chain is None:
            raise RuntimeError(
                "Gemini LLM is not configured. Set GEMINI_API_KEY or GOOGLE_API_KEY in the Render environment."
            )

        return self.chain.invoke({
            "user_input": raw_input
        })

    def process(
        self,
        raw_input: str | Dict[str, Any]
    ) -> ExtractedProfile:

        return self.parse(raw_input)