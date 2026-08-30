# =============================================================================
# pipeline.py
# AI Personalized Loan Recommendation System - Pipeline
# =============================================================================

import os

from dataset_generator import generate_loan_dataset

from customer_profiling_agent import CustomerProfilingAgentLLM
from affordability_agent import AffordabilityAgent
from risk_ml_agent import RiskMLAgent
from policy_rag_agent import PolicyRAGAgent
from offer_discount_agent import OfferDiscountAgentLLM
from loan_simulator_agent import LoanSimulatorAgent
from compliance_explanation_agent import ComplianceExplanationAgentLLM


# =============================================================================
# RUN COMPLETE LOAN PIPELINE
# =============================================================================

def run_pipeline():

    # -------------------------------------------------------------------------
    # 1. Generate training dataset if it does not exist
    # -------------------------------------------------------------------------

    if not os.path.exists("loan_risk_dataset.csv"):
        print("Generating loan risk dataset...")
        generate_loan_dataset()
        print("✓ Dataset generated")

    # -------------------------------------------------------------------------
    # 2. Initialize Agents
    # -------------------------------------------------------------------------

    print("\nInitializing AI agents...")

    profiler = CustomerProfilingAgentLLM()

    affordability_checker = AffordabilityAgent(
        max_foir=0.60
    )

    risk_agent = RiskMLAgent(
        "loan_risk_dataset.csv"
    )

    rag_agent = PolicyRAGAgent()

    offer_agent = OfferDiscountAgentLLM(
        rag_agent,
        base_rate=11.5
    )

    simulator = LoanSimulatorAgent()

    compliance_reporter = ComplianceExplanationAgentLLM()

    print("✓ All agents initialized")

    # -------------------------------------------------------------------------
    # 3. Customer 1
    # -------------------------------------------------------------------------
    # Female + high CIBIL + FOIR within limit
    # Expected: Requested loan should be eligible
    # -------------------------------------------------------------------------

    customer_1_input = {

        "customer_id":
            "APPL-8821",

        "monthly_income":
            95000,

        "age":
            27,

        "gender":
            "Female",

        "house_rent":
            18000,

        "existing_emi":
            8000,

        "cibil_score":
            795,

        "requested_loan":
            500000,

        "requested_tenure":
            36
    }

    # -------------------------------------------------------------------------
    # 4. Customer 2
    # -------------------------------------------------------------------------
    # Male + lower CIBIL + requested loan creates FOIR > 60%
    #
    # The system should NOT crash.
    #
    # If an affordable counter-offer exists:
    #     COUNTER_OFFER
    #
    # If no affordable loan exists:
    #     REJECTED
    # -------------------------------------------------------------------------

    customer_2_input = {

        "customer_id":
            "APPL-3104",

        "monthly_income":
            45000,

        "age":
            34,

        "gender":
            "Male",

        "house_rent":
            16000,

        "existing_emi":
            9000,

        "cibil_score":
            670,

        "requested_loan":
            600000,

        "requested_tenure":
            36
    }

    customers = [
        customer_1_input,
        customer_2_input
    ]

    # -------------------------------------------------------------------------
    # 5. Process Customers
    # -------------------------------------------------------------------------

    for user_input in customers:

        print("\n")
        print("=" * 80)
        print(
            f"Processing Customer: "
            f"{user_input['customer_id']}"
        )
        print("=" * 80)

        # ---------------------------------------------------------------------
        # Agent 1: Customer Profiling
        # ---------------------------------------------------------------------

        profile = profiler.process(
            user_input
        )

        # ---------------------------------------------------------------------
        # Agent 2: Affordability / FOIR
        # ---------------------------------------------------------------------

        aff_res = affordability_checker.assess(
            profile
        )

        # ---------------------------------------------------------------------
        # IMPORTANT
        #
        # If requested loan is not eligible, the affordability agent already
        # returns:
        #
        # adjusted_loan = 0
        #
        # OR a lower counter-offer amount.
        #
        # Therefore we always use adjusted_loan here.
        # ---------------------------------------------------------------------

        adjusted_loan = aff_res.get(
            "adjusted_loan",
            0.0
        )

        adjusted_tenure = aff_res.get(
            "adjusted_tenure",
            0
        )

        calculated_foir = aff_res.get(
            "foir",
            0.0
        )

        # ---------------------------------------------------------------------
        # Agent 3: Risk ML
        # ---------------------------------------------------------------------

        risk_res = risk_agent.evaluate_risk(

            profile,

            loan_amt=adjusted_loan,

            tenure=adjusted_tenure,

            foir=calculated_foir
        )

        # ---------------------------------------------------------------------
        # Agent 4 + 5:
        # Policy RAG + Offer/Discount
        # ---------------------------------------------------------------------

        final_rate, discounts = (
            offer_agent.compute_discounted_rate(
                profile
            )
        )

        # ---------------------------------------------------------------------
        # Agent 6: Loan Simulator
        # ---------------------------------------------------------------------
        #
        # If loan is rejected:
        #
        # adjusted_loan = 0
        #
        # The simulator returns a SAFE structure containing:
        #
        # early_closure
        # outstanding_principal
        # interest_saved
        #
        # Therefore no KeyError occurs.
        # ---------------------------------------------------------------------

        sim_res = (
            simulator.generate_schedule_and_closure(

                principal=adjusted_loan,

                rate=final_rate,

                tenure_months=adjusted_tenure,

                early_closure_month=18
            )
        )

        # ---------------------------------------------------------------------
        # Agent 7: Explanation / Compliance
        # ---------------------------------------------------------------------

        report = (
            compliance_reporter.format_final_decision(

                profile,

                aff_res,

                risk_res,

                final_rate,

                discounts,

                sim_res
            )
        )

        # ---------------------------------------------------------------------
        # Display Result
        # ---------------------------------------------------------------------

        print("\n")
        print("CUSTOMER RESULT")
        print("-" * 80)

        print(
            f"Customer ID: "
            f"{profile.customer_id}"
        )

        print(
            f"Requested Loan: "
            f"₹{profile.requested_loan:,.0f}"
        )

        print(
            f"Requested FOIR: "
            f"{aff_res.get('requested_foir', 0) * 100:.2f}%"
        )

        print(
            f"Final FOIR: "
            f"{aff_res.get('foir', 0) * 100:.2f}%"
        )

        print(
            f"Decision: "
            f"{aff_res.get('decision_type', 'UNKNOWN')}"
        )

        print(
            f"Eligible: "
            f"{aff_res.get('is_eligible', False)}"
        )

        print(
            f"Recommended Loan: "
            f"₹{adjusted_loan:,.0f}"
        )

        print(
            f"Recommended Tenure: "
            f"{adjusted_tenure} months"
        )

        print(
            f"Interest Rate: "
            f"{final_rate:.2f}%"
        )

        # ---------------------------------------------------------------------
        # Eligibility Message
        # ---------------------------------------------------------------------

        print("\nELIGIBILITY MESSAGE")
        print("-" * 80)

        print(
            aff_res.get(
                "eligibility_message",
                aff_res.get(
                    "reason",
                    "No eligibility information available."
                )
            )
        )

        # ---------------------------------------------------------------------
        # Risk Result
        # ---------------------------------------------------------------------

        print("\nRISK ASSESSMENT")
        print("-" * 80)

        print(
            f"Risk Band: "
            f"{risk_res.get('risk_band', 'NOT_EVALUATED')}"
        )

        print(
            f"Default Probability: "
            f"{risk_res.get('default_probability', 0) * 100:.2f}%"
        )

        # ---------------------------------------------------------------------
        # Simulation Result
        # ---------------------------------------------------------------------

        print("\nLOAN SIMULATION")
        print("-" * 80)

        print(
            f"EMI: "
            f"₹{sim_res.get('emi', 0):,.2f}"
        )

        early_closure = sim_res.get(
            "early_closure",
            {}
        )

        print(
            f"Outstanding Principal: "
            f"₹{early_closure.get('outstanding_principal', 0):,.2f}"
        )

        print(
            f"Interest Saved: "
            f"₹{early_closure.get('interest_saved', 0):,.2f}"
        )

        # ---------------------------------------------------------------------
        # Gemini Explanation
        # ---------------------------------------------------------------------

        print("\nAI EXPLANATION")
        print("-" * 80)

        print(
            report.executive_summary
        )

        print(
            "\nAffordability:"
        )

        print(
            report.affordability_rationale
        )

        print(
            "\nRisk & Pricing:"
        )

        print(
            report.risk_and_pricing_rationale
        )

        print(
            "\nPrepayment Strategy:"
        )

        print(
            report.smart_prepayment_strategy
        )

        print("\n")
        print("=" * 80)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    run_pipeline()