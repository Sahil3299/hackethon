import streamlit as st
import pandas as pd
import os

from customer_profiling_agent import CustomerProfilingAgentLLM
from affordability_agent import AffordabilityAgent
from risk_ml_agent import RiskMLAgent
from policy_rag_agent import PolicyRAGAgent
from offer_discount_agent import OfferDiscountAgentLLM
from loan_simulator_agent import LoanSimulatorAgent
from compliance_explanation_agent import ComplianceExplanationAgentLLM

from dataset_generator import generate_loan_dataset


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="CrediAdapt AI",
    page_icon="🛡️",
    layout="wide"
)


# =============================================================================
# DATASET
# =============================================================================

if not os.path.exists("loan_risk_dataset.csv"):
    generate_loan_dataset()


# =============================================================================
# LOAD AGENTS
# =============================================================================

@st.cache_resource
def load_agents():

    profiler = CustomerProfilingAgentLLM()

    affordability = AffordabilityAgent(
        max_foir=0.60,
        benchmark_rate=11.5
    )

    risk_engine = RiskMLAgent(
        "loan_risk_dataset.csv"
    )

    rag_agent = PolicyRAGAgent()

    simulator = LoanSimulatorAgent()

    compliance = ComplianceExplanationAgentLLM()

    return (
        profiler,
        affordability,
        risk_engine,
        rag_agent,
        simulator,
        compliance
    )


(
    profiler,
    affordability,
    risk_engine,
    rag_agent,
    simulator,
    compliance
) = load_agents()


offer_engine = OfferDiscountAgentLLM(
    rag_agent,
    base_rate=11.5
)


# =============================================================================
# TITLE
# =============================================================================

st.title(
    "🛡️ CrediAdapt AI: Dynamic Loan Recommendation & Affordability Engine"
)

st.caption(
    "AI-powered affordability, FOIR, risk and personalized loan recommendation system"
)


# =============================================================================
# SIDEBAR - APPLICANT PROFILE
# =============================================================================

with st.sidebar:

    st.header("👤 Applicant Profile")

    income = st.number_input(
        "Monthly Income (₹)",
        min_value=25000,
        max_value=500000,
        value=85000,
        step=5000
    )

    age = st.slider(
        "Age",
        min_value=21,
        max_value=60,
        value=28
    )

    gender = st.selectbox(
        "Gender",
        ["Female", "Male", "Other"]
    )

    rent = st.number_input(
        "House Rent (₹)",
        min_value=0,
        max_value=150000,
        value=18000,
        step=1000
    )

    existing_emi = st.number_input(
        "Existing EMI (₹)",
        min_value=0,
        max_value=150000,
        value=5000,
        step=1000
    )

    cibil = st.slider(
        "CIBIL Score",
        min_value=450,
        max_value=850,
        value=785
    )

    st.divider()

    st.subheader("💰 Loan Request")

    loan_amt = st.number_input(
        "Requested Principal (₹)",
        min_value=50000,
        max_value=3000000,
        value=600000,
        step=25000
    )

    tenure = st.selectbox(
        "Tenure (Months)",
        [12, 24, 36, 48, 60, 84],
        index=2
    )

    closure_m = st.slider(
        "Early Closure Month Simulation",
        min_value=6,
        max_value=tenure,
        value=min(18, tenure)
    )


# =============================================================================
# EVALUATE BUTTON
# =============================================================================

if st.button(
    "🚀 Evaluate & Generate Loan Offer",
    type="primary",
    use_container_width=True
):

    # =========================================================================
    # 1. CUSTOMER PROFILE
    # =========================================================================

    profile = profiler.process({

        "customer_id": "CUST-AUTO",

        "monthly_income":
            income,

        "age":
            age,

        "gender":
            gender,

        "house_rent":
            rent,

        "existing_emi":
            existing_emi,

        "cibil_score":
            cibil,

        "requested_loan":
            loan_amt,

        "requested_tenure":
            tenure
    })


    # =========================================================================
    # 2. AFFORDABILITY / FOIR
    # =========================================================================

    aff = affordability.assess(profile)


    # Safely retrieve values
    is_eligible = aff.get(
        "is_eligible",
        False
    )

    decision_type = aff.get(
        "decision_type",
        "UNKNOWN"
    )

    requested_foir = aff.get(
        "requested_foir",
        aff.get("foir", 0.0)
    )

    final_foir = aff.get(
        "foir",
        0.0
    )

    adjusted_loan = aff.get(
        "adjusted_loan",
        0.0
    )

    adjusted_tenure = aff.get(
        "adjusted_tenure",
        0
    )

    eligibility_message = aff.get(
        "eligibility_message",
        "The requested loan is not eligible based on the information provided."
    )

    reason = aff.get(
        "reason",
        ""
    )


    # =========================================================================
    # 3. INTEREST RATE / DISCOUNTS
    # =========================================================================

    rate, discounts = offer_engine.compute_discounted_rate(
        profile
    )


    # =========================================================================
    # 4. HANDLE COMPLETE REJECTION
    # =========================================================================
    #
    # IMPORTANT:
    #
    # If loan is rejected:
    #
    # adjusted_loan = 0
    #
    # Therefore we DO NOT run the normal loan simulator.
    #
    # This completely prevents:
    #
    # KeyError: outstanding_principal
    #
    # and also avoids displaying fake EMI/foreclosure information.
    # =========================================================================

    if not is_eligible:

        sim = {
            "is_simulatable": False,
            "emi": 0.0,
            "total_interest": 0.0,
            "total_cost": 0.0,
            "early_closure": {
                "closure_month": 0,
                "outstanding_principal": 0.0,
                "interest_paid_until_closure": 0.0,
                "interest_saved": 0.0,
                "total_repaid": 0.0
            },
            "message":
                "Loan simulation unavailable because the loan is not eligible."
        }

    else:

        # =====================================================================
        # 5. SIMULATE ONLY ELIGIBLE / COUNTER-OFFER LOAN
        # =====================================================================

        sim = simulator.generate_schedule_and_closure(
            adjusted_loan,
            rate,
            adjusted_tenure,
            closure_m
        )


    # =========================================================================
    # 6. RISK ANALYSIS
    # =========================================================================
    #
    # Risk engine itself handles zero/rejected loans.
    # =========================================================================

    risk = risk_engine.evaluate_risk(
        profile,
        adjusted_loan,
        adjusted_tenure,
        final_foir
    )


    # =========================================================================
    # TOP STATUS
    # =========================================================================

    st.divider()


    # =========================================================================
    # CASE A - FULL REJECTION
    # =========================================================================

    if not is_eligible:

        st.error(
            "❌ LOAN NOT ELIGIBLE"
        )

        st.warning(
            "The requested loan is not eligible based on the information provided."
        )

        st.markdown(
            f"""
### 📋 Eligibility Explanation

**Requested Loan:** ₹{loan_amt:,.0f}

**Requested Tenure:** {tenure} months

**Calculated FOIR:** {requested_foir * 100:.2f}%

**Maximum Allowed FOIR:** 60%

**Reason:**  
{reason}
"""
        )

        st.info(
            "💡 Please reduce the requested loan amount, reduce existing "
            "financial obligations, or provide additional verified income "
            "information and try again."
        )

        # ---------------------------------------------------------------------
        # Rejection Metrics
        # ---------------------------------------------------------------------

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Requested Loan",
            f"₹{loan_amt:,.0f}"
        )

        c2.metric(
            "Requested FOIR",
            f"{requested_foir * 100:.1f}%"
        )

        c3.metric(
            "Maximum FOIR",
            "60%"
        )


        st.divider()


        # ---------------------------------------------------------------------
        # Do NOT show foreclosure because loan does not exist
        # ---------------------------------------------------------------------

        st.subheader(
            "🚫 Loan Simulation"
        )

        st.info(
            "Early foreclosure simulation is unavailable because "
            "the requested loan is not eligible."
        )


        # ---------------------------------------------------------------------
        # Risk
        # ---------------------------------------------------------------------

        st.subheader(
            "🧠 Risk Assessment"
        )

        st.info(
            "Risk assessment was not performed because the requested "
            "loan amount is not eligible."
        )


        # ---------------------------------------------------------------------
        # Policies
        # ---------------------------------------------------------------------

        st.subheader(
            "📜 Applicable Policies"
        )

        if discounts.concessions:

            for d in discounts.concessions:

                st.success(
                    f"**{d.policy_name}** "
                    f"(-{d.discount_percentage:.2f}%): "
                    f"{d.justification}"
                )

        else:

            st.info(
                "No promotional campaign discounts triggered."
            )


        # ---------------------------------------------------------------------
        # STOP HERE
        #
        # This prevents the application from continuing into the normal
        # approved-loan UI.
        # ---------------------------------------------------------------------

        st.stop()


    # =========================================================================
    # CASE B - COUNTER OFFER
    # =========================================================================

    if decision_type == "COUNTER_OFFER":

        st.warning(
            "⚠️ REQUESTED LOAN NOT ELIGIBLE — COUNTER-OFFER AVAILABLE"
        )

        st.markdown(
            f"""
### 💡 Recommended Alternative

The requested loan of **₹{loan_amt:,.0f}** exceeds the permitted
affordability limit.

Based on the information provided, an approximate loan amount of:

### ₹{adjusted_loan:,.0f}

may fit within the 60% FOIR threshold.

**Important:** This is a counter-offer recommendation, not guaranteed approval.
"""


        )


    # =========================================================================
    # CASE C - APPROVED AS REQUESTED
    # =========================================================================

    else:

        st.success(
            "✅ REQUESTED LOAN FITS WITHIN THE AFFORDABILITY LIMIT"
        )


    # =========================================================================
    # MAIN METRICS
    # =========================================================================

    col1, col2, col3, col4 = st.columns(4)


    # Loan amount

    delta_value = (
        adjusted_loan - loan_amt
    )

    if adjusted_loan == loan_amt:

        loan_delta = "Full Amount"

    else:

        loan_delta = f"₹{delta_value:,.0f}"


    col1.metric(
        "Recommended Loan",
        f"₹{adjusted_loan:,.0f}",
        delta=loan_delta
    )


    # Interest rate

    if rate < 11.5:

        rate_delta = (
            f"-{11.5 - rate:.2f}% Concession"
        )

    else:

        rate_delta = "Base Rate"


    col2.metric(
        "Interest Rate",
        f"{rate:.2f}%",
        delta=rate_delta
    )


    # EMI

    emi = sim.get(
        "emi",
        0.0
    )

    col3.metric(
        "Monthly EMI",
        f"₹{emi:,.2f}"
    )


    # FOIR

    col4.metric(
        "FOIR Utilization",
        f"{final_foir * 100:.1f}%",
        delta="≤ 60% Limit"
    )


    st.divider()


    # =========================================================================
    # TABS
    # =========================================================================

    tab1, tab2, tab3, tab4 = st.tabs([
        "💰 Offer & Early Closure",
        "🧠 Explainable AI",
        "📜 Applied Policies",
        "📋 Decision Summary"
    ])


    # =========================================================================
    # TAB 1 - EARLY CLOSURE
    # =========================================================================

    with tab1:

        st.subheader(
            "💰 Early Foreclosure Benefits"
        )

        early = sim.get(
            "early_closure",
            {}
        )

        outstanding = early.get(
            "outstanding_principal",
            0.0
        )

        interest_saved = early.get(
            "interest_saved",
            0.0
        )

        total_repaid = early.get(
            "total_repaid",
            0.0
        )

        closure_month_display = early.get(
            "closure_month",
            closure_m
        )


        c1, c2, c3 = st.columns(3)


        c1.metric(
            f"Balance at Month {closure_month_display}",
            f"₹{outstanding:,.2f}"
        )


        c2.metric(
            "Interest Saved by Closing Early",
            f"₹{interest_saved:,.2f}"
        )


        c3.metric(
            "Total Cost with Early Closure",
            f"₹{total_repaid:,.2f}"
        )


        st.info(
            sim.get(
                "message",
                "Early closure simulation calculated successfully."
            )
        )


    # =========================================================================
    # TAB 2 - RISK / SHAP
    # =========================================================================

    with tab2:

        st.subheader(
            "🧠 Risk & Feature Impact Analysis"
        )


        default_probability = risk.get(
            "default_probability",
            0.0
        )

        risk_band = risk.get(
            "risk_band",
            "NOT_EVALUATED"
        )


        st.write(
            f"**Predicted Default Risk:** "
            f"`{default_probability * 100:.2f}%` "
            f"({risk_band} RISK)"
        )


        shap_factors = risk.get(
            "top_shap_factors",
            []
        )


        if shap_factors:

            shap_df = pd.DataFrame(
                shap_factors,
                columns=[
                    "Feature",
                    "SHAP Impact Score"
                ]
            )

            st.dataframe(
                shap_df,
                use_container_width=True
            )

        else:

            st.info(
                "Risk feature analysis was not performed "
                "because the loan was not eligible."
            )


    # =========================================================================
    # TAB 3 - POLICIES
    # =========================================================================

    with tab3:

        st.subheader(
            "📜 Demographic & Campaign Rules Applied"
        )


        if discounts.concessions:

            for d in discounts.concessions:

                st.success(
                    f"""
**{d.policy_name}**

Discount: **-{d.discount_percentage:.2f}%**

Reason: {d.justification}
"""
                )

        else:

            st.info(
                "No promotional campaign discounts triggered."
            )


    # =========================================================================
    # TAB 4 - DECISION SUMMARY
    # =========================================================================

    with tab4:

        st.subheader(
            "📋 Loan Decision Summary"
        )


        st.write(
            f"**Customer ID:** {profile.customer_id}"
        )

        st.write(
            f"**Monthly Income:** ₹{income:,.2f}"
        )

        st.write(
            f"**Existing EMI:** ₹{existing_emi:,.2f}"
        )

        st.write(
            f"**House Rent:** ₹{rent:,.2f}"
        )

        st.write(
            f"**CIBIL Score:** {cibil}"
        )

        st.divider()


        st.write(
            f"**Requested Loan:** ₹{loan_amt:,.0f}"
        )

        st.write(
            f"**Recommended Loan:** ₹{adjusted_loan:,.0f}"
        )

        st.write(
            f"**Requested FOIR:** {requested_foir * 100:.2f}%"
        )

        st.write(
            f"**Final FOIR:** {final_foir * 100:.2f}%"
        )

        st.write(
            f"**Decision:** {decision_type}"
        )

        st.write(
            f"**Interest Rate:** {rate:.2f}%"
        )

        st.write(
            f"**Monthly EMI:** ₹{emi:,.2f}"
        )


        if decision_type == "COUNTER_OFFER":

            st.warning(
                "The originally requested loan amount is not eligible. "
                "The displayed recommended amount is a counter-offer "
                "calculated to fit within the 60% FOIR limit."
            )

        else:

            st.success(
                "The requested loan fits within the calculated "
                "60% FOIR affordability threshold."
            )


# =============================================================================
# FOOTER
# =============================================================================

st.divider()

st.caption(
    "⚠️ This system provides an AI-based financial recommendation. "
    "Final loan approval is subject to verification of income, credit history, "
    "bank policy, documentation and other applicable eligibility criteria."
)

