from pathlib import Path
from typing import Dict, Any

import pandas as pd
import xgboost as xgb
import shap

from customer_profiling_agent import ExtractedProfile


# =============================================================================
# RISK ML AGENT
# =============================================================================

class RiskMLAgent:

    def __init__(
        self,
        data_path: str = "loan_risk_dataset.csv"
    ):

        self.base_dir = Path(__file__).resolve().parent

        data_file = self.base_dir / data_path

        if not data_file.exists():

            raise FileNotFoundError(
                f"Dataset not found: {data_file}"
            )

        self.features = [
            "monthly_income",
            "age",
            "house_rent",
            "existing_emi",
            "cibil_score",
            "loan_amount",
            "tenure_months",
            "foir",
            "gender_encoded"
        ]

        self._train_model(data_file)

    # =========================================================================
    # TRAIN MODEL
    # =========================================================================

    def _train_model(
        self,
        data_path
    ):

        df = pd.read_csv(data_path)

        required_columns = [
            "monthly_income",
            "age",
            "gender",
            "house_rent",
            "existing_emi",
            "cibil_score",
            "loan_amount",
            "tenure_months",
            "foir",
            "defaulted"
        ]

        missing = [
            col
            for col in required_columns
            if col not in df.columns
        ]

        if missing:

            raise ValueError(
                f"Dataset missing columns: {missing}"
            )

        df = df.copy()

        df["gender_encoded"] = (
            df["gender"]
            .astype(str)
            .str.strip()
            .str.capitalize()
            .map({
                "Female": 0,
                "Male": 1,
                "Other": 2
            })
            .fillna(2)
        )

        X = df[self.features]
        y = df["defaulted"]

        self.model = xgb.XGBClassifier(
            n_estimators=75,
            max_depth=4,
            learning_rate=0.08,
            random_state=42,
            eval_metric="logloss"
        )

        self.model.fit(X, y)

        self.explainer = shap.TreeExplainer(
            self.model
        )

    # =========================================================================
    # RISK EVALUATION
    # =========================================================================

    def evaluate_risk(
        self,
        profile: ExtractedProfile,
        loan_amt: float,
        tenure: int,
        foir: float
    ) -> Dict[str, Any]:

        # ---------------------------------------------------------------------
        # REJECTED LOAN
        # ---------------------------------------------------------------------

        if loan_amt <= 0 or tenure <= 0:

            return {

                "default_probability":
                    0.0,

                "risk_band":
                    "NOT_EVALUATED",

                "top_shap_factors":
                    [],

                "message":
                    "Risk assessment was not performed because "
                    "the requested loan is not eligible."
            }

        # ---------------------------------------------------------------------
        # GENDER
        # ---------------------------------------------------------------------

        gender_code = {
            "Female": 0,
            "Male": 1,
            "Other": 2
        }.get(
            profile.gender.strip().capitalize(),
            2
        )

        input_data = pd.DataFrame([
            {

                "monthly_income":
                    profile.monthly_income,

                "age":
                    profile.age,

                "house_rent":
                    profile.house_rent,

                "existing_emi":
                    profile.existing_emi,

                "cibil_score":
                    profile.cibil_score,

                "loan_amount":
                    loan_amt,

                "tenure_months":
                    tenure,

                "foir":
                    foir,

                "gender_encoded":
                    gender_code
            }
        ])

        input_data = input_data[
            self.features
        ]

        # ---------------------------------------------------------------------
        # DEFAULT PROBABILITY
        # ---------------------------------------------------------------------

        default_prob = float(
            self.model.predict_proba(
                input_data
            )[0][1]
        )

        # ---------------------------------------------------------------------
        # SHAP
        # ---------------------------------------------------------------------

        shap_values = self.explainer(
            input_data
        )

        values = shap_values.values[0]

        feature_impacts = dict(
            zip(
                self.features,
                values
            )
        )

        sorted_impacts = sorted(
            feature_impacts.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        # ---------------------------------------------------------------------
        # RISK BAND
        # ---------------------------------------------------------------------

        if default_prob < 0.25:

            risk_band = "LOW"

        elif default_prob < 0.50:

            risk_band = "MEDIUM"

        else:

            risk_band = "HIGH"

        return {

            "default_probability":
                round(default_prob, 4),

            "risk_band":
                risk_band,

            "top_shap_factors":
                sorted_impacts[:3]
        }