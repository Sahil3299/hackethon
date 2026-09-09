from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
import xgboost as xgb
import shap

from customer_profiling_agent import ExtractedProfile


class RiskMLAgent:

    def __init__(
        self,
        data_path: str = "loan_risk_dataset.csv",
        model_path: Optional[str] = None,
        random_state: int = 42,
    ):
        self.base_dir = Path(__file__).resolve().parent
        self.data_file = self.base_dir / data_path
        self.model_file = Path(model_path) if model_path else self.base_dir / "risk_model.joblib"
        self.random_state = random_state

        # NOTE: gender_encoded is intentionally NOT part of the feature set.
        # This preserves fair-lending practice: gender must not be a model
        # input that swings the credit decision.
        self.features = [
            "monthly_income", "age", "house_rent", "existing_emi",
            "cibil_score", "loan_amount", "tenure_months", "foir",
        ]

        self.model = None
        self.explainer = None
        self.metrics: Dict[str, Any] = {
            "auc": None,
            "confusion_matrix": None,
            "threshold": None,
            "trained_rows": 0,
            "fingerprint": None,
        }

        # Load a persisted model if present; else train from the dataset.
        if self.model_file.exists():
            try:
                state = joblib.load(self.model_file)
                self.model = state["model"]
                self.explainer = state.get("explainer")
                self.metrics = state.get("metrics", self.metrics)
                return
            except Exception as exc:
                _ = exc

        if self.data_file.exists():
            self._train_model(self.data_file)

    # ------------------------------------------------------------------
    # Training with an 80/20 split and model-quality reporting.
    # ------------------------------------------------------------------
    def _train_model(self, data_path, force: bool = False):
        df = pd.read_csv(data_path)
        required_columns = [
            "monthly_income", "age", "house_rent", "existing_emi",
            "cibil_score", "loan_amount", "tenure_months", "foir", "defaulted",
        ]
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Dataset missing columns: {missing}")

        # Drop gender-derived column if present so it can never leak in.
        if "gender_encoded" in df.columns:
            df = df.drop(columns=["gender_encoded"])

        X = df[self.features]
        y = df["defaulted"]

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=self.random_state, stratify=y,
        )

        self.model = xgb.XGBClassifier(
            n_estimators=75, max_depth=4, learning_rate=0.08,
            random_state=self.random_state, eval_metric="logloss",
        )
        self.model.fit(X_train, y_train)
        self.explainer = shap.TreeExplainer(self.model)

        self.metrics = self._compute_metrics(X_test, y_test)
        if not force:
            self._persist()

    def _compute_metrics(self, X_test, y_test) -> Dict[str, Any]:
        from sklearn.metrics import auc, confusion_matrix, roc_curve
        probs = self.model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, probs)
        roc_auc = float(auc(fpr, tpr))
        threshold = 0.50
        preds = (probs >= threshold).astype(int)
        cm = confusion_matrix(y_test, preds).tolist()
        return {
            "auc": round(roc_auc, 4),
            "confusion_matrix": cm,
            "threshold": threshold,
            "test_rows": int(len(y_test)),
            "trained_rows": int(len(X_test) + len(y_test)),
            "fingerprint": f"features={self.features}",
        }

    def _persist(self):
        joblib.dump(
            {"model": self.model, "explainer": self.explainer, "metrics": self.metrics},
            self.model_file,
        )

    def retrain(self, data_path: Optional[str] = None):
        """Deliberate retrain when the dataset changes."""
        path = self.data_file if data_path is None else self.base_dir / data_path
        self._train_model(path, force=True)

    def get_metrics(self) -> Dict[str, Any]:
        return dict(self.metrics)

    # ------------------------------------------------------------------
    # Inference.
    # ------------------------------------------------------------------
    def evaluate_risk(
        self,
        profile: ExtractedProfile,
        loan_amt: float,
        tenure: int,
        foir: float
    ) -> Dict[str, Any]:

        if loan_amt <= 0 or tenure <= 0:
            return {
                "default_probability": 0.0,
                "risk_band": "NOT_EVALUATED",
                "top_shap_factors": [],
                "message": "Risk assessment was not performed because the requested loan is not eligible."
            }

        if self.model is None:
            return {
                "default_probability": 0.0,
                "risk_band": "NOT_EVALUATED",
                "top_shap_factors": [],
                "message": "Risk model is not available.",
            }

        input_data = pd.DataFrame([{
            "monthly_income": profile.monthly_income,
            "age": profile.age,
            "house_rent": profile.house_rent,
            "existing_emi": profile.existing_emi,
            "cibil_score": profile.cibil_score,
            "loan_amount": loan_amt,
            "tenure_months": tenure,
            "foir": foir,
        }])
        input_data = input_data[self.features]

        default_prob = float(self.model.predict_proba(input_data)[0][1])

        shap_values = self.explainer(input_data)
        values = shap_values.values[0]
        feature_impacts = dict(zip(self.features, values))
        sorted_impacts = sorted(feature_impacts.items(), key=lambda x: abs(x[1]), reverse=True)

        if default_prob < 0.25:
            risk_band = "LOW"
        elif default_prob < 0.50:
            risk_band = "MEDIUM"
        else:
            risk_band = "HIGH"

        return {
            "default_probability": round(default_prob, 4),
            "risk_band": risk_band,
            "top_shap_factors": sorted_impacts[:3],
        }