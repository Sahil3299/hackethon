import numpy as np
import pandas as pd

def generate_loan_dataset(n_samples=5000, seed=42):
    np.random.seed(seed)
    
    monthly_income = np.random.lognormal(mean=10.8, sigma=0.5, size=n_samples)
    monthly_income = np.clip(monthly_income, 25000, 350000)
    
    age = np.random.randint(21, 62, size=n_samples)
    gender = np.random.choice(['Female', 'Male', 'Other'], size=n_samples, p=[0.45, 0.50, 0.05])
    
    rent_ratio = np.random.uniform(0.10, 0.35, size=n_samples)
    house_rent = monthly_income * rent_ratio
    
    existing_emi_ratio = np.random.uniform(0.0, 0.35, size=n_samples)
    existing_emi = monthly_income * existing_emi_ratio
    
    cibil_score = np.random.normal(loc=720, scale=65, size=n_samples)
    cibil_score = np.clip(cibil_score, 450, 850).astype(int)
    
    loan_amount = np.random.uniform(100000, 2500000, size=n_samples)
    tenure_months = np.random.choice([12, 24, 36, 48, 60, 84], size=n_samples)
    
    # Calculate baseline FOIR
    r = (11.0 / 12) / 100
    proposed_emi = (loan_amount * r * ((1 + r) ** tenure_months)) / (((1 + r) ** tenure_months) - 1)
    foir = (existing_emi + house_rent + proposed_emi) / monthly_income
    
    # Ground-truth risk probability for training
    risk_logits = (
        - 0.008 * (cibil_score - 600)
        + 4.2 * (foir - 0.50)
        - 0.000015 * monthly_income
        + 0.025 * np.maximum(0, age - 50)
        - 0.20 * (gender == 'Female')
        + np.random.normal(0, 0.4, size=n_samples)
    )
    prob_default = 1 / (1 + np.exp(-risk_logits))
    defaulted = (prob_default > 0.45).astype(int)
    
    df = pd.DataFrame({
        'monthly_income': np.round(monthly_income, 2),
        'age': age,
        'gender': gender,
        'house_rent': np.round(house_rent, 2),
        'existing_emi': np.round(existing_emi, 2),
        'cibil_score': cibil_score,
        'loan_amount': np.round(loan_amount, 2),
        'tenure_months': tenure_months,
        'foir': np.round(foir, 4),
        'defaulted': defaulted
    })
    
    df.to_csv('loan_risk_dataset.csv', index=False)
    print(f"Dataset generated: {df.shape[0]} rows saved to loan_risk_dataset.csv")
    return df

if __name__ == "__main__":
    generate_loan_dataset()