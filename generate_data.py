import numpy as np
import pandas as pd

np.random.seed(42)
n = 200

sectors = ["Food & Agri", "Healthcare", "Clean Energy", "Microfinance", "Education", "Affordable Housing"]
sector_weights = [0.25, 0.15, 0.20, 0.20, 0.10, 0.10]

sector_col = np.random.choice(sectors, size=n, p=sector_weights)
loan_amount_lakh = np.round(np.random.choice(
    [50, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000],
    size=n,
    p=[0.05, 0.10, 0.15, 0.20, 0.20, 0.15, 0.08, 0.04, 0.02, 0.01]
), 0)
tenure_months = np.random.choice([12, 18, 24, 36, 48, 60], size=n, p=[0.10, 0.15, 0.25, 0.30, 0.12, 0.08])
promoter_score = np.round(np.clip(np.random.normal(65, 15, n), 20, 100), 1)
repayment_track = np.round(np.clip(np.random.normal(72, 18, n), 0, 100), 1)
irr = np.round(np.clip(np.random.normal(16, 3, n), 10, 26), 2)
investor_backing = np.random.choice(["Strong", "Moderate", "Weak"], size=n, p=[0.40, 0.40, 0.20])
years_in_operation = np.round(np.clip(np.random.normal(5, 3, n), 0.5, 20), 1)

def assign_risk(promoter, repayment, investor, loan_amt, years):
    score = 0
    score += 3 if promoter >= 75 else (2 if promoter >= 55 else 0)
    score += 3 if repayment >= 80 else (2 if repayment >= 60 else 0)
    score += 2 if investor == "Strong" else (1 if investor == "Moderate" else 0)
    score += 1 if loan_amt <= 200 else 0
    score += 1 if years >= 3 else 0
    noise = np.random.randint(-1, 2)
    score = np.clip(score + noise, 0, 10)
    if score >= 7:
        return "Low"
    elif score >= 4:
        return "Medium"
    else:
        return "High"

risk_labels = [
    assign_risk(promoter_score[i], repayment_track[i], investor_backing[i], loan_amount_lakh[i], years_in_operation[i])
    for i in range(n)
]

df = pd.DataFrame({
    "loan_id": [f"UCIC-{1000+i}" for i in range(n)],
    "sector": sector_col,
    "loan_amount_lakh": loan_amount_lakh,
    "tenure_months": tenure_months,
    "promoter_score": promoter_score,
    "repayment_track": repayment_track,
    "irr_percent": irr,
    "investor_backing": investor_backing,
    "years_in_operation": years_in_operation,
    "risk_label": risk_labels
})

df.to_csv("loan_portfolio.csv", index=False)

print("Dataset generated: loan_portfolio.csv")
print(f"Shape: {df.shape}")
print(f"\nRisk distribution:\n{df['risk_label'].value_counts()}")
print(f"\nSector distribution:\n{df['sector'].value_counts()}")
print(f"\nSample rows:\n{df.head(5).to_string()}")
