# UCIC Impact Loan Risk Classifier

A machine learning dashboard that automates credit risk assessment for impact enterprise loan portfolios — built as a prototype aligned with UC Inclusive Credit's (UCIC) lending focus.

🔗 **[Live Demo](#)** &nbsp;|&nbsp; 📂 **[GitHub](#)**

---

## Why this exists

UCIC lends to enterprises across impact sectors — Food & Agriculture, Healthcare, Clean Energy, Microfinance, Education, and Affordable Housing. Evaluating credit risk across a growing portfolio is a time-intensive manual process.

This prototype demonstrates how RPA + AI can automate what a credit analyst does daily:
- Flag high-risk loans instantly across the full portfolio
- Score new loan applications in seconds
- Visualise sector concentration and risk distribution

---

## Dashboard

| Tab | What it does |
|-----|-------------|
| 📊 Portfolio Overview | Sector allocation, risk distribution, heatmap, scatter analysis |
| 🚨 Risk Flags | Color-coded loan table with auto-generated plain-English risk explanations |
| 🔍 Assess New Loan | Input a borrower profile → get Low / Medium / High risk with confidence % |

---

## How it works

```
Simulated portfolio data (200 loans)
        ↓
Feature engineering + label encoding
        ↓
Random Forest Classifier (scikit-learn)
        ↓
Streamlit dashboard with Plotly charts
```

**Key risk signals the model uses (by importance):**

| Rank | Feature | Importance | Why it matters |
|------|---------|-----------|----------------|
| 1 | Repayment track record | 28% | Strongest predictor of future default |
| 2 | Promoter / management score | 24% | Founder quality drives enterprise outcomes |
| 3 | Years in operation | 11% | Mature enterprises default less |
| 4 | Loan amount | 10% | Higher exposure = higher risk |
| 5 | IRR | 10% | Higher yield often signals riskier borrower |

**Model performance:** 77.5% accuracy · High-risk loans caught with 100% precision

---

## Data note

Enterprise impact loan data from NBFCs is proprietary and never publicly released. This prototype uses domain-informed synthetic data built around UCIC's actual lending sectors and credit evaluation criteria — promoter quality, investor backing, repayment track record, and IRR. The data generation logic mirrors how UCIC's credit team thinks about risk.

---

## Tech stack

| Layer | Tool |
|-------|------|
| Data | Python, NumPy, Pandas |
| Model | scikit-learn (Random Forest) |
| Dashboard | Streamlit, Plotly |
| Deployment | Streamlit Community Cloud |

---

## Run locally

```bash
# 1. Clone the repo
git clone https://github.com/your-username/ucic-nbfc-risk-classifier
cd ucic-nbfc-risk-classifier

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate data + train model
python generate_data.py
python train_model.py

# 4. Launch dashboard
python -m streamlit run app.py
```

---

## Project structure

```
ucic-nbfc-risk-classifier/
├── generate_data.py       # Simulates 200 impact loans across UCIC sectors
├── train_model.py         # Trains RandomForest, prints feature importance
├── app.py                 # Streamlit dashboard (3 tabs)
├── loan_portfolio.csv     # Generated dataset
├── model.pkl              # Trained classifier
├── sector_encoder.pkl     # Label encoder for sectors
├── investor_encoder.pkl   # Label encoder for investor backing
├── risk_encoder.pkl       # Label encoder for risk labels
├── requirements.txt
└── README.md
```

---

## Requirements

```
streamlit
pandas
numpy
scikit-learn
plotly
joblib
```

---

## About UCIC

[UC Inclusive Credit Pvt. Ltd.](https://www.ucinclusive.in) is an RBI-registered, impact-focused NBFC based in Bangalore. It extends loans to enterprises driving positive social, financial, and environmental impact across sectors including Food & Agriculture, Healthcare, Clean Energy, Financial Inclusion, Education, and Affordable Housing.

---

*Built as part of an internship application for the RPA & AI Intern role at UCIC.*
