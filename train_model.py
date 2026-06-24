import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

print("Loading data...")
df = pd.read_csv("loan_portfolio.csv")

sector_enc = LabelEncoder()
investor_enc = LabelEncoder()
risk_enc = LabelEncoder()

df["sector_encoded"] = sector_enc.fit_transform(df["sector"])
df["investor_encoded"] = investor_enc.fit_transform(df["investor_backing"])
df["risk_encoded"] = risk_enc.fit_transform(df["risk_label"])

FEATURES = [
    "sector_encoded",
    "loan_amount_lakh",
    "tenure_months",
    "promoter_score",
    "repayment_track",
    "irr_percent",
    "investor_encoded",
    "years_in_operation"
]

X = df[FEATURES]
y = df["risk_encoded"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training on {len(X_train)} loans, testing on {len(X_test)} loans\n")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
risk_classes = risk_enc.classes_

print("=" * 50)
print("MODEL PERFORMANCE")
print("=" * 50)
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.1f}%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=risk_classes))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=risk_classes, columns=risk_classes)
print(cm_df)

print("\n" + "=" * 50)
print("FEATURE IMPORTANCE (what drives credit risk)")
print("=" * 50)
feature_labels = [
    "Sector", "Loan Amount (₹L)", "Tenure (months)",
    "Promoter Score", "Repayment Track", "IRR (%)",
    "Investor Backing", "Years in Operation"
]
importances = model.feature_importances_
importance_df = pd.DataFrame({
    "Feature": feature_labels,
    "Importance": importances
}).sort_values("Importance", ascending=False)

for _, row in importance_df.iterrows():
    bar = "█" * int(row["Importance"] * 50)
    print(f"  {row['Feature']:<22} {bar} {row['Importance']:.3f}")

joblib.dump(model, "model.pkl")
joblib.dump(sector_enc, "sector_encoder.pkl")
joblib.dump(investor_enc, "investor_encoder.pkl")
joblib.dump(risk_enc, "risk_encoder.pkl")

print("\nSaved: model.pkl, sector_encoder.pkl, investor_encoder.pkl, risk_encoder.pkl")
print("Ready for Streamlit dashboard.")
