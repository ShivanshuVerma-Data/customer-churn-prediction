import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics import precision_score, recall_score

# ========================
# Paths
# ========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, "..")
MODEL_PATH = os.path.join(ROOT_DIR, "outputs", "model.pkl")
DATA_PATH = os.path.join(ROOT_DIR, "data", "real.csv")

# ========================
# Load model
# ========================
if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Run main.py to generate outputs/model.pkl")
    st.stop()

model = joblib.load(MODEL_PATH)

# ========================
# Utilities (same logic as training)
# ========================
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering (must match training)"""
    df = df.copy()

    # Ensure numeric
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Engineered features
    df["AvgCharges"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["TenureGroup"] = df["tenure"] // 12
    df["ChargePerTenure"] = df["MonthlyCharges"] * df["tenure"]
    df["IsNewCustomer"] = (df["tenure"] < 12).astype(int)
    df["HighSpender"] = (df["MonthlyCharges"] > df["MonthlyCharges"].median()).astype(int)

    return df


def encode_and_align(df: pd.DataFrame, model):
    """One-hot encode and align to model feature space"""
    df_enc = pd.get_dummies(df)

    if not hasattr(model, "feature_names_in_"):
        st.error("Model does not store feature names. Retrain with sklearn >= 1.0")
        st.stop()

    cols = model.feature_names_in_
    df_enc = df_enc.reindex(columns=cols, fill_value=0)
    return df_enc


def load_validation_split():
    """
    Load the original dataset and recreate the SAME preprocessing
    to compute an optimized threshold (precision-focused).
    """
    if not os.path.exists(DATA_PATH):
        return None

    df = pd.read_csv(DATA_PATH).copy()

    # Minimal cleaning to match training
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Target
    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Features
    df = build_features(df)

    # Encode
    df = pd.get_dummies(df)

    # Align
    cols = model.feature_names_in_
    df = df.reindex(columns=cols.tolist() + ["Churn"], fill_value=0)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # Simple split (same seed/ratio)
    from sklearn.model_selection import train_test_split
    _, X_val, _, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_val, y_val


def find_threshold_with_constraints(model, X_val, y_val, min_recall=0.6):
    """
    Maximize precision subject to recall >= min_recall
    """
    y_probs = model.predict_proba(X_val)[:, 1]
    thresholds = np.linspace(0.3, 0.95, 100)

    best_t = 0.5
    best_p = 0.0

    for t in thresholds:
        preds = (y_probs >= t).astype(int)
        p = precision_score(y_val, preds, zero_division=0)
        r = recall_score(y_val, preds)

        if r >= min_recall and p > best_p:
            best_p = p
            best_t = t

    return best_t, best_p


def get_top_contributors(model, input_row: pd.DataFrame, top_k=5):
    """
    Use feature_importances_ if available to show top contributors
    (magnitude-based approximation).
    """
    if not hasattr(model, "feature_importances_"):
        return None

    importances = model.feature_importances_
    cols = input_row.columns

    # contribution proxy = importance * value
    contrib = importances * input_row.iloc[0].values
    idx = np.argsort(np.abs(contrib))[-top_k:][::-1]

    results = []
    for i in idx:
        results.append((cols[i], float(contrib[i])))

    return results


# ========================
# Compute optimized threshold (once)
# ========================
threshold = 0.5
threshold_info = ""

val_split = load_validation_split()
if val_split is not None:
    X_val, y_val = val_split
    t, p = find_threshold_with_constraints(model, X_val, y_val, min_recall=0.6)
    threshold = t
    threshold_info = f"Optimized threshold: {threshold:.2f} (precision-focused)"

# ========================
# UI
# ========================
st.set_page_config(page_title="Customer Churn Predictor", layout="centered")

st.title("Customer Churn Prediction")

if threshold_info:
    st.caption(threshold_info)

st.write("Provide customer details to estimate churn risk.")

# Inputs
tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 50.0)
total_charges = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)

contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

# ========================
# Input validation
# ========================
errors = []

if total_charges < monthly_charges:
    errors.append("Total Charges should not be less than Monthly Charges.")

if tenure == 0 and total_charges > 0:
    errors.append("Tenure is 0 but Total Charges is non-zero.")

if monthly_charges <= 0:
    errors.append("Monthly Charges must be positive.")

if errors:
    for e in errors:
        st.error(e)

# ========================
# Build input
# ========================
input_dict = {
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "Contract": contract,
    "InternetService": internet,
    "PaymentMethod": payment
}

input_df = pd.DataFrame([input_dict])
input_df = build_features(input_df)
input_df = encode_and_align(input_df, model)

# ========================
# Prediction
# ========================
if st.button("Predict") and not errors:

    prob = model.predict_proba(input_df)[0][1]
    pred = int(prob >= threshold)

    st.subheader("Result")
    st.write(f"Churn Probability: {prob:.3f}")
    st.write(f"Decision Threshold: {threshold:.2f}")

    if pred == 1:
        st.write("Prediction: Likely to churn")
    else:
        st.write("Prediction: Likely to stay")

    # Risk band
    if prob > 0.75:
        st.write("Risk Level: High")
    elif prob > 0.6:
        st.write("Risk Level: Medium")
    else:
        st.write("Risk Level: Low")

    # ========================
    # Top contributing features
    # ========================
    st.subheader("Top Contributing Features")

    top = get_top_contributors(model, input_df, top_k=5)

    if top is None:
        st.write("Feature contributions not available for this model.")
    else:
        for name, val in top:
            st.write(f"{name}: {val:.4f}")