"""
Loan Approval Prediction Dashboard
====================================
A single-file Streamlit application that handles data preprocessing,
model training, EDA visualisation, and interactive prediction.

Usage:
    streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_PATH = "sp1.csv"

CATEGORICAL_COLS = [
    "Gender", "Married", "Dependents",
    "Education", "Self_Employed", "Property_Area",
]
NUMERICAL_COLS = [
    "ApplicantIncome", "CoapplicantIncome",
    "LoanAmount", "Loan_Amount_Term", "Credit_History",
]
TARGET_COL = "Loan_Status"
DROP_COLS  = ["Loan_ID"]

# ---------------------------------------------------------------------------
# Backend: data loading, preprocessing, and model training
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Training model… please wait.")
def load_and_train():
    """
    Load sp1.csv, impute missing values, encode categoricals, train a
    RandomForestClassifier, and return everything needed by the UI.

    Returns
    -------
    model        : trained RandomForestClassifier
    label_encoders : dict of {column: LabelEncoder} for every categorical
    feature_order  : list of column names in the exact order the model expects
    raw_df         : original (un-preprocessed) DataFrame for EDA
    accuracy       : float – test-set accuracy
    report         : str  – sklearn classification report
    """
    # --- Load -----------------------------------------------------------
    raw_df = pd.read_csv(DATA_PATH)

    df = raw_df.copy()

    # --- Drop identifier ------------------------------------------------
    df.drop(columns=DROP_COLS, errors="ignore", inplace=True)

    # --- Impute missing values ------------------------------------------
    # Numerical  → median
    for col in NUMERICAL_COLS:
        if col in df.columns:
            df[col].fillna(df[col].median(), inplace=True)

    # Categorical → mode (most frequent)
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col].fillna(df[col].mode()[0], inplace=True)

    # Target: drop any remaining NaN rows
    df.dropna(subset=[TARGET_COL], inplace=True)

    # --- Encode categoricals --------------------------------------------
    label_encoders = {}

    # Encode target separately (Y → 1, N → 0)
    le_target = LabelEncoder()
    df[TARGET_COL] = le_target.fit_transform(df[TARGET_COL])

    # Encode feature categoricals
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

    # --- Feature / target split -----------------------------------------
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    feature_order = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Train model ----------------------------------------------------
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    # --- Evaluate -------------------------------------------------------
    y_pred   = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report   = classification_report(y_test, y_pred, target_names=["Rejected", "Approved"])

    return model, label_encoders, feature_order, raw_df, accuracy, report


# ---------------------------------------------------------------------------
# Helper: build a single-row DataFrame from sidebar inputs
# ---------------------------------------------------------------------------

def build_input_df(inputs: dict, label_encoders: dict, feature_order: list) -> pd.DataFrame:
    """
    Encode the user's raw string/numeric inputs exactly as the training
    pipeline did, and return a single-row DataFrame aligned to feature_order.
    """
    row = {}
    for feature in feature_order:
        value = inputs[feature]
        if feature in label_encoders:
            le = label_encoders[feature]
            # Handle unseen labels gracefully by falling back to the first class
            if str(value) in le.classes_:
                row[feature] = le.transform([str(value)])[0]
            else:
                row[feature] = 0
        else:
            row[feature] = value
    return pd.DataFrame([row], columns=feature_order)


# ---------------------------------------------------------------------------
# Frontend: Sidebar prediction form
# ---------------------------------------------------------------------------

def render_sidebar() -> dict:
    """Render all input widgets and return a dict of raw user inputs."""
    st.sidebar.header("📋 Applicant Details")
    st.sidebar.markdown("Fill in the form below and click **Predict** on the main page.")

    inputs = {}

    with st.sidebar.form(key="prediction_form"):
        st.subheader("Personal Information")
        inputs["Gender"]        = st.selectbox("Gender",        ["Male", "Female"])
        inputs["Married"]       = st.selectbox("Marital Status", ["Yes", "No"])
        inputs["Dependents"]    = st.selectbox("Dependents",    ["0", "1", "2", "3+"])
        inputs["Education"]     = st.selectbox("Education",     ["Graduate", "Not Graduate"])
        inputs["Self_Employed"] = st.selectbox("Self Employed", ["No", "Yes"])

        st.subheader("Financial Information")
        inputs["ApplicantIncome"]    = st.number_input("Applicant Income ($/mo)",    min_value=0,   max_value=100_000, value=5_000,  step=500)
        inputs["CoapplicantIncome"]  = st.number_input("Co-applicant Income ($/mo)", min_value=0,   max_value=50_000,  value=0,      step=500)
        inputs["LoanAmount"]         = st.number_input("Loan Amount (thousands)",    min_value=1,   max_value=700,     value=150,    step=5)
        inputs["Loan_Amount_Term"]   = st.slider(       "Loan Term (months)",        min_value=12,  max_value=480,     value=360,    step=12)
        inputs["Credit_History"]     = st.selectbox(    "Credit History",            [1, 0],        format_func=lambda x: "Good (1)" if x == 1 else "Bad (0)")

        st.subheader("Property")
        inputs["Property_Area"] = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

        submitted = st.form_submit_button("🔍 Predict", use_container_width=True)

    return inputs, submitted


# ---------------------------------------------------------------------------
# Frontend: EDA tab
# ---------------------------------------------------------------------------

def render_eda(raw_df: pd.DataFrame) -> None:
    """Render the Exploratory Data Analysis section."""
    st.header("📊 Exploratory Data Analysis")

    # --- Overview metrics -----------------------------------------------
    approved = (raw_df[TARGET_COL] == "Y").sum()
    rejected = (raw_df[TARGET_COL] == "N").sum()
    total    = len(raw_df)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Applications", total)
    col2.metric("Approved",  f"{approved}  ({approved/total*100:.1f}%)")
    col3.metric("Rejected",  f"{rejected}  ({rejected/total*100:.1f}%)")

    st.divider()

    # --- Chart row 1: Income distribution + Loan Amount distribution ----
    st.subheader("Income & Loan Amount Distributions")
    fig1, axes1 = plt.subplots(1, 2, figsize=(12, 4))
    fig1.patch.set_facecolor("#0e1117")
    for ax in axes1:
        ax.set_facecolor("#0e1117")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    # Applicant Income histogram
    axes1[0].hist(
        raw_df["ApplicantIncome"].dropna(),
        bins=40, color="#3b82d4", edgecolor="#0e1117", alpha=0.85
    )
    axes1[0].set_title("Distribution of Applicant Income")
    axes1[0].set_xlabel("Income")
    axes1[0].set_ylabel("Count")

    # Loan Amount histogram
    axes1[1].hist(
        raw_df["LoanAmount"].dropna(),
        bins=40, color="#7c5cd8", edgecolor="#0e1117", alpha=0.85
    )
    axes1[1].set_title("Distribution of Loan Amount")
    axes1[1].set_xlabel("Loan Amount (thousands)")
    axes1[1].set_ylabel("Count")

    plt.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)

    # --- Chart row 2: Loan Approval by Credit History + by Education ----
    st.subheader("Approval Rates by Key Features")
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))
    fig2.patch.set_facecolor("#0e1117")
    for ax in axes2:
        ax.set_facecolor("#0e1117")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    palette = {"Y": "#22c55e", "N": "#ef4444"}

    # Approval rate by Credit History
    ch_data = (
        raw_df.dropna(subset=["Credit_History"])
        .groupby(["Credit_History", TARGET_COL])
        .size()
        .reset_index(name="Count")
    )
    ch_data["Credit_History"] = ch_data["Credit_History"].map(
        {1.0: "Good (1)", 0.0: "Bad (0)"}
    ).fillna(ch_data["Credit_History"].astype(str))

    credit_pivot = ch_data.pivot_table(
        index="Credit_History", columns=TARGET_COL, values="Count", fill_value=0
    )
    credit_pivot.plot(
        kind="bar", ax=axes2[0],
        color=[palette.get(c, "#888") for c in credit_pivot.columns],
        edgecolor="#0e1117", rot=0
    )
    axes2[0].set_title("Loan Approval by Credit History")
    axes2[0].set_xlabel("Credit History")
    axes2[0].set_ylabel("Number of Applications")
    axes2[0].legend(title="Loan Status", labelcolor="white", facecolor="#1a1d23", edgecolor="#444")

    # Approval rate by Education
    edu_data = (
        raw_df.groupby(["Education", TARGET_COL])
        .size()
        .reset_index(name="Count")
    )
    edu_pivot = edu_data.pivot_table(
        index="Education", columns=TARGET_COL, values="Count", fill_value=0
    )
    edu_pivot.plot(
        kind="bar", ax=axes2[1],
        color=[palette.get(c, "#888") for c in edu_pivot.columns],
        edgecolor="#0e1117", rot=0
    )
    axes2[1].set_title("Loan Approval by Education")
    axes2[1].set_xlabel("Education Level")
    axes2[1].set_ylabel("Number of Applications")
    axes2[1].legend(title="Loan Status", labelcolor="white", facecolor="#1a1d23", edgecolor="#444")

    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    # --- Chart row 3: Correlation heatmap (numerical features only) -----
    st.subheader("Feature Correlation Heatmap")
    num_df = raw_df[NUMERICAL_COLS].dropna()
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    fig3.patch.set_facecolor("#0e1117")
    ax3.set_facecolor("#0e1117")
    ax3.tick_params(colors="white")
    ax3.title.set_color("white")

    sns.heatmap(
        num_df.corr(),
        annot=True, fmt=".2f", cmap="coolwarm",
        linewidths=0.5, linecolor="#0e1117",
        ax=ax3, annot_kws={"size": 9},
    )
    ax3.set_title("Correlation Matrix – Numerical Features")
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

    # --- Raw data preview -----------------------------------------------
    with st.expander("🔎 View Raw Dataset"):
        st.dataframe(raw_df, use_container_width=True)


# ---------------------------------------------------------------------------
# Frontend: Model performance tab
# ---------------------------------------------------------------------------

def render_model_metrics(accuracy: float, report: str, model, feature_order: list) -> None:
    """Show model accuracy, classification report, and feature importances."""
    st.header("🤖 Model Performance")

    st.metric("Test-Set Accuracy", f"{accuracy * 100:.2f} %")

    st.subheader("Classification Report")
    st.code(report, language="text")

    # Feature importance bar chart
    st.subheader("Feature Importances")
    importances = pd.Series(model.feature_importances_, index=feature_order).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    colors = ["#3b82d4" if i < len(importances) - 3 else "#7c5cd8" for i in range(len(importances))]
    ax.barh(importances.index, importances.values, color=colors, edgecolor="#0e1117")
    ax.set_xlabel("Importance Score")
    ax.set_title("Random Forest – Feature Importances")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main application entry point
# ---------------------------------------------------------------------------

def main():
    # ---- Page header ---------------------------------------------------
    st.title("🏦 Loan Approval Prediction Dashboard")
    st.markdown(
        "Upload **sp1.csv** to the same directory as this script, then explore "
        "the data and predict loan outcomes using the sidebar form."
    )

    # ---- Load model (cached) -------------------------------------------
    try:
        model, label_encoders, feature_order, raw_df, accuracy, report = load_and_train()
    except FileNotFoundError:
        st.error(
            "❌ **sp1.csv not found.** "
            "Please place `sp1.csv` in the same directory as `app.py` and restart."
        )
        st.stop()

    # ---- Sidebar form --------------------------------------------------
    inputs, submitted = render_sidebar()

    # ---- Tabs ----------------------------------------------------------
    tab_eda, tab_model, tab_predict = st.tabs(
        ["📊 EDA", "🤖 Model Metrics", "🔍 Prediction Result"]
    )

    with tab_eda:
        render_eda(raw_df)

    with tab_model:
        render_model_metrics(accuracy, report, model, feature_order)

    with tab_predict:
        st.header("🔍 Loan Approval Prediction")

        if not submitted:
            st.info("👈  Fill in the **Applicant Details** form in the sidebar and click **Predict**.")
        else:
            # Build encoded input and run inference
            input_df  = build_input_df(inputs, label_encoders, feature_order)
            prediction      = model.predict(input_df)[0]
            probabilities   = model.predict_proba(input_df)[0]
            approval_prob   = probabilities[1] * 100
            rejection_prob  = probabilities[0] * 100

            st.subheader("Applicant Summary")
            summary_df = pd.DataFrame(inputs.items(), columns=["Feature", "Value"])
            st.table(summary_df)

            st.subheader("Prediction Result")

            # Probability gauge (simple progress bars)
            col_a, col_b = st.columns(2)
            col_a.metric("Approval Probability",  f"{approval_prob:.1f}%")
            col_b.metric("Rejection Probability", f"{rejection_prob:.1f}%")
            st.progress(int(approval_prob))

            st.divider()

            # Final verdict
            if prediction == 1:
                st.success(
                    "✅ **Loan APPROVED**\n\n"
                    f"Based on the provided information, the model predicts this application "
                    f"will be **approved** with a confidence of **{approval_prob:.1f}%**."
                )
            else:
                st.error(
                    "❌ **Loan REJECTED**\n\n"
                    f"Based on the provided information, the model predicts this application "
                    f"will be **rejected** with a confidence of **{rejection_prob:.1f}%**."
                )

            # Show raw probability breakdown
            with st.expander("📈 Detailed Probability Breakdown"):
                prob_df = pd.DataFrame({
                    "Outcome":     ["Rejected", "Approved"],
                    "Probability": [f"{rejection_prob:.2f}%", f"{approval_prob:.2f}%"],
                })
                st.table(prob_df)


if __name__ == "__main__":
    main()
