import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt


st.set_page_config(page_title="Fraud Detection System", layout="wide")

st.title("Credit Card Fraud Detection System")
st.markdown("Upload transaction dataset to detect fraudulent transactions.")


FEATURES = [
'Time','V1','V2','V3','V4','V5','V6','V7','V8','V9','V10',
'V11','V12','V13','V14','V15','V16','V17','V18','V19','V20',
'V21','V22','V23','V24','V25','V26','V27','V28','Amount'
]

@st.cache_resource
def load_model():
    return joblib.load("xgb_pipeline.pkl")

model = load_model()

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])


if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data Preview")
    st.dataframe(data.head())

    try:
        data = data[FEATURES]

        predictions = model.predict(data)
        probabilities = model.predict_proba(data)[:, 1]

        result_df = data.copy()
        result_df["Fraud_Prediction"] = predictions
        result_df["Fraud_Probability"] = probabilities

        st.success("Prediction Completed Successfully")

        total_transactions = len(predictions)
        fraud_count = sum(predictions)
        non_fraud_count = total_transactions - fraud_count
        fraud_percentage = (fraud_count / total_transactions) * 100

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Transactions", total_transactions)
        col2.metric("Fraud Transactions", fraud_count)
        col3.metric("Fraud %", f"{fraud_percentage:.2f}%")

        st.subheader("Fraud vs Non-Fraud Distribution")


        fig, ax = plt.subplots(figsize=(4,3))
        ax.bar(["Non-Fraud", "Fraud"], [non_fraud_count, fraud_count])
        ax.set_yscale("log")
        ax.set_ylabel("Count (log scale)")
        st.pyplot(fig)


        st.subheader("Prediction Results")
        st.dataframe(result_df.head())

        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Predictions",
            csv,
            "fraud_predictions.csv",
            "text/csv"
        )

    except Exception as e:
     st.error(str(e))