import os
import time
import requests
import sqlite3
import numpy as np
import pandas as pd
import streamlit as st

#  Page Configuration
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide"
)

#  Dynamic API URL Resolution (Supports Local & Docker Environments)
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")
DB_PATH = os.getenv("DB_PATH", "fraud_predictions.db")

st.title(" Real-Time Fraud Detection Dashboard")
st.caption("Production MLOps Monitoring & Live Inference Portal")

#  Database Utility
def fetch_logs():
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(DB_PATH)
        query = "SELECT * FROM prediction_logs ORDER BY id DESC LIMIT 50"
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Database connection error: {str(e)}")
        return pd.DataFrame()

#  Interface Tabs
tab1, tab2 = st.tabs([" Live Audit Logs", " Transaction Simulator"])

# --- TAB 1: LIVE AUDIT LOGS ---
with tab1:
    st.subheader("System Metrics & Real-time Logs")

    col_btn, _ = st.columns([1, 5])
    with col_btn:
        if st.button(" Refresh Data", use_container_width=True):
            st.rerun()

    df_logs = fetch_logs()

    if df_logs.empty:
        st.info("No prediction records found yet. Run simulations in Tab 2 or invoke the API.")
    else:
        # Key Performance Indicators (KPIs)
        total_logs = len(df_logs)
        fraud_count = int(df_logs["is_fraud"].sum())
        avg_latency = df_logs["prediction_time_ms"].mean()
        avg_risk = df_logs["fraud_probability"].mean()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Requests", total_logs)
        m2.metric("Fraud Flagged", fraud_count, delta_color="inverse")
        m3.metric("Avg Latency", f"{avg_latency:.2f} ms")
        m4.metric("Avg Risk Score", f"{avg_risk:.4f}")

        st.divider()

        # Display Data Table with Formatting
        st.dataframe(
            df_logs,
            column_config={
                "id": st.column_config.NumberColumn("ID"),
                "timestamp": st.column_config.DatetimeColumn("Timestamp"),
                "transaction_amount": st.column_config.NumberColumn("Amount ($)", format="$%.2f"),
                "fraud_probability": st.column_config.ProgressColumn(
                    "Fraud Probability",
                    format="%.4f",
                    min_value=0.0,
                    max_value=1.0,
                ),
                "is_fraud": st.column_config.CheckboxColumn("Is Fraud?"),
                "prediction_time_ms": st.column_config.NumberColumn("Latency (ms)", format="%.2f ms"),
            },
            use_container_width=True,
            hide_index=True
        )

# --- TAB 2: TRANSACTION SIMULATOR ---
with tab2:
    st.subheader("Manual Inference Tester")
    st.write("Simulate single credit card transactions and evaluate real-time ONNX inference risk score.")

    col1, col2 = st.columns(2)
    with col1:
        amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=120.50, step=10.0)
    with col2:
        time_seconds = st.number_input("Time Elapsed (Seconds)", min_value=0.0, value=3600.0, step=100.0)

    is_fraud_scenario = st.checkbox("Inject Fraud-like Anomaly Features (V1..V28)")

    if st.button(" Run Fraud Assessment", type="primary"):
        # Generate 30 features
        if is_fraud_scenario:
            features = np.random.normal(loc=-3.0, scale=2.0, size=30).astype(np.float32).tolist()
        else:
            features = np.random.normal(loc=0.0, scale=0.5, size=30).astype(np.float32).tolist()

        features[0] = float(time_seconds)
        features[29] = float(amount)

        payload = {"features": features}

        try:
            start_t = time.time()
            response = requests.post(API_URL, json=payload, timeout=5)
            client_latency = round((time.time() - start_t) * 1000, 2)

            if response.status_code == 200:
                data = response.json()
                is_fraud = data.get("is_fraud", False)
                prob = data.get("fraud_probability", 0.0)
                status = data.get("status", "UNKNOWN")
                server_latency = data.get("latency_ms", 0.0)

                st.divider()
                if is_fraud:
                    st.error(f" **HIGH RISK DETECTED** | Status: **{status}**")
                else:
                    st.success(f" **TRANSACTION APPROVED** | Status: **{status}**")

                res_col1, res_col2, res_col3 = st.columns(3)
                res_col1.metric("Fraud Risk Score", f"{prob * 100:.2f}%")
                res_col2.metric("Server Latency", f"{server_latency:.2f} ms")
                res_col3.metric("Network Latency", f"{client_latency:.2f} ms")

            else:
                st.error(f"API Error [{response.status_code}]: {response.text}")

        except requests.exceptions.RequestException as err:
            st.error(f"Failed to connect to API service at `{API_URL}`: {str(err)}")
