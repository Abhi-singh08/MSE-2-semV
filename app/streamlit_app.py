import sys
from pathlib import Path

# Add project root to Python import path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import requests
import streamlit as st
import pandas as pd
import plotly.express as px

from src.config import API_URL, FEATURE_COLUMNS

st.set_page_config(page_title="Predictive Maintenance", page_icon="⚙️", layout="wide")
st.title("⚙️ AI-Powered Predictive Maintenance")
st.caption("Machine failure prediction · Remaining Useful Life · Anomaly detection")

with st.sidebar:
    st.subheader("System")
    api_url = st.text_input("FastAPI URL", API_URL)
    if st.button("Check API health", use_container_width=True):
        try:
            r = requests.get(f"{api_url}/health", timeout=10)
            st.json(r.json())
        except requests.RequestException as e:
            st.error(str(e))
    st.divider()
    st.subheader("Model lifecycle")
    if st.button("Run controlled retraining", type="primary", use_container_width=True):
        with st.spinner("Training and evaluating new models..."):
            try:
                r = requests.post(f"{api_url}/retrain", timeout=1900)
                if r.ok:
                    st.success("Retraining completed; API loaded the registered models.")
                    st.json(r.json())
                else:
                    st.error(r.text)
            except requests.RequestException as e:
                st.error(f"Retraining request failed: {e}")

st.subheader("Sensor readings")
st.info("Enter a sensor row. For the NASA dataset, use the 24 input features: cycle, 3 operating settings and 21 sensors.")
defaults = {
    "cycle": 100.0, "setting_1": 0.0, "setting_2": 0.0, "setting_3": 0.0,
}
defaults.update({f"s{i}": 100.0 + i * 2 for i in range(1, 22)})

with st.form("prediction_form"):
    cols = st.columns(4)
    values = {}
    for idx, name in enumerate(FEATURE_COLUMNS):
        with cols[idx % 4]:
            values[name] = st.number_input(name, value=float(defaults[name]), format="%.4f")
    submitted = st.form_submit_button("Predict equipment health", type="primary")

if submitted:
    try:
        response = requests.post(f"{api_url}/predict", json=values, timeout=30)
        if response.ok:
            result = response.json()
            c1, c2, c3 = st.columns(3)
            c1.metric("Failure within 30 cycles", "YES" if result["failure_soon"] else "NO")
            c2.metric("Estimated RUL", f'{result["estimated_rul_cycles"]} cycles')
            c3.metric("Anomaly", "Detected" if result["is_anomaly"] else "Not detected" if result["is_anomaly"] is not None else "Unavailable")
            if result["risk"] == "HIGH":
                st.error("High predicted failure risk. Maintenance inspection is recommended.")
            else:
                st.success("Low predicted failure risk.")
            st.json(result)
        else:
            st.error(response.text)
    except requests.RequestException as e:
        st.error(f"Could not contact prediction API: {e}")

st.divider()
st.subheader("About this starter")
st.write("The initial model uses Random Forest for failure classification and RUL regression, plus Isolation Forest for anomaly detection. Predictions are decision-support estimates, not a substitute for industrial safety systems.")
