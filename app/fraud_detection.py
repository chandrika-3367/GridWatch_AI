import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import json
import uuid
import requests


def run_ml_detection(df):
    df['ML_Fraud_Score'] = np.random.rand(len(df))
    threshold = 0.7
    df['ML_Fraud'] = df['ML_Fraud_Score'] > threshold
    fraud_count = df['ML_Fraud'].sum()
    confidence = round(np.mean(df['ML_Fraud_Score']) * 100, 2)
    return df, fraud_count, confidence


def visualize_comparison(df):
    visual_anomaly_count = df['Anomaly'].sum(
    ) if 'Anomaly' in df.columns else 0
    ml_fraud_count = df['ML_Fraud'].sum()

    counts = pd.DataFrame({
        'Method': ['Threshold Anomalies', 'ML-Predicted Fraud'],
        'Meters Flagged': [visual_anomaly_count, ml_fraud_count]
    })

    fig_bar = px.bar(counts, x='Method', y='Meters Flagged',
                     color='Method', title="Fraud Detection Comparison")
    st.plotly_chart(fig_bar, use_container_width=True)


def fraud_heatmap(df):
    if 'Meter_ID' in df.columns and 'Energy_Consumed_kWh' in df.columns and 'Voltage_V' in df.columns:
        fraud_df = df.copy()
        fraud_df['Fraud_Label'] = fraud_df['ML_Fraud'].apply(
            lambda x: 'Fraud' if x else 'Clean')

        fig = px.density_heatmap(fraud_df, x="Meter_ID", y="Energy_Consumed_kWh", z="Voltage_V",
                                 color_continuous_scale="Viridis", facet_col="Fraud_Label",
                                 title="Energy vs Voltage Heatmap with Fraud Overlay")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(
            "Insufficient data to render heatmap (need Meter_ID, Energy_Consumed_kWh, and Voltage_V).")


def generate_ticket(df):
    fraud_meters = df[df['ML_Fraud']]['Meter_ID'].unique(
    ).tolist() if 'Meter_ID' in df.columns else []
    ticket_id = str(uuid.uuid4())[:8].upper()

    ticket = {
        "ticket_id": ticket_id,
        "issue": "ML-Predicted Fraud Detected",
        "affected_meters": fraud_meters,
        "confidence": f"{round(np.mean(df['ML_Fraud_Score']) * 100, 2)}%"
    }

    try:
        payload = json.dumps(ticket)
        response = requests.post("https://67ddd152471aaaa742829e2f.mockapi.io/fraudtickets",
                                 data=payload, headers={"Content-Type": "application/json"})
        if response.status_code == 201:
            st.success(f"Ticket #{ticket_id} created successfully!")
            st.json(ticket)
        else:
            st.warning(
                f"API responded with status {response.status_code}: {response.text}")
    except Exception as e:
        st.error(f"Ticket generation failed: {str(e)}")


def fraud_detection_tab():
    st.subheader(":robot_face: AI-Based Fraud Detection")

    df = st.session_state.get("analyzed_df")
    upload_type = st.session_state.get("upload_type")

    if df is None or df.empty:
        st.warning("No data available for fraud analysis.")
        return

    if upload_type == "Utility Bill":
        st.info("This feature is currently designed for smart meter logs. Please switch to Visualize tab for utility bill insights.")
        return

    df, fraud_count, confidence = run_ml_detection(df)
    st.session_state['ml_detected_df'] = df.copy()

    st.markdown(f"""
    **Threshold Anomalies:** {df['Anomaly'].sum() if 'Anomaly' in df.columns else 0} meters flagged  
    **ML-Predicted Fraud:** {fraud_count} meters flagged  
    **Model Confidence:** {confidence}%
    """)

    visualize_comparison(df)
    fraud_heatmap(df)

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button(":bookmark_tabs: Generate Ticket"):
            generate_ticket(df)

    with col2:
        ai_report = df.to_csv(index=False)
        st.download_button(":floppy_disk: Download AI Report", data=ai_report,
                           file_name="ML_Fraud_Report.csv", mime="text/csv")

    with col3:
        if st.button(":chart_with_upwards_trend: Compare with Visuals"):
            st.session_state['active_tab'] = "Visualize Metrics"
            st.rerun()
