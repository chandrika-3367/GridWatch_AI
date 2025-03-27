import streamlit as st
import pandas as pd
import plotly.express as px
import ast
import fitz
from docx import Document
import json
import requests
import uuid
import re
from dotenv import load_dotenv
from groq import Groq
import os

# Load Groq API key
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Helper to parse logs manually with generalized logic


def robust_parse_text_to_df(text):
    records = []
    lines = text.strip().splitlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            clean_line = line.replace("\u2018", "'").replace(
                "\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
            clean_line = clean_line.replace(
                '\u00A0', ' ').replace('“', '"').replace('”', '"')
            try:
                record = ast.literal_eval(clean_line.rstrip(','))
            except:
                json_line = clean_line.replace("'", '"')
                record = json.loads(json_line)
            if isinstance(record, dict):
                records.append(record)
        except Exception:
            continue
    return pd.DataFrame(records) if records else None

# Fallback LLM parsing via Groq API with bill detection


def fallback_llm_parse(text):
    try:
        if not groq_api_key:
            st.error(
                "Groq API key not found. Please set GROQ_API_KEY in your .env file.")
            return None

        client = Groq(api_key=groq_api_key)

        bill_keywords = ["account number", "billing period",
                         "meter id", "total usage", "charges", "bill date"]
        is_bill = any(kw in text.lower() for kw in bill_keywords)

        if is_bill:
            prompt = f"""
            Extract the following details from the utility bill text:
            - Account Number
            - Meter ID
            - Billing Period or Dates
            - Meter Readings (initial/final)
            - Total Energy Consumption (kWh)
            - Total Charges or Cost

            Format the output as valid JSON with double quotes. Example:
            {{
            "Account_Number": "123456",
            "Meter_ID": "MTR001",
            "Billing_Period": "2025-02-01 to 2025-02-28",
            "Usage_kWh": 234.56,
            "Total_Charges": "$54.23"
            }}

            Bill Text:
            ```
            {text[:2000]}
            ```
            """
        else:
            prompt = f"""
            You are an expert data parser. Extract structured tabular data from the following smart meter logs. 
            Return the output as a JSON array of objects with the following fields: 
            Timestamp (ISO format), Meter_ID (string), Energy_Consumed_kWh (number), Voltage_V (number), Current_A (number), Note (optional string). 
            Ensure numeric fields are numbers (not strings), and Timestamp is formatted as YYYY-MM-DD HH:MM:SS. Only return valid entries in JSON format. No explanation needed.

            Logs:
            ```
            {text[:2000]}
            ```
            """

        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-70b-8192"
        )

        llm_output = chat_completion.choices[0].message.content.strip()

        json_match = re.search(r'\{.*\}|\[.*\]', llm_output, re.DOTALL)
        if json_match:
            json_text = json_match.group().replace("'", '"')
            parsed_json = json.loads(json_text)

            if isinstance(parsed_json, list):
                return pd.DataFrame(parsed_json)
            elif isinstance(parsed_json, dict):
                return pd.DataFrame([parsed_json])
            else:
                st.error("Unexpected response format from LLM.")
                return None
        else:
            st.error("No JSON-like content found in LLM response.")
            return None

    except Exception as e:
        st.error(f"LLM fallback failed: {str(e)}")
        return None

# Visualize tab with redirect support


def visualize_tab():
    st.subheader("\U0001F4CA Metrics Visualization")

    df = st.session_state.get('analyzed_df')
    upload_type = st.session_state.get('upload_type')

    # If data exists (redirected), skip upload
    if df is not None and not df.empty:
        st.info("Visualizing previously uploaded data.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Upload Utility Bill"):
                st.session_state['upload_type'] = "Utility Bill"
        with col2:
            if st.button("Upload Smart Meter Log"):
                st.session_state['upload_type'] = "Smart Meter Log"

        upload_type = st.session_state.get('upload_type')

        if not upload_type:
            st.info("Please select the data type before uploading.")
            return

        uploaded_files = st.file_uploader("Upload file(s)", type=[
                                          "txt", "csv", "json", "log", "pdf", "docx"], accept_multiple_files=True)

        if not uploaded_files:
            return

        all_data = []
        for uploaded_file in uploaded_files:
            file_type = uploaded_file.name.split(".")[-1].lower()
            try:
                if file_type == "csv":
                    df = pd.read_csv(uploaded_file)
                elif file_type == "json":
                    raw = uploaded_file.read().decode("utf-8")
                    try:
                        df = pd.read_json(raw)
                    except:
                        data = [json.loads(line) for line in raw.strip().split(
                            '\n') if line.strip()]
                        df = pd.DataFrame(data)
                elif file_type in ["txt", "log"]:
                    content = uploaded_file.read().decode("utf-8")
                    df = robust_parse_text_to_df(content)
                    if df is None or df.empty:
                        df = fallback_llm_parse(content)
                elif file_type in ["pdf", "docx"]:
                    if file_type == "pdf":
                        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
                            text = "\n".join(page.get_text().strip()
                                             for page in doc)
                    else:
                        docx_file = Document(uploaded_file)
                        text = "\n".join(
                            para.text.strip() for para in docx_file.paragraphs if para.text.strip())
                    df = robust_parse_text_to_df(text)
                    if df is None or df.empty:
                        df = fallback_llm_parse(text)
                else:
                    st.warning(f"Unsupported file type: {file_type}")
                    continue

                if df is not None and not df.empty:
                    df['Source_File'] = uploaded_file.name
                    all_data.append(df)

            except Exception as e:
                st.warning(f"Failed to parse {uploaded_file.name}: {str(e)}")

        if not all_data:
            st.error("No valid data found across uploaded files.")
            return

        df = pd.concat(all_data, ignore_index=True)
        st.session_state['analyzed_df'] = df.copy()

    # Proceed to visualization
    available_kpis = [
        col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    if not available_kpis:
        st.warning("No numeric KPIs available for visualization.")
        return

    st.markdown("### Select KPI and Meter")
    selected_kpi = st.selectbox(
        "Select KPI to visualize:", options=available_kpis)
    meter_ids = df['Meter_ID'].unique().tolist(
    ) if 'Meter_ID' in df.columns else []
    selected_meters = st.multiselect(
        "Filter by Meter_ID:", options=meter_ids, default=meter_ids)

    filtered_df = df[df['Meter_ID'].isin(selected_meters)] if meter_ids else df

    kpi_min = float(filtered_df[selected_kpi].min())
    kpi_max = float(filtered_df[selected_kpi].max())

    if kpi_min == kpi_max:
        st.info(
            f"All values for {selected_kpi} are {kpi_min}. Threshold slider disabled.")
        threshold = kpi_min
    else:
        threshold = st.slider(f"Set threshold for {selected_kpi}", kpi_min, kpi_max, float(
            filtered_df[selected_kpi].mean()))

    filtered_df['Anomaly'] = filtered_df[selected_kpi] > threshold

    if 'Timestamp' in filtered_df.columns:
        try:
            filtered_df['Timestamp'] = pd.to_datetime(
                filtered_df['Timestamp'], errors='coerce')
            filtered_df = filtered_df.dropna(subset=['Timestamp'])
            filtered_df = filtered_df.sort_values('Timestamp')
            fig_line = px.line(filtered_df, x='Timestamp', y=selected_kpi, color='Anomaly',
                               title=f'{selected_kpi} Over Time (Anomaly Highlighted)', line_shape='spline', markers=True, color_discrete_map={True: 'red', False: 'blue'})
            st.plotly_chart(fig_line, use_container_width=True)
        except Exception as e:
            st.warning(f"Timestamp parsing failed: {str(e)}")

    st.markdown(f"### Box Plot for {selected_kpi}")
    fig_box = px.box(filtered_df, x='Meter_ID', y=selected_kpi,
                     title=f"Box Plot of {selected_kpi} by Meter")
    st.plotly_chart(fig_box, use_container_width=True)

    if upload_type == "Utility Bill":
        if 'Billing_Period' in filtered_df.columns:
            filtered_df['Billing_Period'] = filtered_df['Billing_Period'].astype(
                str)
            fig_trend = px.line(filtered_df, x='Billing_Period', y=selected_kpi, color='Meter_ID',
                                title=f"{selected_kpi} Trend Across Billing Periods", markers=True)
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("Billing_Period column not found for trend visualization.")
    else:
        st.markdown(f"### Histogram of {selected_kpi}")
        fig_hist = px.histogram(filtered_df, x=selected_kpi, nbins=30,
                                title=f"Distribution of {selected_kpi}", color_discrete_sequence=['steelblue'])
        st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown(f"### Average {selected_kpi} per Meter")
    try:
        avg_per_meter = filtered_df.groupby(
            'Meter_ID')[selected_kpi].mean().reset_index()
        fig_bar = px.bar(avg_per_meter, x='Meter_ID', y=selected_kpi,
                         title=f"Average {selected_kpi} by Meter", color='Meter_ID')
        st.plotly_chart(fig_bar, use_container_width=True)
    except Exception as e:
        st.warning(f"Bar chart rendering failed: {str(e)}")

    anomalies = filtered_df[filtered_df['Anomaly']]
    st.markdown("### Anomalies Detected")
    if not anomalies.empty:
        st.dataframe(anomalies, use_container_width=True)
    else:
        st.success("No anomalies detected above threshold.")

    if 'Meter_ID' in filtered_df.columns:
        st.markdown("### Fraud Distribution Map (Mock Data)")
        meter_geo = pd.DataFrame({
            'Meter_ID': filtered_df['Meter_ID'].unique(),
            'lat': [31.9686 + i*0.1 for i in range(len(filtered_df['Meter_ID'].unique()))],
            'lon': [-99.9018 + i*0.1 for i in range(len(filtered_df['Meter_ID'].unique()))]
        })
        merged_geo = pd.merge(filtered_df, meter_geo,
                              on='Meter_ID', how='left')
        fig_map = px.scatter_mapbox(merged_geo, lat='lat', lon='lon', color=selected_kpi, size=selected_kpi, hover_name='Meter_ID',
                                    zoom=5, mapbox_style='carto-positron', title="Fraud Distribution Map (Mock Coordinates)")
        st.plotly_chart(fig_map, use_container_width=True)

    st.markdown("---")
    csv = df.to_csv(index=False)
    st.download_button(label="Download CSV Report", data=csv,
                       file_name="GridWatch_Report.csv", mime="text/csv")

    if st.button("\U0001F6A8 Report Fraud / Generate Ticket"):
        if not anomalies.empty:
            sample_records = anomalies.head(1).copy()
            sample_records['Timestamp'] = sample_records['Timestamp'].astype(
                str)

            ticket_id = str(uuid.uuid4())[:8].upper()

            ticket = {
                "ticket_id": ticket_id,
                "issue": f"Anomalies detected in {selected_kpi} above threshold {threshold}",
                "affected_meters": anomalies['Meter_ID'].unique().tolist(),
                "sample_timestamp": sample_records['Timestamp'].iloc[0],
                "sample_note": sample_records['Note'].iloc[0] if 'Note' in sample_records.columns else "N/A"
            }

            try:
                payload = json.dumps(ticket)
                response = requests.post("https://67ddd152471aaaa742829e2f.mockapi.io/fraudtickets",
                                         data=payload, headers={"Content-Type": "application/json"})
                if response.status_code == 201:
                    st.success(
                        f"\u2705 Ticket #{ticket_id} created successfully! Your fraud report has been logged.")
                    st.json(ticket)
                    ack_text = f"""
GridWatch AI - Fraud Report Acknowledgment
Ticket ID: {ticket_id}
Issue: {ticket['issue']}
Affected Meters: {', '.join(ticket['affected_meters'])}
Timestamp: {ticket['sample_timestamp']}
Note: {ticket['sample_note']}
Status: Logged
"""
                    st.download_button("\U0001F4C4 Download Acknowledgment", data=ack_text,
                                       file_name=f"FraudTicket_{ticket_id}.txt", mime="text/plain")
                else:
                    st.warning(
                        f"Mock API responded with status {response.status_code}: {response.text}")
            except Exception as e:
                st.error(f"Mock API call failed: {str(e)}")
        else:
            st.info("No anomalies to report.")

    if st.button("\U0001F916 Detect Fraud Using AI"):
        st.session_state['active_tab'] = "AI-Based Detection"
        st.rerun()
