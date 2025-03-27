import re
import streamlit as st
import json
import pandas as pd
import fitz  # PyMuPDF for PDFs
from docx import Document
import ast
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

DEFAULT_KPIS = ["Energy_Consumed_kWh", "Voltage_V", "Current_A"]


def preview_log_content(content, label):
    st.text_area(f"{label} Preview", content, height=300)


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


def summarize_logs_with_ai(df):
    log_records = df.head(10).to_dict(orient='records')
    if not groq_api_key:
        st.error("Groq API key not found. Please set GROQ_API_KEY in your .env file.")
        return

    client = Groq(api_key=groq_api_key)

    prompt = f"""
    Summarize these smart-meter logs:
    1. Consumption trends
    2. Anomalies or suspicious patterns
    3. Recommended actions

    Logs:
    {log_records}
    """

    with st.spinner("\u2728 Generating AI-powered summary..."):
        try:
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-70b-8192"
            )
            ai_summary = chat_completion.choices[0].message.content.strip()
            st.success("AI-Powered Log Summary")
            st.write(ai_summary)
        except Exception as e:
            st.error(f"AI summarization failed: {str(e)}")


def analyze_log_content(df):
    if df is None or df.empty:
        st.warning("No data available for analysis.")
        return

    st.session_state["analyzed_df"] = df

    if 'action' not in st.session_state:
        st.session_state.action = None

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        if st.button("Preview"):
            st.session_state.action = 'preview'

    with col2:
        if st.button("Analyze"):
            st.session_state.action = 'analyze'

    with col3:
        if st.button("Summarize"):
            st.session_state.action = 'summarize'

    with col4:
        if st.button("Ask GridWatchAI"):
            st.session_state['log_context_available'] = True
            st.session_state['active_tab'] = "Chat with GridWatch AI"
            st.rerun()

    with col5:
        if st.button("Visualize"):
            st.session_state['active_tab'] = "Visualize Metrics"
            st.rerun()

    if st.session_state.action == 'preview':
        preview_log_content(df.head().to_json(indent=2), "Log Data Preview")

    elif st.session_state.action == 'analyze':
        st.subheader("Meter Reading Data Analysis")
        selected_kpis = st.multiselect(
            "Select KPIs:",
            options=df.columns.tolist(),
            default=[kpi for kpi in DEFAULT_KPIS if kpi in df.columns]
        )
        if selected_kpis:
            st.dataframe(df[selected_kpis].describe())
        st.dataframe(df.isnull().sum().rename("Missing Values"))
        if 'Meter_ID' in df.columns:
            st.dataframe(df['Meter_ID'].unique(),
                         column_config={"value": "Meter_ID"})

    elif st.session_state.action == 'summarize':
        summarize_logs_with_ai(df)


def log_upload_tab():
    st.subheader("Upload Logs for Analysis")
    uploaded_files = st.file_uploader("Choose log files", type=[
                                      "txt", "csv", "json", "log", "pdf", "docx"], accept_multiple_files=True)

    if not uploaded_files:
        return

    all_data = []

    for uploaded_file in uploaded_files:
        file_type = uploaded_file.name.split(".")[-1].lower()

        df = None
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
            elif file_type == "pdf":
                with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
                    text = "\n".join(page.get_text().strip() for page in doc)
                df = robust_parse_text_to_df(text)
                if df is None or df.empty:
                    df = fallback_llm_parse(text)
            elif file_type == "docx":
                docx_file = Document(uploaded_file)
                paragraphs = [para.text.strip()
                              for para in docx_file.paragraphs if para.text.strip()]
                text = "\n".join(paragraphs)
                df = robust_parse_text_to_df(text)
                if df is None or df.empty:
                    df = fallback_llm_parse(text)
            else:
                st.error("Unsupported file type.")
                continue

            if df is not None and not df.empty:
                all_data.append(df)

        except Exception as e:
            st.warning(
                f"Parsing failed for {uploaded_file.name}. Attempting LLM fallback... Error: {str(e)}")
            try:
                uploaded_file.seek(0)
                content = uploaded_file.read().decode("utf-8")
                df = fallback_llm_parse(content)
                if df is not None and not df.empty:
                    all_data.append(df)
                else:
                    st.error(f"LLM parsing failed for {uploaded_file.name}.")
            except Exception as ee:
                st.error(f"Final parsing error: {str(ee)}")

    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        analyze_log_content(combined_df)
    else:
        st.error("No valid data found in uploaded files.")
