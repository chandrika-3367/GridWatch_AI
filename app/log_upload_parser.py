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
from datetime import datetime
from chat_interface import extract_provider_and_location

# Setup
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Static KPIs
DEFAULT_KPIS = ["Energy_Consumed_kWh", "Voltage_V", "Current_A"]

# Convert timestamp fields inside log dictionaries into ISO format


def convert_timestamps(obj):
    """Convert timestamps in data structures."""
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, datetime):  # Handles standard Python datetime objects
        return obj.isoformat()
    elif isinstance(obj, str):
        try:
            # Detect if string is a timestamp and convert it
            return pd.to_datetime(obj).isoformat()
        except (ValueError, TypeError):
            return obj  # If conversion fails, return the original string
    elif isinstance(obj, dict):
        return {k: convert_timestamps(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_timestamps(i) for i in obj]
    return obj

# Previews the log content


def preview_log_content(content, label):
    st.text_area(f"{label} Preview", content, height=300)

# Extract the text from pdf


def extract_text_from_pdf(uploaded_file):
    try:
        with fitz.open(stream=uploaded_file.getvalue(), filetype="pdf") as doc:
            text = "\n".join(page.get_text("text") for page in doc)
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""


# Python parser
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

            record = None

            try:
                record = ast.literal_eval(clean_line.rstrip(','))
            except (SyntaxError, ValueError):
                try:
                    json_line = clean_line.replace("'", '"')
                    record = json.loads(json_line)
                except json.JSONDecodeError:
                    continue  # suppress and fallback to LLM in upload tab

            if isinstance(record, dict):
                records.append(record)

        except Exception:
            continue  # suppress line-level warnings

    return pd.DataFrame(records) if records else None

# Fallback LLM Parser


def fallback_llm_parse(text):
    try:
        if not groq_api_key:
            return None

        client = Groq(api_key=groq_api_key)

        bill_keywords = r"(account number|billing period|meter id|total usage|charges|bill date)"
        is_bill = re.search(bill_keywords, text, re.IGNORECASE) is not None

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

        json_text = None
        json_match = re.search(r'(\{.*\}|\[.*\])', llm_output, re.DOTALL)
        if json_match:
            json_text = json_match.group(1).replace("'", '"')

        parsed_json = None
        try:
            parsed_json = json.loads(json_text)
        except json.JSONDecodeError:
            try:
                json_text = re.search(
                    r'(\{.*\}|\[.*\])', json_text, re.DOTALL).group(1)
                parsed_json = json.loads(json_text)
            except Exception:
                return None

        if isinstance(parsed_json, list):
            return pd.DataFrame(parsed_json)
        elif isinstance(parsed_json, dict):
            return pd.DataFrame([parsed_json])
        else:
            return None

    except Exception:
        return None


# Summarize logs with AI

def summarize_logs_with_ai(df):
    if df is None or df.empty:
        st.warning("No log data available for summarization.")
        return

    log_records = df.head(10).to_dict(orient="records")

    formatted_logs = json.dumps(convert_timestamps(log_records), indent=2)

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
    ```
    {formatted_logs}
    ```
    """

    with st.spinner("\u2728 Generating AI-powered summary..."):
        try:
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-70b-8192"
            )
            ai_summary = chat_completion.choices[0].message.content.strip()

            if not ai_summary:
                st.warning(
                    "AI did not return a valid summary. Try again or check logs.")
                return

            st.success("AI-Powered Log Summary")
            st.write(ai_summary)

        except Exception as e:
            st.error(f"AI summarization failed: {str(e)}")

# Analyze Log Content


def analyze_log_content(df):
    if df is None or df.empty:
        st.warning("No data available for analysis.")
        return

    st.session_state["analyzed_df"] = df

    if "action" not in st.session_state:
        st.session_state["action"] = None

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        if st.button("Preview Data"):
            st.session_state["action"] = 'preview'
            st.rerun()

    with col2:
        if st.button("Analyze Data"):
            st.session_state["action"] = 'analyze'
            st.rerun()

    with col3:
        if st.button("Summarize"):
            st.session_state["action"] = 'summarize'
            st.rerun()

    with col4:
        if st.button("Ask GridWatch AI"):
            st.session_state['log_context_available'] = True
            st.session_state['active_tab'] = "Chat with GridWatch AI"
            st.rerun()

    with col5:
        if st.button("Visualize Metrics"):
            st.session_state['active_tab'] = "Visualize Metrics"
            st.rerun()

    # Handle actions
    if st.session_state["action"] == 'preview':
        preview_log_content(df.head().to_json(indent=2), "Log Data Preview")

    elif st.session_state["action"] == 'analyze':
        st.subheader("Meter Reading Data Analysis")

        selected_kpis = st.multiselect(
            "Select KPIs:",
            options=df.columns.tolist(),
            default=[kpi for kpi in DEFAULT_KPIS if kpi in df.columns]
        )

        if selected_kpis:
            st.dataframe(df[selected_kpis].describe())

        st.subheader("Missing Values in Data")
        st.dataframe(df.isnull().sum().rename("Missing Values"))

        if 'Meter_ID' in df.columns:
            st.subheader("Unique Meter IDs")
            st.write(df["Meter_ID"].unique().tolist())

    elif st.session_state["action"] == 'summarize':
        summarize_logs_with_ai(df)

# Actual upload tab logic


def log_upload_tab():
    st.subheader("Upload Logs for Analysis")
    uploaded_files = st.file_uploader(
        "Choose log files",
        type=["txt", "csv", "json", "log", "pdf", "docx"],
        accept_multiple_files=True
    )

    if not uploaded_files:
        return

    # Clear stale bill data when new files are uploaded
    st.session_state["uploaded_bills"] = []
    st.session_state["provider_names"] = []
    st.session_state["billing_locations"] = []

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
                    try:
                        data = [json.loads(line) for line in raw.strip().split(
                            '\n') if line.strip()]
                        df = pd.DataFrame(data)
                    except:
                        df = fallback_llm_parse(raw)

            elif file_type in ["txt", "log"]:
                content = uploaded_file.read().decode("utf-8")
                df = robust_parse_text_to_df(content)
                if df is None or df.empty:
                    df = fallback_llm_parse(content)

            elif file_type == "pdf":
                try:
                    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
                        pdf_text = "\n".join(page.get_text().strip()
                                             for page in doc)
                except:
                    pdf_text = ""

                if pdf_text:
                    df = robust_parse_text_to_df(pdf_text)
                    if df is None or df.empty:
                        df = fallback_llm_parse(pdf_text)

                # Always store current bills only (old ones are wiped above)
                st.session_state["uploaded_bills"].append({
                    "name": uploaded_file.name,
                    "text": pdf_text
                })
                bill_texts = [bill["text"]
                              for bill in st.session_state.get("uploaded_bills", [])]
                if bill_texts:
                    extract_provider_and_location(bill_texts)

            elif file_type == "docx":
                try:
                    docx_file = Document(uploaded_file)
                    paragraphs = [para.text.strip()
                                  for para in docx_file.paragraphs if para.text.strip()]
                    text = "\n".join(paragraphs)
                    df = robust_parse_text_to_df(text)
                    if df is None or df.empty:
                        df = fallback_llm_parse(text)
                except:
                    pass

        except:
            try:
                uploaded_file.seek(0)
                content = uploaded_file.read().decode("utf-8")
                df = fallback_llm_parse(content)
            except:
                pass

        if df is not None and not df.empty:
            all_data.append(df)

    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        st.session_state["analyzed_df"] = combined_df
        st.session_state["files_ready"] = True
        analyze_log_content(combined_df)
    elif st.session_state.get("uploaded_bills"):
        st.session_state["files_ready"] = True
    else:
        st.error("No valid data found in uploaded files.")
