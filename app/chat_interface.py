import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import json
import uuid
import requests
import re
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# Setup
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm_rag = ChatOpenAI(
    model="llama3-70b-8192",
    base_url="https://api.groq.com/openai/v1",
    api_key=groq_api_key,
    temperature=0
)


def chat_tab():
    st.markdown("""
    Ask questions about energy theft, smart meter patterns, or recommendations from your utility bill.
    """)

    logs_df = st.session_state.get("analyzed_df")
    bill_data = st.session_state.get("bill_context")

    if logs_df is not None and not logs_df.empty and not bill_data:
        from groq import Groq
        client = Groq(api_key=groq_api_key)
        sample = logs_df.head(1).to_dict(orient="records")[0]
        prompt = f"The following data is extracted from a utility bill:\n{json.dumps(sample, indent=2)}\n..."
        try:
            result = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[{"role": "user", "content": prompt}]
            ).choices[0].message.content.strip()
            json_match = re.search(r'{.*}', result, re.DOTALL)
            if json_match:
                try:
                    st.session_state.bill_context = json.loads(
                        json_match.group())
                except json.JSONDecodeError:
                    st.session_state.bill_context = json.loads(
                        json_match.group().replace("'", '"'))
        except:
            pass
        bill_data = st.session_state.bill_context

    prompt_prefill = "e.g., What anomalies are shown in the uploaded logs? | What is the cancellation fee for my provider? | What are the signs of meter tampering?"
    query = st.text_input("Ask GridWatch AI:", placeholder=prompt_prefill)

    use_bill_context = bill_data is not None and any(
        keyword in query.lower() for keyword in [
            "bill", "provider", "usage", "plan", "cancellation", "reliant", "txu", "green mountain", "constellation", "shell energy"]
    )

    if st.button("Get Insights") and query:
        try:
            if use_bill_context:
                provider = bill_data.get("provider")
                location = bill_data.get("location")
                terms = bill_data.get("terms_conditions")
                billing_summary = bill_data.get("billing_summary")
                bill_prompt = f"""
                A user uploaded a utility bill. Below are the extracted details:
                Provider: {provider}
                Location: {location}
                Terms: {terms}
                Billing Summary: {billing_summary}
                Question: {query}
                """
                response = llm_rag.invoke(bill_prompt)

            elif logs_df is not None and not logs_df.empty:
                logs_preview = logs_df.head(10).to_dict(orient="records")
                log_prompt = f"""
                The following smart meter logs have been uploaded:
                {json.dumps(logs_preview, indent=2)}
                Question: {query}
                """
                response = llm_rag.invoke(log_prompt)

            else:
                rag_qa = RetrievalQA.from_chain_type(
                    llm=llm_rag, retriever=Chroma(persist_directory=os.path.abspath("../db"), embedding_function=embedding_function).as_retriever())
                response = rag_qa.run(query)

            st.success("Response:")
            st.write(response)

            st.download_button(
                "📄 Download Response as Report",
                data=f"GridWatch AI Chat Response\n\nQuery: {query}\n\nResponse:\n{response}",
                file_name="GridWatchAI_Chat_Report.txt",
                mime="text/plain"
            )

        except Exception as e:
            st.error(f"Error during chat response: {str(e)}")

    if logs_df is not None and not logs_df.empty:
        if st.button("📊 Visualize Logs"):
            st.session_state["active_tab"] = "Visualize Metrics"
            st.session_state["upload_type"] = "Utility Bill" if bill_data else "Smart Meter Log"
            st.rerun()
