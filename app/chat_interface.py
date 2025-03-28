import re
import streamlit as st
import pandas as pd
import json
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from groq import Groq
import fitz  # PyMuPDF for PDF extraction

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

qa_chain = RetrievalQA.from_chain_type(
    llm=llm_rag,
    retriever=Chroma(
        persist_directory=os.path.abspath("../db"),
        embedding_function=embedding_function
    ).as_retriever()
)

client = Groq(api_key=groq_api_key)


def convert_timestamps(obj):
    """Convert timestamps in data structures."""
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: convert_timestamps(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_timestamps(i) for i in obj]
    return obj


def process_uploaded_bill(uploaded_file):
    """Store uploaded bill in session state and switch to chat tab."""
    if uploaded_file is not None:
        st.session_state.uploaded_bill = uploaded_file
        st.session_state["active_tab"] = "Chat with GridWatch AI"
        st.rerun()


def extract_provider_and_location(pdf_text):
    """
    Extracts the service provider name and billing location from the PDF text.
    Uses regex-based detection for common patterns found in utility bills.
    """
    provider_patterns = [
        r"(?:Service Provider|Company Name|Energy Provider|Utility Provider|Billing from|Account managed by):?\s*(.+)",
        r"^(.*?Energy|.*?Electric|.*?Power|.*?Utilities|.*?Gas|.*?Service)\b",
        r"\b(Reliant Energy|TXU Energy|Green Mountain Energy|Dominion Energy|Shell Energy|Constellation Energy)\b"
    ]

    location_patterns = [
        r"(?:Service Address|Billing Address|Customer Address|Location):?\s*([\w\s,]+ \b[A-Z]{2}\b \d{5}(-\d{4})?)",
        r"(\b[A-Z][a-z]+(?: [A-Z][a-z]+)*,? [A-Z]{2} \d{5}(-\d{4})?)"
    ]

    provider_name = None
    billing_location = None

    for pattern in provider_patterns:
        match = re.search(pattern, pdf_text, re.IGNORECASE)
        if match:
            provider_name = match.group(1).strip()
            break  # Stop at the first match

    for pattern in location_patterns:
        match = re.search(pattern, pdf_text, re.IGNORECASE)
        if match:
            billing_location = match.group(1).strip()
            break

    # Store extracted values in session state
    st.session_state["provider_name"] = provider_name or "Not Found"
    st.session_state["billing_location"] = billing_location or "Not Found"


def chat_tab():
    """GridWatch AI Chat Tab"""
    st.markdown("""
    **Ask questions about energy theft, smart meter patterns, or recommendations from your utility bill.**

    _Example Queries:_  
    - *What anomalies were detected in the uploaded logs?*  
    - *What are common fraud indicators in smart meter data?*  
    - *How does my provider compare to other companies in terms of pricing?*  
    """)

    if "uploaded_bill" not in st.session_state:
        st.session_state.uploaded_bill = None
    if "analyzed_df" not in st.session_state:
        st.session_state.analyzed_df = pd.DataFrame()

    logs_df = st.session_state.analyzed_df
    uploaded_bill = st.session_state.uploaded_bill
    query = st.text_input(
        "Ask GridWatch AI:", placeholder="e.g., What are the best energy plans available in my area?")

    use_log_context = logs_df is not None and not logs_df.empty and "log" in query.lower()
    use_bill_context = uploaded_bill is not None and any(keyword in query.lower() for keyword in [
        "bill", "provider", "usage", "plan", "cancellation", "reliant", "txu", "green mountain", "constellation",
        "shell energy", "compare", "cheaper options"
    ])

    if st.button("Get Insights") and query:
        try:
            response = "Analyzing your query, please wait..."

            if use_bill_context:
                with fitz.open(stream=uploaded_bill.getvalue(), filetype="pdf") as doc:
                    pdf_text = "\n".join(page.get_text("text") for page in doc)

                if st.checkbox("Show Debug - Extracted Bill Text"):
                    st.text_area("Extracted Bill Text:",
                                 pdf_text[:1000], height=200)

                # Extract Provider Name & Location
                extract_provider_and_location(pdf_text)

                bill_prompt = f"""
                ### **Persona:**
                You are an **AI energy analyst** with expertise in **utility bill interpretation, rate comparisons, and consumer energy guidance**.

                ### **Task:**
                Analyze the uploaded **utility bill** for **{st.session_state['provider_name']}** in **{st.session_state['billing_location']}**, extract relevant details, and answer the user's query.

                **User Query:** {query}

                - **Extract provider details:** {st.session_state['provider_name']}
                - **Extract billing location:** {st.session_state['billing_location']}
                - **Compare this provider’s rates & services against competitors in the same location.**
                - **Provide energy-saving recommendations tailored to the user’s bill details.**
                - **If the provider is recognized, include their official website for further details.**

                ### **Provider Comparison & Recommendations**
                If the user is looking for alternative energy providers, suggest reputable competitors in their region. Provide **direct website links** for each, ensuring they can easily explore alternative options.

                **Known Provider Websites:**
                - **Reliant Energy** → [https://www.reliant.com](https://www.reliant.com)
                - **TXU Energy** → [https://www.txu.com](https://www.txu.com)
                - **Green Mountain Energy** → [https://www.greenmountainenergy.com](https://www.greenmountainenergy.com)
                - **Constellation Energy** → [https://www.constellation.com](https://www.constellation.com)
                - **Shell Energy** → [https://www.shellenergy.com](https://www.shellenergy.com)
                - **Direct Energy** → [https://www.directenergy.com](https://www.directenergy.com)
                - **Just Energy** → [https://www.justenergy.com](https://www.justenergy.com)
                - **Champion Energy Services** → [https://www.championenergyservices.com](https://www.championenergyservices.com)
                - **Gexa Energy** → [https://www.gexaenergy.com](https://www.gexaenergy.com)

                ### **Reference Websites for Market Comparison**
                - **[PowerToChoose.org](https://www.powertochoose.org)** → Official site for **Texas deregulated energy plans**.
                - **[EnergySage.com](https://www.energysage.com)** → Compare **solar & energy rates** across states.
                - **State Public Utility Commission website** → Direct users to their **state’s regulatory authority** for official energy rates.
                
                **Limitations:**
                - Do **not fabricate** pricing or provider details.
                - If data is missing, provide clear fallback recommendations.
                """

                response = llm_rag.invoke(bill_prompt).content.strip()

            elif use_log_context:
                logs_preview = logs_df.head(10).to_dict(orient="records")
                logs_preview = convert_timestamps(logs_preview)

                log_prompt = f"""
                **Smart Meter Log Analysis**  
                Detect fraud, anomalies, and unusual patterns in the uploaded data.

                ### **Uploaded Log Sample**
                {json.dumps(logs_preview, indent=2)}

                ### **Analysis Task**
                - Detect **anomalous usage patterns**
                - Identify **meter tampering attempts**
                - Spot **unusual power consumption trends**
                - Compare against **expected usage behavior**

                ### **User Query:**  
                **{query}**

                **Response Guidelines:**
                - **Clearly explain anomalies** found (or confirm data looks normal).
                - Provide **structured, professional insights** using real log data.
                - **Suggest actions** if fraud is suspected.
                - **Do not return JSON. Respond in well-structured text.**
                """

                response = llm_rag.invoke(log_prompt).content.strip()

            else:
                rag_qa = RetrievalQA.from_chain_type(
                    llm=llm_rag, retriever=Chroma(persist_directory=os.path.abspath("../db"), embedding_function=embedding_function).as_retriever())
                response = rag_qa.run(query)

                # Fallback: Provide external resources if AI can't generate a confident response
                fallback_keywords = [
                    "i don't know", "not mentioned", "don't know", "unclear", "insufficient information",
                    "not enough information", "no relevant context", "couldn't find", "don't have enough information",
                    "based on the provided context", "cannot determine", "lack of data", "no data available",
                    "cannot find relevant info", "not specified"
                ]

                if any(keyword in response.lower() for keyword in fallback_keywords):
                    st.warning(
                        "We couldn't find a detailed answer. Explore more:")
                    suggestions = [
                        "[Smart Meter Tampering Explained](https://www.smart-energy.com/?s=Energy+theft+smart+meters&ctg=60)",
                        "[Energy Theft Detection Techniques](https://www.sciencedirect.com/search?qs=energy%20theft%20detection)",
                        "[How AI Detects Energy Theft](https://venturebeat.com)"
                    ]

                    for link in suggestions:
                        st.markdown(f"🔗 {link}")

                    search_link = f"https://www.google.com/search?q={query.replace(' ', '+')}"
                    st.markdown(f"🔍 [Search this topic online]({search_link})")

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
        st.markdown("---")
        if st.button("📊 Visualize Logs"):
            st.session_state["active_tab"] = "Visualize Metrics"
            st.session_state["upload_type"] = "Utility Bill" if uploaded_bill else "Smart Meter Log"
            st.rerun()
