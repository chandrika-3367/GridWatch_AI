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
from difflib import SequenceMatcher
from datetime import datetime
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize embedding function
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
client = Groq(api_key=groq_api_key)

# RAG setup
llm_rag = ChatOpenAI(
    model="llama3-70b-8192",
    base_url="https://api.groq.com/openai/v1",
    api_key=groq_api_key,
    temperature=0
)

# Embedding setup
qa_chain = RetrievalQA.from_chain_type(
    llm=llm_rag,
    retriever=Chroma(
        persist_directory=os.path.abspath("../db"),
        embedding_function=embedding_function
    ).as_retriever()
)

# convert timestamp fields inside log dictionaries into ISO format


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

# Concatenates all uploaded bill texts


def extract_combined_bill_text():
    """Retrieve and combine text from all uploaded bills."""
    if "uploaded_bills" in st.session_state and isinstance(st.session_state["uploaded_bills"], list):
        bill_texts = [bill.get("text", "").strip(
        ) for bill in st.session_state["uploaded_bills"] if isinstance(bill, dict) and "text" in bill]

        if bill_texts:
            return "\n\n---\n\n".join(bill_texts)

    return ""  # Return empty string instead of None to prevent errors

# Function to extract provider and location


def extract_provider_and_location(pdf_texts):
    known_providers = [
        "Reliant Energy", "TXU Energy", "Green Mountain Energy", "Dominion Energy",
        "Shell Energy", "Constellation Energy", "Duke Energy", "Georgia Power",
        "PG&E", "SoCal Edison", "Entergy", "FPL", "Con Edison", "National Grid",
        "Xcel Energy", "PSE&G", "APS", "O'Connell Energy"
    ]

    providers, locations = set(), set()

    for pdf_text in pdf_texts:
        flat_text = pdf_text.replace("\n", " ").replace("  ", " ")

        for prov in known_providers:
            if prov.lower() in flat_text.lower():
                providers.add(prov)

        known_line_match = re.search(
            r"(" + "|".join(known_providers) + r".*?)\s+(Retail|Services|LLC)", flat_text, re.IGNORECASE)
        if known_line_match:
            providers.add(known_line_match.group(1).strip())

        addr_match = re.search(
            r"(?:Service Address|Billing Address|Customer Address|Meter usage Service Address):\s*(.*?)(?=\s{2,}|\n|$)",
            flat_text, re.IGNORECASE)
        if addr_match:
            candidate = addr_match.group(1).strip()
            if re.search(r"[A-Z]{2} \d{5}", candidate):
                locations.add(candidate)

    if not providers or not locations:
        try:
            combined_text = "\n\n---\n\n".join(pdf_texts)
            fallback_prompt = f"""
            Extract only the utility provider and billing location from all the uploaded bill texts.

            Return your output in valid JSON like this:
            {{
              "provider_name": "...",
              "billing_location": "City, State ZIP"
            }}

            Bill Text:
            {combined_text[:2000]}
            """

            from groq import Groq
            groq_api_key = os.getenv("GROQ_API_KEY")
            client = Groq(api_key=groq_api_key)

            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": fallback_prompt}],
                model="llama3-70b-8192"
            )

            llm_response = chat_completion.choices[0].message.content
            match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if not match:
                print(
                    "⚠️ [GridWatch] Fallback LLM response contained no valid JSON block.")
                return

            extracted = json.loads(match.group())
            if extracted.get("provider_name"):
                providers.add(extracted["provider_name"].strip())
            if extracted.get("billing_location"):
                locations.add(extracted["billing_location"].strip())

        except Exception as e:
            print(
                f"⚠️ [GridWatch] Silent fallback error in provider/location LLM parsing: {str(e)}")

    st.session_state["provider_names"] = list(
        providers) if providers else ["Not Found"]
    st.session_state["billing_locations"] = list(
        locations) if locations else ["Not Found"]

# Actual Chat Tab logic


def chat_tab():
    st.markdown("""
    **Ask questions about energy anomalies, smart meter patterns, or recommendations from your utility bill.**

    _Example Queries:_
    - *What anomalies were detected in the uploaded logs?*
    - *What are common fraud indicators in smart meter data?*
    - *How does my provider compare to other companies in terms of pricing?*
    """)

    if st.session_state.get("files_ready", False):
        st.info("Utility bills/logs uploaded. You can now ask questions.")
    else:
        st.info(
            "Ask questions based on energy usage anamolies or upload logs/bills for specific insights.")

    query = st.text_input(
        "Ask GridWatch AI:", placeholder="e.g., How does smart meter fraud detection work?")

    logs_df = st.session_state.get("analyzed_df", pd.DataFrame())
    pdf_text = extract_combined_bill_text()

    use_log_context = not logs_df.empty and "log" in query.lower()
    use_bill_context = pdf_text is not None and any(keyword in query.lower() for keyword in [
        "bill", "provider", "usage", "plan", "cancellation", "reliant", "txu", "green mountain", "constellation",
        "shell energy", "compare", "cheaper options"
    ])

    if st.button("Get Insights") and query:
        try:
            response = "Analyzing your query, please wait..."

            if use_bill_context:
                pdf_texts = [bill["text"]
                             for bill in st.session_state.get("uploaded_bills", [])]
                extract_provider_and_location(pdf_texts)

                providers = st.session_state.get(
                    "provider_names", ["Not Found"])
                locations = st.session_state.get(
                    "billing_locations", ["Not Found"])

                if "Not Found" in providers or "Not Found" in locations:
                    st.warning(
                        "Could not extract provider details or location. AI response might be limited.")

                # Construct AI prompt for multiple providers & locations
                provider_str = ", ".join(providers)
                location_str = ", ".join(locations)

                bill_prompt = f"""
                ### **Persona:**
                You are an **AI energy analyst** with expertise in **utility bill interpretation, rate comparisons, and consumer energy guidance**.

                ### **Task:**
                Analyze the uploaded **utility bills** for the following service providers:
                **{provider_str}**

                Locations detected: **{location_str}**

                Answer the user's query based on the available bill details.

                **User Query:** {query}

                - **Compare provider rates & services if multiple providers exist.**
                - **Provide energy-saving recommendations based on bill details.**
                - **If providers are recognized, include their official website links.**

                ### **Provider Comparison & Recommendations**
                If the user is looking for alternative energy providers, suggest reputable competitors in their region.
                Provide **direct website links** for each, ensuring they can easily explore alternative options.

                **Limitations:**
                - Do **not fabricate** pricing or provider details.
                - If data is missing, provide clear fallback recommendations.
                """

                response = llm_rag.invoke(bill_prompt).content.strip()
                st.success("Response:")
                st.write(response)

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
                """

                response = llm_rag.invoke(log_prompt).content.strip()
                st.success("Response:")
                st.write(response)

            else:

              # Construct your retrieval-aware prompt
                template = """
                You are GridWatch AI — a domain expert in smart meter fraud, energy theft, and anomaly detection.

                Below is an excerpt retrieved from internal energy fraud case studies:

                {context}

                ---

                Answer the user's question based only on the above context.
                If the context is unrelated or insufficient, say:
                "I don't have enough case study context to answer that."

                User Question: {question}
                """

                prompt = PromptTemplate(template=template, input_variables=[
                                        "context", "question"])

                # Create RAG QA chain with explicit template
                rag_qa = RetrievalQA.from_chain_type(
                    llm=llm_rag,
                    retriever=Chroma(
                        persist_directory=os.path.abspath("../db"),
                        embedding_function=embedding_function
                    ).as_retriever(),
                    chain_type_kwargs={"prompt": prompt}
                )

                # Run RAG
                response = rag_qa.run(query)

                st.success("Response:")
                st.write(response)

                # Fallback keyword patterns that signal weak/confused responses
                fallback_keywords = [
                    "i don't know", "not mentioned", "unclear", "insufficient information",
                    "no relevant context", "cannot determine", "lack of data", "no data available",
                    "cannot find relevant info", "not specified", "don't have enough information"
                ]

                # Generic non-domain queries
                general_knowledge_keywords = [
                    "who is", "what is", "where is", "biography", "how to",
                    "history of", "meaning of", "definition of", "fun facts about"
                ]

                # Domain-specific keywords to override fallback
                domain_keywords = [
                    "fraud", "tampering", "usage", "smart meter", "billing", "energy theft",
                    "utility provider", "kwh", "meter id", "rate plan"
                ]

                def is_low_confidence(resp):
                    return any(
                        SequenceMatcher(None, resp.lower(), kw).ratio(
                        ) > 0.8 or kw in resp.lower()
                        for kw in fallback_keywords
                    )

                is_general_knowledge = any(kw in query.lower()
                                           for kw in general_knowledge_keywords)
                is_domain_relevant = any(kw in query.lower()
                                         for kw in domain_keywords)

                # Trigger fallback ONLY if non-domain and weak response
                if (is_low_confidence(response) or is_general_knowledge) and not is_domain_relevant:
                    st.warning(
                        "This topic seems outside GridWatch AI’s case study knowledge base.")

                    st.markdown("""
                    Try asking a question related to:

                    - Smart meter fraud patterns
                    - Energy tampering detection
                    - Billing anomalies or usage discrepancies
                    - Utility provider comparisons based on uploaded bills
                    """)

                    suggestions = [
                        "[Smart Meter Tampering Explained](https://www.smart-energy.com/?s=Energy+theft+smart+meters&ctg=60)",
                        "[Energy Tampering Detection Techniques](https://www.sciencedirect.com/search?qs=energy%20theft%20detection)",
                        "[How AI Detects Energy anamolies](https://venturebeat.com)"
                    ]

                    for link in suggestions:
                        st.markdown(f"🔗 {link}")

                    search_link = f"https://www.google.com/search?q={query.replace(' ', '+')}"
                    st.markdown(f"🔍 [Search this topic online]({search_link})")

            st.download_button(
                "📄 Download Response as Report",
                data=f"GridWatch AI Chat Response\n\nQuery: {query}\n\nResponse:\n{response}",
                file_name="GridWatchAI_Chat_Report.txt",
                mime="text/plain"
            )

        except Exception as e:
            st.error(f"Error during chat response: {str(e)}")
            st.warning("Please check your query or try again later.")

    if logs_df is not None and not logs_df.empty:
        st.markdown("---")
        if st.button("📊 Visualize Logs"):
            st.session_state["active_tab"] = "Visualize Metrics"
            st.session_state["upload_type"] = "Utility Bill" if (
                "uploaded_bills") else "Smart Meter Log"
            st.rerun()
