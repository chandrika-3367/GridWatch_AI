import streamlit as st
import json
from groq import Groq
import os
from dotenv import load_dotenv
import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize embedding and vector DB for RAG
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
persist_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../db"))
vectordb = Chroma(persist_directory=persist_dir,
                  embedding_function=embedding_function)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

llm_rag = ChatOpenAI(
    model="llama3-70b-8192",
    base_url="https://api.groq.com/openai/v1",
    api_key=groq_api_key,
    temperature=0
)
qa_chain = RetrievalQA.from_chain_type(llm=llm_rag, retriever=retriever)

# Direct LLM for logs
client = Groq(api_key=groq_api_key)

# Enhanced chat tab with RAG + LLM switch


def chat_tab():
    st.markdown("""
    Ask questions about energy theft, smart meter patterns, or anomalies in uploaded logs.
    Example: *"What anomalies were detected in the uploaded logs?"*
    """)

    logs_df = st.session_state.get("analyzed_df")

    if logs_df is not None and not logs_df.empty:
        prompt_prefill = "e.g., What anomalies can be found in the uploaded logs?"
    else:
        prompt_prefill = "e.g., How to detect reverse meter tampering?"

    query = st.text_input("Ask GridWatch AI:", placeholder=prompt_prefill)

    use_log_context = logs_df is not None and not logs_df.empty and "uploaded logs" in query.lower()

    if st.button("Get Insights", key="get_insights_btn") and query:
        if not groq_api_key:
            st.error(
                "Groq API key not found. Please set GROQ_API_KEY in your .env file.")
            return

        try:
            if use_log_context:
                sample_logs = logs_df.head(10).to_dict(orient='records')
                log_context = f"""
                The following smart meter logs have been uploaded:
                {json.dumps(sample_logs, indent=2)}
                Use this log context to answer the user's question.
                """
                final_prompt = f"{log_context}\nUser Question: {query}"

                with st.spinner("Analyzing uploaded logs..."):
                    chat_completion = client.chat.completions.create(
                        messages=[
                            {"role": "user", "content": final_prompt.strip()}],
                        model="llama3-70b-8192"
                    )
                    response = chat_completion.choices[0].message.content.strip(
                    )

            else:
                with st.spinner("Retrieving case study insights..."):
                    response = qa_chain.run(query)

            st.success("Response:")
            st.write(response)

            fallback_keywords = [
                "i don't know", "not mentioned", "don't know", "unclear", "insufficient information",
                "not enough information", "no relevant context", "couldn't find", "don't have enough information",
                "based on the provided context", "cannot determine", "lack of data", "no data available",
                "cannot find relevant info", "not specified"
            ]
            if any(keyword in response.lower() for keyword in fallback_keywords):
                st.warning("We couldn't find a detailed answer. Explore more:")
                suggestions = [
                    "[Smart Meter Tampering Explained](https://www.smart-energy.com/?s=Energy+theft+smart+meters&ctg=60)",
                    "[Energy Theft Detection Techniques](https://www.sciencedirect.com/search?qs=energy%20theft%20detection)",
                    "[How AI Detects Energy Theft](https://venturebeat.com)"
                ]
                for link in suggestions:
                    st.markdown(f"🔗 {link}")

                search_link = f"https://www.google.com/search?q={query.replace(' ', '+')}"
                st.markdown(f"🔍 [Search this topic online]({search_link})")

        except Exception as e:
            st.error(f"Error: {str(e)}")

    if logs_df is not None and not logs_df.empty:
        st.markdown("---")
        if st.button("📊 Visualize Logs", key="visualize_btn_chat"):
            st.session_state["active_tab"] = "Visualize Metrics"
            st.rerun()
