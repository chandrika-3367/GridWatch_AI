import streamlit as st
from chat_interface import chat_tab
from log_upload_parser import log_upload_tab
from metrics_visualization import visualize_tab
from fraud_detection import fraud_detection_tab
from dotenv import load_dotenv
import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize embedding function
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load ChromaDB (persistent)
persist_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../db"))
vectordb = Chroma(persist_directory=persist_dir,
                  embedding_function=embedding_function)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# Initialize LLM with Groq API for document search (QA Chain)
llm = ChatOpenAI(
    model="llama3-70b-8192",
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY,
    temperature=0
)

# Setup Retrieval QA chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Streamlit UI Setup
st.set_page_config(
    page_title="GridWatch AI - Fraud Detection Chat", layout="wide")
st.title("GridWatch AI - Smart Meter Fraud Detection")

# Inject custom sidebar CSS for aesthetics
st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background-color: #f0f8ff;
    }
    .css-1cpxqw2 {
        font-size: 16px;
        font-weight: 500;
        color: #004080;
    }
    </style>
""", unsafe_allow_html=True)

# Define tab labels
tab_labels = ["Chat with GridWatch AI", "Upload Logs for Analysis",
              "Visualize Metrics", "AI-Based Detection"]

# Initialize session state explicitly
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = tab_labels[0]

if 'log_context_available' not in st.session_state:
    st.session_state.log_context_available = False

# Sidebar with styled radio buttons
with st.sidebar:
    st.image("/Users/chandrikajallipalli/Desktop/GridWatch_AI/gridwatch_logo.png.png",
             width=150)
    st.markdown(
        "GridWatch AI — A Multi-Modal AI Agent for Smart Meter and Utility Bill Analysis.")

    st.markdown("---")
    st.markdown("Switch between GridWatch AI features:")

    selected_tab = st.radio(
        "Go to:",
        tab_labels,
        index=tab_labels.index(st.session_state.active_tab),
        key="sidebar_nav"
    )

    st.markdown("---")
    st.markdown("GridWatch AI detects energy anomalies in smart meter data using AI-powered analytics. Analyze logs, visualize trends, and uncover insights to ensure energy integrity and operational efficiency.")

st.session_state.active_tab = selected_tab

# Tab routing
if selected_tab == "Upload Logs for Analysis":
    log_upload_tab()

elif selected_tab == "Visualize Metrics":
    visualize_tab()

elif selected_tab == "AI-Based Detection":
    df = st.session_state.get("analyzed_df")
    if df is not None and not df.empty:
        fraud_detection_tab()
    else:
        st.info(
            "Please upload and analyze logs in the 'Upload Logs' or 'Visualize Metrics' tab first.")

elif selected_tab == "Chat with GridWatch AI":
    logs_df = st.session_state.get("analyzed_df")
    if logs_df is not None and not logs_df.empty:
        st.session_state.log_context_available = True
    else:
        st.session_state.log_context_available = False
    chat_tab()
