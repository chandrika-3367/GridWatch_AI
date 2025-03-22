# GridWatch AI

> **AI-Powered Smart Meter Fraud Detection using RAG + ML Analysis**

GridWatch AI is an advanced fraud detection system that leverages document-based insights through Retrieval-Augmented Generation (RAG) and real-time smart meter data via AI and Machine Learning (ML). It identifies anomalies, predicts fraudulent activity, and helps utilities ensure energy integrity through smart analytics and actionable reports.

---

## Features

- **Smart Log Parsing & Analysis** – Upload diverse log files and dynamically extract structured data using rule-based and LLM-based fallback parsers.
- **Chat with GridWatch AI** – Interact via AI chat powered by RAG (for case studies) or LLM (for log-specific queries).
- **Visualize Metrics** – KPI analysis, anomaly detection, geospatial fraud mapping, and downloadable reports.
- **AI-Based Fraud Detection** – ML model flags fraudulent patterns, compares with visual analysis, and provides confidence scores.
- **Auto-Ticket Generation** – Raise fraud tickets with a mock API integration; download acknowledgment reports.
- **Actionable Insights** – Compare visual vs ML fraud detection; take targeted actions.

---

## Tech Stack

| Component         | Technology Used                                       |
|------------------|------------------------------------------------------|
| **Frontend**     | Streamlit                                            |
| **Backend**      | Python (Pandas, Plotly, PyMuPDF, LangChain)          |
| **Embedding**    | HuggingFace Embeddings (all-MiniLM-L6-v2)            |
| **Vector DB**    | ChromaDB                                             |
| **RAG Engine**   | LangChain                                            |
| **LLM**          | Groq (OpenAI-compatible endpoint)                    |
| **ML Module**    | Custom fraud detection model (RFC)                   |

---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/chandrika-3367/GridWatch_AI.git
cd GridWatch_AI
```

### 2. Install Requirements
```bash
pip install -r requirements.txt
```

### 3. Prepare Case Studies
Place your fraud-related PDF case studies in the `case_studies/` folder.

### 4. Chunk and Embed PDFs
```bash
python Scripts/chunk_and_embed.py
```
> This splits documents, creates embeddings, and stores them in ChromaDB.

### 5. Run the App
```bash
streamlit run app/main.py
```

---

## Updated Usage Overview

1. Upload logs (TXT, CSV, JSON, PDF, DOCX, XLSX) and parse dynamically.
2. Visualize metrics (KPI trends, anomalies, fraud mapping).
3. **NEW:** Click "Detect Fraud Using AI" to trigger ML analysis.
4. Compare ML fraud detection with visual anomalies.
5. Generate tickets and download reports.


---

## Acknowledgements
- HuggingFace for embeddings.
- LangChain and ChromaDB for RAG pipelines.
- Groq for blazing-fast LLM access.

---

## Project Status
GridWatch AI is live with:

- RAG + LLM dual-mode chat
- Real-time log parsing and analysis
- ML-powered fraud detection module
- Dynamic UI with fraud reporting and AI insights
- Mock ticket API integration

