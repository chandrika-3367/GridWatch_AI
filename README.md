# GridWatch AI

> **AI-Powered Smart Meter Fraud Detection using RAG (Retrieval-Augmented Generation)**

GridWatch AI is an innovative tool that uses AI to detect fraudulent energy usage patterns through smart meters. By leveraging document-based case studies, it enables proactive detection and visualization of energy theft while suggesting actionable insights.

---

## Features

- **Summarize Smart Meter Data** – Quickly summarize patterns and anomalies from uploaded logs or datasets.
- **Search Case Studies (RAG)** – Retrieve relevant insights from embedded fraud-related case studies.
- **Visualize KPIs** – Traffic metrics, energy usage trends, and potential fraud spikes.
- **Preventive Suggestions** – Context-aware recommendations to prevent energy theft.
- **Auto-Ticket Generation** – Raise automated complaints or reports when fraud patterns are detected.

---

## Tech Stack

| Component       | Technology Used                                       |
|----------------|------------------------------------------------------|
| **Frontend**    | Streamlit                                            |
| **Backend**     | Python, Flask                                        |
| **Embedding**   | HuggingFace Embeddings (all-MiniLM-L6-v2)            |
| **Vector DB**   | ChromaDB                                             |
| **RAG Engine**  | LangChain                                            |
| **LLM (Optional)**| Groq (GPT-compatible) or OpenAI                    |

---

##  Setup Instructions

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
> This will split the documents, create embeddings, and store them in a persistent ChromaDB.

### 5. Run the App
```bash
streamlit run app/main.py
```

---

##  Usage Overview

1. Upload energy usage logs or select pre-existing records.
2. Interact via chatbot to summarize or inquire about suspicious patterns.
3. View suggested actions and visualize fraud metrics.
4. Automatically generate reports or tickets based on fraud indicators.

---

## Acknowledgements
- HuggingFace for open embeddings.
- LangChain for seamless RAG integrations.
- ChromaDB for efficient vector storage.

---

## Project Status
GridWatch AI has been successfully developed and deployed with the following completed milestones:

- PDF-based case studies ingestion and embedding
- Persistent vector storage via ChromaDB
- AI-powered summarization and visualization modules
- Fraud detection logic and ticket generation
- Fully functional Streamlit UI for user interaction



