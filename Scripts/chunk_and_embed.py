import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Load environment variables
load_dotenv()

# Use HuggingFace Embeddings (no API key needed)
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Paths for PDFs and persistent storage
pdf_folder = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "../case_studies"))
persist_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../db"))

# Ensure persistence directory exists
os.makedirs(persist_dir, exist_ok=True)

# Load and process PDFs
all_docs = []
for file_name in os.listdir(pdf_folder):
    if file_name.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(pdf_folder, file_name))
        docs = loader.load()
        all_docs.extend(docs)

print(f"Loaded {len(all_docs)} documents from {pdf_folder}")

# Chunk documents for embedding
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200)
chunked_docs = text_splitter.split_documents(all_docs)
print(f"Chunked into {len(chunked_docs)} document chunks.")


# Embed and store in ChromaDB (persistent)
print("Embedding and storing chunks in ChromaDB...")
vectordb = Chroma.from_documents(
    chunked_docs, embedding_function, persist_directory=persist_dir)
vectordb.persist()

print(f"Embedding complete! Stored in '{persist_dir}'. Ready for querying.")
