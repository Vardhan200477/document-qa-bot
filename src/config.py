import streamlit as st

GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

DB_PATH = "./db"
COLLECTION_NAME = "document_knowledge_base"

CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

EMBEDDING_MODEL = "all-MiniLM-L6-v2"