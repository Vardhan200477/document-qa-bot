import streamlit as st
import os
import chromadb

from ingest import ingest_pdf
from config import DB_PATH, COLLECTION_NAME

st.set_page_config(page_title="Document ChatBot")

st.title("📄 Document ChatBot")

# Initialize ChromaDB
client = chromadb.PersistentClient(path=DB_PATH)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)

# Automatically ingest sample PDF if collection is empty
if collection.count() == 0:
    ingest_pdf("data/sample_resume.pdf")

# Import query AFTER database initialization
from query import ask_question

# Sidebar
st.sidebar.title("Upload PDF")

uploaded_file = st.sidebar.file_uploader(
    "Choose a PDF",
    type=["pdf"]
)

if uploaded_file is not None:

    save_path = os.path.join(
        "data",
        uploaded_file.name
    )

    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    ingest_pdf(save_path)

    st.sidebar.success("PDF uploaded successfully!")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input(
    "Ask something about your document"
)

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    answer, sources = ask_question(question)

    source_text = "\n\n### Sources\n"

    for source in sources:

        source_text += (
            f"- {source['source']} "
            f"(Page {source['page']})\n"
        )

    final_answer = answer + source_text

    with st.chat_message("assistant"):
        st.markdown(final_answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": final_answer
        }
    )