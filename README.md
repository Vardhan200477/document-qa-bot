# 📄 Document Q&A Bot using RAG

## Overview

Document Q&A Bot is a Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and ask questions about their contents. The system extracts text from PDFs, creates embeddings, stores them in ChromaDB, and uses Google's Gemini API to generate context-aware answers.

---

## Features

- Upload PDF documents
- Extract text from pages
- Create embeddings
- Store vectors in ChromaDB
- Semantic search
- Generate answers using Gemini API
- Interactive Streamlit interface

---

## Tech Stack

- Python
- Streamlit
- Google Gemini API
- ChromaDB
- Sentence Transformers
- PyPDF
- LangChain

---

## Project Structure

```
document-qa-bot
│
├── data/
├── src/
│   ├── app.py
│   ├── config.py
│   ├── ingest.py
│   └── query.py
├── .env
├── README.md
└── requirements.txt
```

---

## Workflow

1. Upload PDF document.
2. Extract text from pages.
3. Generate embeddings.
4. Store vectors in ChromaDB.
5. Retrieve relevant chunks.
6. Generate answers using Gemini API.

---

## Installation

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run src/app.py
```

---

## Sample Questions

- What are the skills?
- What projects has the candidate worked on?
- Summarize the document.

---

## Future Enhancements

- Multiple PDF support
- OCR support
- Chat history
- Source citations
- Streamlit Cloud deployment