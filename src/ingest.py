from pypdf import PdfReader
import os
import chromadb
from sentence_transformers import SentenceTransformer

from config import (
    DB_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)

# Load embedding model
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

# Connect to ChromaDB
client = chromadb.PersistentClient(path=DB_PATH)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)


def extract_pdf_pages(file_path):

    extracted_pages = []

    file_name = os.path.basename(file_path)

    reader = PdfReader(file_path)

    for index, page in enumerate(reader.pages):

        text = page.extract_text()

        if text and text.strip():

            extracted_pages.append(
                {
                    "text": text,
                    "metadata": {
                        "source": file_name,
                        "page": index + 1
                    }
                }
            )

    return extracted_pages


def chunk_text(text,
               chunk_size=CHUNK_SIZE,
               overlap=CHUNK_OVERLAP):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def ingest_pdf(file_path):

    pages = extract_pdf_pages(file_path)

    document_chunks = []
    metadatas = []
    ids = []

    current_count = collection.count()

    for page in pages:

        chunks = chunk_text(page["text"])

        for chunk in chunks:

            document_chunks.append(chunk)

            metadatas.append(page["metadata"])

            ids.append(str(current_count))

            current_count += 1

    if len(document_chunks) == 0:

        print("No text found inside PDF.")
        return

    embeddings = embedding_model.encode(
        document_chunks
    ).tolist()

    collection.add(
        documents=document_chunks,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

    print(f"Successfully stored {len(document_chunks)} chunks.")


if __name__ == "__main__":

    pdf_path = "data/sample_resume.pdf"

    ingest_pdf(pdf_path)