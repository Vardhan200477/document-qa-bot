import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

from config import (
    DB_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    GEMINI_API_KEY
)

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel("gemini-2.5-flash")

# Load embedding model
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

# Connect to ChromaDB
client = chromadb.PersistentClient(path=DB_PATH)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)


def ask_question(question):
    # Convert question to embedding
    query_embedding = embedding_model.encode(question).tolist()

    # Search similar chunks
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    # Create context
    context = "\n\n".join(documents)

    prompt = f"""
You are a helpful AI assistant.

Answer ONLY using the information present in the context.

If the answer is not present in the context, reply:
"I could not find that information in the document."

Context:
{context}

Question:
{question}

Answer:
"""

    response = model.generate_content(prompt)

    return response.text, metadatas


if __name__ == "__main__":

    while True:

        question = input("\nAsk a question (type 'exit' to quit): ")

        if question.lower() == "exit":
            break

        answer, sources = ask_question(question)

        print("\nAnswer:\n")
        print(answer)

        print("\nSources:")
        for source in sources:
            print(
                f"{source['source']} "
                f"(Page {source['page']})"
            )