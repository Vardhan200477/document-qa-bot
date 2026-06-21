import chromadb
from sentence_transformers import SentenceTransformer
from transformers import pipeline

from config import (
    DB_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL
)

# Load embedding model
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

# Load FLAN-T5 Small model
qa_model = pipeline(
    "text2text-generation",
    model="google/flan-t5-small"
)

# Connect to ChromaDB
client = chromadb.PersistentClient(path=DB_PATH)

# Create collection if it doesn't exist
collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)


def ask_question(question):
    try:
        # Convert question to embedding
        query_embedding = embedding_model.encode(question).tolist()

        # Retrieve top 3 relevant chunks
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]

        # If nothing found
        if len(documents) == 0:
            return "No information found in the document.", []

        # Remove duplicate chunks
        documents = list(dict.fromkeys(documents))

        # Create context
        context = "\n".join(documents)

        # Prompt for FLAN-T5
        prompt = f"""
Answer the question based only on the context below.

Context:
{context}

Question:
{question}

Answer:
"""

        # Generate answer
        response = qa_model(
            prompt,
            max_new_tokens=50
        )

        answer = response[0]["generated_text"]

        return answer, metadatas

    except Exception as e:
        print("ERROR:", str(e))
        return f"Error: {str(e)}", []


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