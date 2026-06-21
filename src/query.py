import chromadb
from sentence_transformers import SentenceTransformer

from config import (
    DB_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL
)

# Load embedding model
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

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

        # Search similar chunks
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=1
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]

        print("Question:", question)
        print("Documents found:", documents)

        # No matching documents
        if len(documents) == 0:
            return "No information found in the document.", []

        # Return the most relevant chunk
        answer = documents[0]

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