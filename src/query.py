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

        # Retrieve top matching chunk
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=1
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]

        if len(documents) == 0:
            return "No information found in the document.", []

        context = documents[0]

        lines = context.split("\n")

        question_lower = question.lower()

        # Identify field being asked
        if "name" in question_lower:
            keyword = "name:"
        elif "education" in question_lower:
            keyword = "education:"
        elif "skill" in question_lower:
            keyword = "skills:"
        elif "project" in question_lower:
            keyword = "projects:"
        elif "experience" in question_lower:
            keyword = "experience:"
        elif "objective" in question_lower:
            keyword = "objective:"
        else:
            # Return entire chunk if field not recognized
            return context, metadatas

        # Find matching line
        for line in lines:
            if line.lower().startswith(keyword):
                answer = line.split(":", 1)[1].strip()
                return answer, metadatas

        return "Information not found in the document.", metadatas

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