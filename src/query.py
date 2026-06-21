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

# Get collection
collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)


def ask_question(question):
    try:
        # Convert question to embedding
        query_embedding = embedding_model.encode(question).tolist()

        # Retrieve most relevant chunk
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=1
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]

        if len(documents) == 0:
            return "No information found in the document.", []

        context = documents[0]

        print("Context:")
        print(context)

        question_lower = question.lower()

        # Map question keywords
        if "name" in question_lower:
            keyword = "name"
        elif "skill" in question_lower:
            keyword = "skill"
        elif "education" in question_lower:
            keyword = "education"
        elif "project" in question_lower:
            keyword = "project"
        elif "experience" in question_lower:
            keyword = "experience"
        elif "objective" in question_lower:
            keyword = "objective"
        else:
            return context, metadatas

        # Search line by line
        lines = context.split("\n")

        for line in lines:
            if keyword in line.lower():
                if ":" in line:
                    return line.split(":", 1)[1].strip(), metadatas
                else:
                    return line.strip(), metadatas

        # Fallback: search in full context
        if keyword in context.lower():
            return context, metadatas

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