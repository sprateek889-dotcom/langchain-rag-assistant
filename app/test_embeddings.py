from app.embeddings import create_embeddings

embeddings = create_embeddings()

text = "LangChain is a framework for building LLM applications."

vector = embeddings.embed_query(text)

print("Vector dimensions:", len(vector))

print("\nFirst 10 values:")
print(vector[:10])