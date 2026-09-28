#from app.vector_store import create_vector_store
from app.vector_store import load_vector_store

#Before load vector store, we need to create a 
#retriever from the vector store. 
#The retriever will be used to fetch 
#document chunks based on user queries.
#def create_retriever(chunks):
def create_retriever():
#    vector_store = create_vector_store()
    vector_store = load_vector_store()

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 3}
    )

    return retriever