# utils/vector_store.py
import os
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def initialize_embeddings(api_key, model_name):
    """Initialize Gemini embeddings"""
    os.environ["GOOGLE_API_KEY"] = api_key
    embeddings = GoogleGenerativeAIEmbeddings(model=model_name)
    return embeddings

def create_vector_store(embeddings, collection_name, persist_directory):
    """Create a new vector store"""
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory
    )
    return vector_store

def load_vector_store(embeddings, collection_name, persist_directory):
    """Load existing vector store"""
    if os.path.exists(persist_directory):
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_directory
        )
        return vector_store
    else:
        return create_vector_store(embeddings, collection_name, persist_directory)

def add_documents_to_vector_store(vector_store, documents):
    """Add documents to vector store"""
    vector_store.add_documents(documents=documents)
    return len(documents)

def query_vector_store(vector_store, query, k=4, filter=None):
    """Query vector store with optional metadata filter"""
    if filter:
        results = vector_store.similarity_search(query, k=k, filter=filter)
    else:
        results = vector_store.similarity_search(query, k=k)
    return results