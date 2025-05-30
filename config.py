import os

# Google API Key (Replace with your actual API key)
GOOGLE_API_KEY = "AIzaSyDaoVveb-GN6JdMM7gog1dxkjrRpok-qAU"

# Model configurations
EMBEDDING_MODEL = "models/embedding-001"  # Gemini embedding model
LLM_MODEL = "gemini-2.0-flash"  # Gemini Pro model for text generation

# Vector store configurations
COLLECTION_NAME = "sri_lanka_electoral"
PERSIST_DIRECTORY = "chroma_db"

# Document processing configurations
CHUNK_SIZE = 1000  # Increased for better context
CHUNK_OVERLAP = 200  # Increased for better continuity
DEFAULT_TOP_K = 4  # Number of documents to retrieve