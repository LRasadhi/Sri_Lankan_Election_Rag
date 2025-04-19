# app.py
import os
import argparse
from config import (
    GOOGLE_API_KEY, EMBEDDING_MODEL, LLM_MODEL, 
    COLLECTION_NAME, PERSIST_DIRECTORY,
    CHUNK_SIZE, CHUNK_OVERLAP, DEFAULT_TOP_K
)
from utils.document_processor import process_document
from utils.vector_store import (
    initialize_embeddings, load_vector_store, 
    add_documents_to_vector_store, query_vector_store
)
from utils.translation import translate_text
from rag.retriever import HybridRetriever
from rag.generator import AnswerGenerator
from rag.chain import RAGChain

def setup_directories():
    """Set up necessary directories"""
    os.makedirs('data', exist_ok=True)
    os.makedirs(PERSIST_DIRECTORY, exist_ok=True)

def load_documents(file_paths):
    """Load and process documents"""
    all_chunks = []
    for file_path in file_paths:
        if os.path.exists(file_path):
            print(f"Processing {file_path}...")
            chunks = process_document(
                file_path, 
                chunk_size=CHUNK_SIZE, 
                chunk_overlap=CHUNK_OVERLAP
            )
            all_chunks.extend(chunks)
            print(f"Added {len(chunks)} chunks from {file_path}")
        else:
            print(f"File not found: {file_path}")
    
    return all_chunks

def initialize_rag_system():
    """Initialize the RAG system"""
    # Set up embeddings
    embeddings = initialize_embeddings(GOOGLE_API_KEY, EMBEDDING_MODEL)
    
    # Load or create vector store
    vector_store = load_vector_store(
        embeddings, 
        COLLECTION_NAME, 
        PERSIST_DIRECTORY
    )
    
    # Initialize retriever
    retriever = HybridRetriever(vector_store)
    
    # Initialize generator
    generator = AnswerGenerator(GOOGLE_API_KEY, LLM_MODEL)
    
    # Create RAG chain
    rag_chain = RAGChain(retriever, generator)
    
    return vector_store, retriever, generator, rag_chain

def answer_question(question, rag_chain, translate_to_sinhala=False):
    """Answer a question using the RAG system"""
    # Run RAG chain
    result = rag_chain.run(question)
    answer = result["answer"]
    
    if translate_to_sinhala:
        sinhala_answer = translate_text(answer)
        return {
            "question": question,
            "english_answer": answer,
            "sinhala_answer": sinhala_answer,
            "sources": [doc.metadata.get('source', 'Unknown') for doc in result["context"]]
        }
    else:
        return {
            "question": question,
            "answer": answer,
            "sources": [doc.metadata.get('source', 'Unknown') for doc in result["context"]]
        }

def main():
    parser = argparse.ArgumentParser(description='Sri Lankan Constitution RAG System')
    parser.add_argument('--add', nargs='+', help='Add documents to the vector store')
    parser.add_argument('--query', type=str, help='Query the RAG system')
    parser.add_argument('--translate', action='store_true', help='Translate answer to Sinhala')
    
    args = parser.parse_args()
    
    # Set up directories
    setup_directories()
    
    # Initialize RAG system
    vector_store, retriever, generator, rag_chain = initialize_rag_system()
    
    # Process documents if provided
    if args.add:
        chunks = load_documents(args.add)
        if chunks:
            # Add to vector store
            add_documents_to_vector_store(vector_store, chunks)
            print(f"Added {len(chunks)} total chunks to vector store")
            
            # Update retriever's BM25 index
            retriever.initialize_bm25(chunks)
    
    # Answer query if provided
    if args.query:
        result = answer_question(args.query, rag_chain, args.translate)
        
        print("\nQuestion:")
        print(result["question"])
        
        print("\nAnswer:")
        print(result["answer"])
        
        if args.translate:
            print("\nSinhala Translation:")
            print(result["sinhala_answer"])
        
        print("\nSources:")
        for src in result["sources"]:
            print(f"- {src}")

if __name__ == "__main__":
    main()