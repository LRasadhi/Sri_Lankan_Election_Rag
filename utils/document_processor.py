# utils/document_processor.py
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

def load_pdf(file_path):
    """Load PDF document"""
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    return docs

def split_documents(docs, chunk_size=800, chunk_overlap=150):
    """Split documents into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", ";", ",", " ", ""],
    )
    chunks = text_splitter.split_documents(docs)
    return chunks

def enrich_chunks_with_metadata(chunks, source_file):
    """Add metadata to document chunks"""
    enriched_chunks = []
    
    for i, chunk in enumerate(chunks):
        # Copy existing metadata and add more
        metadata = chunk.metadata.copy()
        metadata['chunk_id'] = i
        metadata['source'] = source_file
        
        # Extract article or section information if present
        content = chunk.page_content
        article_match = re.search(r'Article (\d+)', content)
        chapter_match = re.search(r'Chapter (\d+)', content)
        
        if article_match:
            metadata['article'] = article_match.group(1)
        if chapter_match:
            metadata['chapter'] = chapter_match.group(1)
        
        # Create new document with enriched metadata
        enriched_chunk = Document(
            page_content=content,
            metadata=metadata
        )
        enriched_chunks.append(enriched_chunk)
    
    return enriched_chunks

def process_document(file_path, chunk_size=800, chunk_overlap=150):
    """Process document from loading to chunking with metadata"""
    docs = load_pdf(file_path)
    chunks = split_documents(docs, chunk_size, chunk_overlap)
    enriched_chunks = enrich_chunks_with_metadata(chunks, file_path)
    return enriched_chunks