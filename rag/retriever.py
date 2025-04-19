# rag/retriever.py
from rank_bm25 import BM25Okapi
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class HybridRetriever:
    """Hybrid retriever combining vector search and BM25"""
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.stop_words = set(stopwords.words('english'))
        
        # We'll initialize BM25 on demand with the documents
        self.bm25 = None
        self.doc_map = None
        self.all_documents = []
    
    def initialize_bm25(self, documents):
        """Initialize BM25 with documents"""
        self.all_documents = documents
        tokenized_docs = []
        
        for doc in documents:
            tokens = self._preprocess_text(doc.page_content)
            tokenized_docs.append(tokens)
        
        self.bm25 = BM25Okapi(tokenized_docs)
        self.doc_map = {i: doc for i, doc in enumerate(documents)}
    
    def _preprocess_text(self, text):
        """Preprocess text for BM25"""
        # Convert to lowercase and split on non-alphanumeric characters
        tokens = re.findall(r'\w+', text.lower())
        return [t for t in tokens if t not in self.stop_words]
    
    def retrieve(self, query, k=4, use_hybrid=True, filter=None):
        """Retrieve documents using hybrid approach"""
        # Vector search
        vector_results = self.vector_store.similarity_search(
            query, 
            k=k,
            filter=filter
        )
        
        if not use_hybrid or self.bm25 is None:
            return vector_results
        
        # BM25 search
        query_tokens = self._preprocess_text(query)
        if not query_tokens:
            return vector_results
            
        scores = self.bm25.get_scores(query_tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        bm25_results = [self.doc_map[idx] for idx in top_indices]
        
        # Combine results
        combined_results = vector_results.copy()
        seen_content = {doc.page_content for doc in combined_results}
        
        for doc in bm25_results:
            if doc.page_content not in seen_content:
                combined_results.append(doc)
                seen_content.add(doc.page_content)
        
        # Return top-k
        return combined_results[:k]