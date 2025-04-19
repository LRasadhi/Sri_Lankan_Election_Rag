# rag/chain.py
from langgraph.graph import START, END, StateGraph
from typing_extensions import TypedDict, List
from langchain_core.documents import Document

class RAGState(TypedDict):
    question: str
    context: List[Document]
    answer: str

class RAGChain:
    """RAG Chain using LangGraph"""
    def __init__(self, retriever, generator):
        self.retriever = retriever
        self.generator = generator
        self.chain = self._build_chain()
    
    def _retrieve(self, state: RAGState):
        """Retrieve relevant documents"""
        question = state["question"]
        docs = self.retriever.retrieve(question, k=4)
        return {"context": docs, "question": question}
    
    def _generate(self, state: RAGState):
        """Generate answer"""
        question = state["question"]
        context = state["context"]
        answer = self.generator.generate(question, context)
        return {"question": question, "context": context, "answer": answer}
    
    def _build_chain(self):
        """Build the RAG chain using LangGraph"""
        workflow = StateGraph(RAGState)
        
        workflow.add_node("retrieve", self._retrieve)
        workflow.add_node("generate", self._generate)
        
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)
        
        return workflow.compile()
    
    def run(self, question):
        """Run the RAG chain"""
        initial_state = {
            "question": question,
            "context": [],
            "answer": ""
        }
        
        result = self.chain.invoke(initial_state)
        return result