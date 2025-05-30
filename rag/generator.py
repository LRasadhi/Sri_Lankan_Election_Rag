# rag/generator.py
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI

class AnswerGenerator:
    """Generate answers using Gemini"""
    def __init__(self, api_key, model_name="gemini-2.0-flash", temperature=0.7):
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=temperature
        )
    
    def generate(self, question, context_docs):
        """Generate answer from context documents"""
        context = "\n\n".join(doc.page_content for doc in context_docs)
        
        prompt_template = """You are an expert on Sri Lankan electoral systems, focusing on:
1. Various apportionment methods used in Sri Lanka
2. Electoral systems and voting mechanisms
3. Constitutional provisions related to elections
4. District-wise seat allocation methods
5. Proportional representation system

Using ONLY the provided context, answer the following question. If you cannot find enough information in the context, clearly state what specific information is missing rather than making assumptions.

Context:
{context}

Question: {question}

Please provide a clear and structured answer that:
- Directly addresses the question
- Uses specific examples from the context when available
- Explains technical terms related to electoral systems
- Cites relevant constitutional articles or provisions if mentioned in the context

Answer:"""
        
        messages = [
            {"role": "system", "content": "You are a specialized assistant focusing on Sri Lankan electoral systems and apportionment methods. Provide accurate, well-structured answers based solely on the given context."},
            {"role": "user", "content": prompt_template.format(context=context, question=question)}
        ]
        
        response = self.llm.invoke(messages)
        return response.content