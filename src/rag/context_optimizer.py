import re
from typing import List, Tuple, Optional
from langchain_core.documents import Document

class ContextOptimizer:
    def __init__(self, max_tokens = 4000, reserve_tokens = 500):
        self.max_tokens = max_tokens
        self.reserve_tokens = reserve_tokens
        self.available_context_tokens = self.max_tokens - self.reserve_tokens

    def estimate_tokens(self, text):
        return len(text) // 4
    
    def count_question_tokens(self, question, template):
        sample_prompt = template.format(context="[CONTEXT_PLACEHOLDER]",
                                        question=question)
        base_tokens = self.estimate_tokens(sample_prompt) - self.estimate_tokens("[CONTEXT_PLACEHOLDER]")
        return base_tokens + 50
    
    def calculate_available_context_tokens(self, question, template):
        question_tokens = self.count_question_tokens(question, template)
        available_tokens = self.available_context_tokens - question_tokens
        return max(available_tokens, 500)
    
    def optimize_chunks_for_context(self, chunks, question, template):
        available_tokens = self.calculate_available_context_tokens(question, template)
        selected_chunks = []
        used_tokens = 0

        for chunk in chunks:
            chunk_tokens = self.estimate_tokens(chunk.page_content)
            if used_tokens + chunk_tokens <= available_tokens:
                selected_chunks.append(chunk)
                used_tokens += chunk_tokens

            else:
                break
        return selected_chunks
    

if __name__ == "__main__":
    optimizer = ContextOptimizer()
    
    # Test with sample text
    template = """Use the following context to answer the question.
    
    Context: {context}
    
    Question: {question}
    
    Answer:"""
    
    question = "How do attention mechanisms work?"
    question_tokens = optimizer.count_question_tokens(question, template)
    print(f"Question + template tokens: {question_tokens}")
    available = optimizer.calculate_available_context_tokens(question, template)
    print(f"Available tokens for context: {available}")

    sample_chunks = [
        Document(page_content="First chunk about attention mechanisms." * 10),
        Document(page_content="Second chunk about transformers." * 15), 
        Document(page_content="Third chunk about neural networks." * 20)
    ]
    
    optimized_chunks = optimizer.optimize_chunks_for_context(sample_chunks, question, template)
    print(f"Original chunks: {len(sample_chunks)}")
    print(f"Optimized chunks: {len(optimized_chunks)}")