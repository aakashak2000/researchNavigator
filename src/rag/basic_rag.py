from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.vectorstores import VectorStore
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM

from src.rag.query_expander import QueryExpander

from typing import List
import numpy as np
import chromadb
import os
import re

class BasicRAG:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.query_expander = QueryExpander()

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap,
            length_function = len,
            separators = ["\n\n", "\n", " ", ""]
        )

        self.embeddings = HuggingFaceEmbeddings(
            model_name = "sentence-transformers/all-MiniLM-L6-v2", 
            model_kwargs = {'device': 'mps'}
            )
        
        self.vector_store: VectorStore = None
        self.llm = OllamaLLM(
            model = "llama3.1:8b",
            temperature=0.1)
        self.prompt_template = PromptTemplate(
                input_variables=["context", "question"],
                template="""Use the following context to answer the question. If you cannot answer based on the context, say so.

                            Context: {context}

                            Question: {question}

                            Answer:"""                            
        )
        
    def load_documents(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_path.endswith('pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith('txt'):
            loader = TextLoader(file_path)
        else:
            raise ValueError(f"Unsupported File Type: {file_path}")
        
        documents = loader.load()
        print(f"Loaded {len(documents)} documents/pages from {file_path}")
        return documents
    
    def deduplicate_chunks(self, chunks, similarity_threshold = 0.85):

        if not chunks:
            return chunks
        
        unique_chunks = [chunks[0]]
        
        for i, chunk in enumerate(chunks[1:], 1):
            is_duplicate = False
            max_similarity = 0.0

            for j, uc in enumerate(unique_chunks):
                
                score = self._cosine_similarity(
                    self.embeddings.embed_query(chunk.page_content),
                    self.embeddings.embed_query(uc.page_content)
                )

                if score > similarity_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_chunks.append(chunk)

        return unique_chunks

    
    def split_documents(self, documents):
        chunks = self.text_splitter.split_documents(documents)
        print(f"Split into {len(chunks)} chunks")
        return chunks
    
    def _cosine_similarity(self, vec1, vec2):
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)

        return float(similarity)
        
    
    def _question_answer_alignment(self, question, chunk_text):
        question_embedding = self.embeddings.embed_query(question)
        chunk_embedding = self.embeddings.embed_query(chunk_text)

        similarity = self._cosine_similarity(question_embedding, chunk_embedding)
        return similarity
    
    def _content_density(self, chunk_text):
        word_count = len(chunk_text.split())
        sentence_count = len([s for s in chunk_text.split('.') if s.strip()])
        if sentence_count == 0:
            return 0.1
        average_sentence_length = word_count / sentence_count
        if average_sentence_length > 20:
            density = 0.9
        elif average_sentence_length > 15:
            density = 0.7
        elif average_sentence_length > 10:
            density = 0.5
        else:
            density = 0.3

        return density
    
    def _explanation_quality(self, chunk_text):
        chunk_lower = chunk_text.lower()
        explanation_words = [
            'describes', 'defined as', 'works by', 'computed as', 'calculated by',
            'consists of', 'composed of', 'mechanism', 'process', 'algorithm',
            'formula', 'equation', 'method', 'approach', 'technique'
        ]
        
        weak_words = [
            'conclusion', 'summary', 'in this paper', 'we present', 
            'results show', 'performance', 'evaluation', 'comparison'
        ]

        explanation_count = sum(1 for word in explanation_words if word in chunk_lower)
        weak_count = sum(1 for word in weak_words if word in chunk_lower)

        explanation_score = min(explanation_count * 0.3, 1.0)
        weak_penalty = min(weak_count * 0.2, 0.5)

        final_score = max(explanation_score - weak_penalty, 0.1)

        return final_score
        
    def rerank_chunks(self, question, chunks):
        scored_chunks = []

        print(f"\n\n\n\n\n=== RERANKING {len(chunks)} CHUNKS ===")
        for i, chunk in enumerate(chunks):
            qa_score = self._question_answer_alignment(question, chunk.page_content)
            density_score = self._content_density(chunk.page_content)
            explanation_score = self._explanation_quality(chunk.page_content)

            total_score = (
                qa_score * 0.5 + 
                density_score * 0.3 + 
                explanation_score * 0.2
                )
            
            print(f"\n--- CHUNK {i+1} ---")
            print(f"Preview: {chunk.page_content[:150]}...")
            print(f"Scores: qa={qa_score:.3f} | density={density_score:.3f} | explanation={explanation_score:.3f}")
            print(f"Total: {total_score:.3f}")
            
            scored_chunks.append((total_score, chunk))

        ranked_chunks = [chunk for score, chunk in sorted(scored_chunks, key = lambda x: x[0], reverse=True)]
        return ranked_chunks

    def create_vector_store(self, chunks, collection_name="documents"):
        
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=collection_name,
            persist_directory='./chromadb'
        )

        return self.vector_store
    
    def search_similar(self, query, k = 5):
        if self.vector_store is None:
            raise ValueError(f"Vector Store not initialised. Call create_vector_store first")
    
        similar_docs = self.vector_store.similarity_search(query, k=k)
        return similar_docs
  
    def answer_question(self, question, k = 3):
        if self.vector_store is None:
            raise ValueError("Vector Store is not initialised. Call create_vector_store first")
        
        expanded_question = self.query_expander.expand_query(question)
        
        relevant_docs = self.search_similar(expanded_question, k * 3)    
        unique_docs = self.deduplicate_chunks(relevant_docs)[:k]
        reranked_docs = self.rerank_chunks(question, unique_docs)

        context = "\n\n".join(doc.page_content for doc in reranked_docs)

        prompt = self.prompt_template.format(context=context, question=question)
        answer = self.llm.invoke(prompt)

        return answer
