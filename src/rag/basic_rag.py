from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.vectorstores import VectorStore
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from typing import List
import chromadb
import os

class BasicRAG:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

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
    
    def split_documents(self, documents):
        chunks = self.text_splitter.split_documents(documents)
        print(f"Split into {len(chunks)} chunks")
        return chunks

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
        
        print(f"Question: {question}")

        relevant_docs = self.search_similar(question, k)    
        context = "\n\n".join(doc.page_content for doc in relevant_docs)

        prompt = self.prompt_template.format(context=context, question=question)
        answer = self.llm.invoke(prompt)

        return answer
