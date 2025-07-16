import sys
sys.path.append('../.')
from src.rag.basic_rag import BasicRAG


file_path = sys.argv[-1]
rag = BasicRAG()
docs = rag.load_documents(file_path)
chunks = rag.split_documents(docs)
vector_store = rag.create_vector_store(chunks, "test")
answer = rag.answer_question("What is deep learning?")
print('\n=== FINAL ANSWER ===')
print(answer)

