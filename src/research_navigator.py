import os

from datetime import datetime, timedelta

from typing import List, Dict
from src.config import Config
from src.rag.basic_rag import BasicRAG
from src.paper_downloader import ArxivDownloader, PaperInfo

class ResearchNavigator:
    def __init__(self, config=None):
        self.config = config or Config()
        self.downloader = ArxivDownloader(download_dir='./papers')
        self.rag = BasicRAG(
            chunk_size = self.config.chunk_size if hasattr(self.config, 'chunk_size') else 1000,
            chunk_overlap = 200
        )
        self.processed_topics = {}

    def research_topic(self, query, max_papers, force_refresh=False):
        needs_refresh = self._needs_topic_refresh(query, force_refresh)

        if needs_refresh:
            papers = self.downloader.search_and_download(query=query, max_results=max_papers)
            if not papers:
                return "No papers found"
            
            self._process_papers_into_rag(papers, query)
            self.processed_topics[query] = datetime.now()

        else:
            print(f"Using cached samples")

        return f"Successfully processed documents - ready for questions!"
    
    def _process_papers_into_rag(self, papers, query):
        all_chunks = []
        processed_count = 0

        for paper in papers:
            if paper.file_path and os.path.exists(paper.file_path):
                try:
                    docs = self.rag.load_documents(paper.file_path)
                    chunks = self.rag.split_documents(docs)

                    for chunk in chunks:
                        chunk.metadata.update({
                            'source_paper': paper.title,
                            'arxiv_id': paper.arxiv_id,
                            'query_topic': query,
                            'authors': ", ".join(paper.authors[:5])
                        })

                    all_chunks.extend(chunks)
                    processed_count += 1
                except Exception as e:
                    print(f"Failed to process {paper.title}- {e}")

        if all_chunks:
            collection_name = f"research_{query.replace(' ', '-')}"
            self.rag.create_vector_store(all_chunks, collection_name=collection_name)
            print(f"Successfully processed {processed_count} papers")

        else:
            print(f"No papers could be processed")

    def _needs_topic_refresh(self, query, force_refresh):
        if force_refresh:
            print(f"==========Force Refresh Requested==========")
            return True

        if query not in self.processed_topics:
            print(f"==========New Topic: Needs Processing==========")
            return True
        
        last_processed = self.processed_topics[query]
        age = datetime.now() - last_processed
        refresh_threshold = timedelta(self.config.refresh_threshold_days)

        if age > refresh_threshold:
            print(f"Topic is {age.days} days old - needs refresh")

        print(f"Topic is fresh - ({age.days} days old)")
        return False
    
    def ask_question(self, question: str) -> str:
        if self.rag.vector_store is None:
            return "No papers processed yet. Run research_topic() first."
        answer = self.rag.answer_question(question, k=5)
        
        return answer