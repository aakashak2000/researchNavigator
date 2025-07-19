import os
import time
import arxiv
import requests
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Optional

from langchain_ollama import OllamaLLM

from src.rag.query_expander import QueryExpander

@dataclass
class PaperInfo:
    title: str
    authors: List[str]
    abstract: str
    arxiv_id: str
    url: str
    pdf_url: str
    published: datetime
    categories: List[str]
    filepath: Optional[str] = None

class ArxivDownloader:
    def __init__(self, download_dir = './papers', llm = None):
        self.download_dir = download_dir
        os.makedirs(download_dir, exist_ok=True)
        if llm:
            self.query_expander = QueryExpander(llm)
        else:
            self.query_expander = None

    def search_papers(self, query, max_results = 10):
        print(f"{query=}")
        client = arxiv.Client()
        search = arxiv.Search(
            query = query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
            sort_order=arxiv.SortOrder.Descending
        )

        papers = []
        for result in client.results(search):
            paper_info = PaperInfo(
                title = result.title,
                authors = [author.name for author in result.authors],
                abstract = result.summary,
                arxiv_id = result.entry_id.split('/')[-1],
                url = result.entry_id,
                pdf_url = result.pdf_url,
                published = result.published,
                categories = result.categories
            )
            papers.append(paper_info)

        print(f"Found {len(papers)} papers")
        return papers   
    
    def download_pdf(self, paper):
        safe_title = "".join(c for c in paper.title if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_title = safe_title[:50]
        filename = f"{paper.arxiv_id}_{safe_title}.pdf"
        file_path = os.path.join(self.download_dir, filename)

        if os.path.exists(file_path):
            print(f"Paper already exists!")
            paper.file_path = file_path
            return file_path
        
        try:
            response = requests.get(paper.pdf_url, stream=True)
            response.raise_for_status()


            with open(file_path, 'wb') as f:
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            print(f"Progress: {progress:.1f}%", flush = True)

                print()

            paper.file_path = file_path
            print(f"Successfully downloaded: {file_path}")
            return file_path
        
        except requests.RequestException as e:
            print(f"Failed to download {paper.title}: {e}")
            return None
        
    def search_and_download(self, query, max_results=10):

        if self.query_expander:
            expanded_query = self.query_expander.expand_query(query)
        else:
            expanded_query = query
        
        papers = self.search_papers(expanded_query, max_results)
        downloaded_papers = []
        for i, paper in enumerate(papers, 1):
            print(f"\n[{i}/{len(papers)}] Processing: {paper.title[:60]}...")
            
            if i > 1:
                time.sleep(1)

            file_path = self.download_pdf(paper)
            if file_path:
                downloaded_papers.append(paper)
            else:
                print(f"Skipping paper due to download failure")

        print(f"\n=== Download Complete: {len(downloaded_papers)}/{len(papers)} papers ===")
        return downloaded_papers