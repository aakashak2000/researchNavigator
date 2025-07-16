import os
import time
import arxiv
import requests
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Optional

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
    def __init__(self, download_dir = './papers'):
        self.download_dir = download_dir
        os.makedirs(download_dir, exist_ok=True)

    def search_papers(self, query, max_results = 10):
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
                authors = [name for name in result.authors],
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