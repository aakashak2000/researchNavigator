from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum
from typing import List, Optional


class Config(BaseModel):
    llm_model: str = "llama3.2:3b"
    temperature: float = 0.1
    vector_db: str = "./chroma_db"
    collection_name: str = "research_docs"

    paper_cache_days: int = Field(default=45, description="How long to keep papers in cache")
    max_papers_per_topic: int = Field(default=50, description="Maximum papers to store per topic")
    refresh_threshold_days: int = Field(default=15, description="When to fetch new papers for existing topics")

    enable_topic_persistence: bool = True
    auto_cleanup: bool = True

class PaperSource(str, Enum):
    ARXIV = "arxiv"
    PUBMED = "pubmed"
    SEMANTIC_SCHOLAR = "semantic_scholar"
    MANUAL_UPLOAD = "manual_upload"

class PaperMetadata(BaseModel):
    paper_id: str = Field(..., description="Unique Identifier for the Paper")
    title: str = Field(..., description="Paper Title")
    query_topic: str = Field(..., description="The topic/query the paper was downloaded for")
    source: PaperSource = Field(..., description="Where the paper came from")
    
    download_timestamp: datetime = Field(default_factory=datetime.now, description="When the paper was downloaded")
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0, description="How relevant is the paper to the query (0-1)")
    authors: List[str] = Field(default_factory=list, description="List of paper authors")
    abstract: Optional[str] = Field(None, description="Paper Abstract")
    url: Optional[str] = Field(None, description="Original Paper URL")
    file_path: Optional[str] = Field(None, description="Local Path to downloaded paper")
