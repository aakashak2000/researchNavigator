import sys
sys.path.append('../.')
from src.paper_downloader import ArxivDownloader
downloader = ArxivDownloader()
papers = downloader.search_and_download('attention mechanism', max_results=10)
print(f'\nDownloaded papers:')
for paper in papers:
    print(f'- {paper.title}')
    print(f'  File: {paper.file_path}')