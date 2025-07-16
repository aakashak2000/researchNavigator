import sys
sys.path.append('../.')
from src.paper_downloader import ArxivDownloader
downloader = ArxivDownloader()
papers = downloader.search_papers('cash flow forecasting', max_results=3)
print([paper.title for paper in papers])
print(f'\nFirst paper:')
print(f'Title: {papers[0].title}')
print(f'Authors: {papers[0].authors[:2]}')  # First 2 authors
print(f'Abstract: {papers[0].abstract[:150]}...')