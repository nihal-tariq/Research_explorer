"""Download arXiv papers, extract text, and chunk."""
import arxiv
import pdfplumber
from pathlib import Path
from typing import List, Dict
import logging
from src.config import PAPERS_DIR, TEXT_DIR, ARXIV_QUERY, MAX_PAPERS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def search_arxiv_papers(query: str, max_results: int = 30) -> List[Dict]:
    """
    Search arXiv and return metadata for papers.
    
    Args:
        query: arXiv search query
        max_results: Maximum number of papers to retrieve
    
    Returns:
        List of dicts with title, paper_id, pdf_url
    """
    client = arxiv.Client()
    search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
    papers = []
    for result in client.results(search):
        papers.append({
            "title": result.title,
            "paper_id": result.entry_id.split("/")[-1],
            "pdf_url": result.pdf_url,
            "summary": result.summary
        })
    logger.info(f"Found {len(papers)} papers")
    return papers

def download_pdf(paper_id: str, pdf_url: str) -> Path:
    """
    Download PDF to local disk if not already present.
    
    Args:
        paper_id: arXiv paper ID
        pdf_url: URL of PDF
    
    Returns:
        Path to downloaded PDF file
    """
    pdf_path = Path(PAPERS_DIR) / f"{paper_id}.pdf"
    if pdf_path.exists():
        logger.info(f"PDF already exists: {pdf_path}")
        return pdf_path
    
    import requests
    response = requests.get(pdf_url, stream=True)
    with open(pdf_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    logger.info(f"Downloaded {pdf_path}")
    return pdf_path

def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract all text from a PDF file using pdfplumber.
    
    Args:
        pdf_path: Path to PDF file
    
    Returns:
        Extracted text as string
    """
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def chunk_text(text: str, chunk_size: int = 1500, overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks for better entity extraction.
    
    Args:
        text: Full document text
        chunk_size: Approximate characters per chunk
        overlap: Overlap between chunks
    
    Returns:
        List of text chunks
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += (chunk_size - overlap)
    return chunks

def collect_papers() -> List[Dict]:
    """
    Main function: search, download, extract text, and chunk papers.
    
    Returns:
        List of dicts with paper_id, title, chunks (list of text chunks)
    """
    papers_meta = search_arxiv_papers(ARXIV_QUERY, MAX_PAPERS)
    papers_data = []
    
    for meta in papers_meta:
        paper_id = meta["paper_id"]
        pdf_path = download_pdf(paper_id, meta["pdf_url"])
        full_text = extract_text_from_pdf(pdf_path)
        if len(full_text.strip()) < 500:
            # Fallback to abstract if PDF extraction fails
            full_text = meta["summary"]
        chunks = chunk_text(full_text)
        papers_data.append({
            "paper_id": paper_id,
            "title": meta["title"],
            "chunks": chunks
        })
        logger.info(f"Processed {paper_id}: {len(chunks)} chunks")
    
    total_pages_estimate = sum(len(chunk_text(extract_text_from_pdf(Path(PAPERS_DIR)/f"{p['paper_id']}.pdf"))) // 500 for p in papers_meta)
    logger.info(f"Collected {len(papers_data)} papers, estimated > {total_pages_estimate} pages")
    return papers_data

if __name__ == "__main__":
    # quick test
    data = collect_papers()
    print(f"Collected {len(data)} papers")