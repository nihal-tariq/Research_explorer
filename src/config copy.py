"""Configuration and environment variables."""
import os
from dotenv import load_dotenv

load_dotenv()

# Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Neo4j
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Paths
DATA_DIR = "data"
PAPERS_DIR = os.path.join(DATA_DIR, "pdfs")
TEXT_DIR = os.path.join(DATA_DIR, "text")

# Ensure directories exist
os.makedirs(PAPERS_DIR, exist_ok=True)
os.makedirs(TEXT_DIR, exist_ok=True)

# arXiv query
ARXIV_QUERY = "ti:(retrieval augmented generation) OR ti:(knowledge graph) OR ti:(RAG) AND cat:cs.AI"
MAX_PAPERS = 30   