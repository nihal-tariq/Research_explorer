# Research Explorer - Graph RAG System

A FastAPI-based Knowledge Graph Retrieval Augmented Generation (Graph RAG) system. This project retrieves research papers from arXiv, extracts entities and relationships to build a Neo4j knowledge graph using Gemini, performs community detection to summarize broad themes, and serves an API to query the graph.

## Features
- **Data Collection:** Automated fetching of PDFs from arXiv based on topic queries and text chunking.
- **Knowledge Graph Construction:** Uses Google Gemini to extract entities (Papers, Authors, Methods) and relationships from the text and loads them into Neo4j.
- **Community Detection:** Uses Louvain algorithms on the graph to find clusters of research themes and summarizes them.
- **FastAPI Endpoint:** A smart query router that supports specific multi-hop reasoning (Cypher queries) and global/broad questions (Community summaries).

---

## Complete Startup Guide

### 1. Prerequisites
You will need the following installed on your machine:
- Python 3.9+
- A running instance of **Neo4j** (local or AuraDB).
- A valid **Google Gemini API Key**.

### 2. Environment Variables
Create a `.env` file in the root of the repository with the following keys:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password
```

### 3. Virtual Environment Setup
A virtual environment ensures all dependencies are kept separate from your system Python. 

**Activate the virtual environment:**
```bash
# On Mac/Linux:
source venv/bin/activate
```
*(You will know it is activated when you see `(venv)` at the beginning of your terminal prompt. You must run this command every time you open a new terminal window).*

**Install dependencies (if not already installed):**
```bash
pip install -r requirements.txt
```

### 4. Running the Indexing Pipeline
Before you can query the system, you must populate the Neo4j database. Ensure your Neo4j database is running and credentials are correct in your `.env`.

Run the full indexing pipeline:
```bash
python scripts/run_indexing.py
```
*This step takes a while! It will download papers, use Gemini to parse entities (this requires API usage), and run community detection.*

### 5. Starting the API Server
Once the data is indexed, start the FastAPI server:
```bash
uvicorn src.main:app --reload
```

### 6. Using the API
The API exposes a `/query` endpoint.

**Example Request:**
```bash
curl -X POST "http://127.0.0.1:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"question": "What are the main research trends?", "mode": "auto"}'
```

**Query Modes:**
- `specific`: Forces a graph traversal query (Best for: *Which papers cite X?*).
- `global`: Forces a community summary retrieval (Best for: *What are the broad themes?*).
- `auto`: Automatically decides which mode to use based on the question.
