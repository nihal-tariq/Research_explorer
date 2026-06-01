"""FastAPI application for Graph RAG queries."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.query_engine import GraphRAGQueryEngine
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Graph RAG API", description="Query research paper knowledge graph")

# Initialize query engine once at startup
engine = None

class QueryRequest(BaseModel):
    question: str
    mode: str = "auto"  # "specific", "global", or "auto"

class QueryResponse(BaseModel):
    answer: str
    method_used: str

@app.on_event("startup")
async def startup_event():
    global engine
    engine = GraphRAGQueryEngine()
    logger.info("GraphRAGQueryEngine initialized")

@app.on_event("shutdown")
async def shutdown_event():
    if engine and engine.graph:
        engine.graph.close()

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Endpoint to ask questions to the Graph RAG system.
    
    - mode="specific": forces graph traversal (best for multi-hop facts)
    - mode="global": uses community summaries (best for broad themes)
    - mode="auto": auto-selects based on keywords
    """
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not ready")
    
    try:
        if request.mode == "specific":
            answer = engine.answer_specific_question(request.question)
            method = "cypher_graph"
        elif request.mode == "global":
            answer = engine.answer_global_question(request.question)
            method = "global_community"
        else:  # auto
            result = engine.hybrid_answer(request.question)
            answer = result["answer"]
            method = result["method"]
        
        return QueryResponse(answer=answer, method_used=method)
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}