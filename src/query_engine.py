"""Query engine: answer questions using graph and community summaries."""
import logging
from typing import List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain.schema import Document
from src.config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, GOOGLE_API_KEY
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class GraphRAGQueryEngine:
    """Handles both specific (Cypher) and global (community) queries."""
    
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-004", google_api_key=GOOGLE_API_KEY)
        
        # LangChain wrapper for Neo4j
        self.graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)
        
        # Cypher QA chain for specific questions
        self.cypher_chain = GraphCypherQAChain.from_llm(
            llm=self.llm,
            graph=self.graph,
            verbose=True,
            allow_dangerous_requests=True
        )
    
    def answer_specific_question(self, question: str) -> str:
        """
        Answer a question that requires graph traversal (multi-hop).
        Example: "Which papers cite Graph RAG paper and also use MS MARCO?"
        
        Args:
            question: Natural language question
        
        Returns:
            Answer string
        """
        return self.cypher_chain.run(question)
    
    def retrieve_community_summaries(self, question: str, top_k: int = 3) -> List[str]:
        """
        Retrieve the most relevant community summaries for a global question.
        
        Args:
            question: Global thematic question
            top_k: Number of communities to retrieve
        
        Returns:
            List of summary texts
        """
        # Get all community summaries from Neo4j
        query = "MATCH (c:Community) RETURN c.id AS id, c.summary AS summary"
        results = self.graph.query(query)
        if not results:
            return []
        
        summaries = [r["summary"] for r in results if r["summary"]]
        if not summaries:
            return []
        
        # Embed question and summaries
        q_embed = self.embeddings.embed_query(question)
        summary_embeds = [self.embeddings.embed_query(s) for s in summaries]
        
        # Compute similarities
        similarities = cosine_similarity([q_embed], summary_embeds)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [summaries[i] for i in top_indices]
    
    def answer_global_question(self, question: str) -> str:
        """
        Answer a broad, thematic question using community summaries.
        
        Args:
            question: e.g., "What are the main research trends in these papers?"
        
        Returns:
            Answer synthesized from communities
        """
        summaries = self.retrieve_community_summaries(question)
        if not summaries:
            return "No community summaries available yet. Please run indexing first."
        
        context = "\n\n".join([f"Community {i+1}: {s}" for i, s in enumerate(summaries)])
        prompt = f"""
        You are a research analyst. Based on the following community summaries from a knowledge graph of papers:
        {context}
        
        Answer the question: {question}
        Provide a concise, informative answer.
        """
        response = self.llm.invoke(prompt)
        return response.content
    
    def hybrid_answer(self, question: str) -> Dict[str, Any]:
        """
        Attempt specific answer first, if fails or is too broad, fallback to global.
        
        Args:
            question: User query
        
        Returns:
            Dict with answer and method used.
        """
        # Simple heuristic: if question contains words like "list", "summarize", "themes" -> global
        global_keywords = ["summarize", "themes", "trends", "overview", "main topics", "research directions"]
        if any(kw in question.lower() for kw in global_keywords):
            answer = self.answer_global_question(question)
            return {"answer": answer, "method": "global_community"}
        else:
            try:
                answer = self.answer_specific_question(question)
                return {"answer": answer, "method": "cypher_graph"}
            except Exception as e:
                logger.error(f"Cypher failed: {e}, falling back to global")
                answer = self.answer_global_question(question)
                return {"answer": answer, "method": "fallback_global"}