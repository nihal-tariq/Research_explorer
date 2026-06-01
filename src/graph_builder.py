"""Build knowledge graph: entity/relation extraction and Neo4j insertion."""
import logging
from typing import List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from neo4j import GraphDatabase
from src.config import GOOGLE_API_KEY, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for structured extraction
class Node(BaseModel):
    name: str = Field(description="Entity name (e.g., 'BERT', 'Graph RAG', 'John Doe')")
    type: str = Field(description="Entity type: Paper, Author, Model, Dataset, Concept")

class Relationship(BaseModel):
    subject: str = Field(description="Subject entity name")
    predicate: str = Field(description="Relation like CITES, USES, AUTHORS, IMPROVES")
    object: str = Field(description="Object entity name")

class GraphTriples(BaseModel):
    nodes: List[Node] = Field(description="All unique entities in this chunk")
    relationships: List[Relationship] = Field(description="Relationships between entities")

class GraphBuilder:
    """Handles entity extraction, embedding, and Neo4j storage."""
    
    def __init__(self):
        """Initialize LLM, embeddings, and Neo4j driver."""
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.0,
            convert_system_message_to_human=True
        )
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-004",
            google_api_key=GOOGLE_API_KEY
        )
        self.neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        
        # Bind structured output
        self.extractor = self.llm.with_structured_output(GraphTriples)
        
    def extract_from_chunk(self, chunk_text: str, paper_title: str) -> GraphTriples:
        """
        Extract entities and relationships from a text chunk using Gemini.
        
        Args:
            chunk_text: One chunk of paper text
            paper_title: Title of the paper (for context)
        
        Returns:
            GraphTriples object with nodes and relationships
        """
        prompt = f"""
        You are extracting knowledge from a research paper chunk. Paper title: {paper_title}
        
        Text chunk:
        {chunk_text}
        
        Extract:
        - Entities: unique names of papers, authors, models, datasets, technical concepts.
        - Relationships: connections like CITES (paper->paper), USES (paper->dataset), AUTHORS (paper->author), IMPROVES_UPON (paper->model), COMPARED_IN (paper->benchmark).
        
        Return as structured data.
        """
        try:
            result = self.extractor.invoke(prompt)
            return result
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            return GraphTriples(nodes=[], relationships=[])
    
    def store_triples(self, triples: GraphTriples):
        """
        Store nodes and relationships into Neo4j using Cypher MERGE.
        
        Args:
            triples: GraphTriples object
        """
        with self.neo4j_driver.session() as session:
            # Create nodes (with embedding)
            for node in triples.nodes:
                # Generate embedding for the node name (for later similarity search)
                embedding = self.embeddings.embed_query(node.name)
                session.run(
                    """
                    MERGE (n:Entity {name: $name})
                    SET n.type = $type,
                        n.embedding = $embedding
                    """,
                    name=node.name, type=node.type, embedding=embedding
                )
            
            # Create relationships
            for rel in triples.relationships:
                session.run(
                    """
                    MATCH (s:Entity {name: $subj})
                    MATCH (o:Entity {name: $obj})
                    MERGE (s)-[r:RELATES {type: $predicate}]->(o)
                    SET r.predicate = $predicate
                    """,
                    subj=rel.subject, obj=rel.object, predicate=rel.predicate
                )
        logger.info(f"Stored {len(triples.nodes)} nodes and {len(triples.relationships)} relationships")
    
    def add_paper_node(self, paper_title: str, paper_id: str):
        """Create a Paper node for each document."""
        with self.neo4j_driver.session() as session:
            session.run(
                """
                MERGE (p:Paper {id: $id, title: $title})
                """,
                id=paper_id, title=paper_title
            )
    
    def close(self):
        self.neo4j_driver.close()