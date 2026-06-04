"""Build knowledge graph: entity/relation extraction and Neo4j insertion."""
import logging
from typing import List
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
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
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.0
        )
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-004",
            google_api_key=GOOGLE_API_KEY
        )
        self.neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        self.extractor = self.llm.with_structured_output(GraphTriples)

    def extract_from_chunk(self, chunk_text: str, paper_title: str) -> GraphTriples:
        prompt = f"""
        You are extracting knowledge from a research paper chunk. Paper title: {paper_title}
        
        Text chunk:
        {chunk_text}
        
        Extract:
        - Entities: unique names of papers, authors, models, datasets, technical concepts.
        - Relationships: connections like CITES (paper->paper), USES (paper->dataset), 
          AUTHORS (paper->author), IMPROVES_UPON (paper->model), COMPARED_IN (paper->benchmark).
        
        Return as structured data.
        """
        try:
            result = self.extractor.invoke(prompt)
            return result
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            return GraphTriples(nodes=[], relationships=[])

    def store_triples(self, triples: GraphTriples):
        with self.neo4j_driver.session() as session:
            if triples.nodes:
                node_names = [node.name for node in triples.nodes]
                embeddings = self.embeddings.embed_documents(node_names)
                for node, embedding in zip(triples.nodes, embeddings):
                    session.run(
                        """
                        MERGE (n:Entity {name: $name})
                        SET n.type = $type, n.embedding = $embedding
                        """,
                        name=node.name, type=node.type, embedding=embedding
                    )
            for rel in triples.relationships:
                # Sanitize predicate to be a valid Cypher relationship type
                predicate_type = ''.join(c for c in rel.predicate if c.isalnum() or c == '_').upper()
                if not predicate_type:
                    predicate_type = "RELATES"
                    
                query = f"""
                MATCH (s:Entity {{name: $subj}})
                MATCH (o:Entity {{name: $obj}})
                MERGE (s)-[r:{predicate_type}]->(o)
                SET r.predicate = $predicate
                """
                session.run(query, subj=rel.subject, obj=rel.object, predicate=rel.predicate)
        logger.info(f"Stored {len(triples.nodes)} nodes and {len(triples.relationships)} relationships")

    def add_paper_node(self, paper_title: str, paper_id: str):
        with self.neo4j_driver.session() as session:
            session.run(
                "MERGE (p:Paper {id: $id, title: $title})",
                id=paper_id, title=paper_title
            )

    def close(self):
        self.neo4j_driver.close()
