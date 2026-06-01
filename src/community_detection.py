"""Detect communities in the graph and generate summaries."""
import logging
import networkx as nx
from typing import List, Dict, Any
from neo4j import GraphDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
from src.config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, GOOGLE_API_KEY

logger = logging.getLogger(__name__)

class CommunityDetector:
    """Perform community detection and summarization."""
    
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)
    
    def fetch_graph_to_networkx(self) -> nx.Graph:
        """
        Pull all nodes and edges from Neo4j and build a NetworkX graph.
        
        Returns:
            NetworkX graph with entity names as nodes.
        """
        query = """
        MATCH (n:Entity) RETURN n.name AS name
        """
        with self.driver.session() as session:
            nodes_result = session.run(query)
            node_names = [record["name"] for record in nodes_result]
            
            edges_query = """
            MATCH (s:Entity)-[r:RELATES]->(o:Entity)
            RETURN s.name AS source, o.name AS target
            """
            edges_result = session.run(edges_query)
            edges = [(record["source"], record["target"]) for record in edges_result]
        
        G = nx.Graph()
        G.add_nodes_from(node_names)
        G.add_edges_from(edges)
        logger.info(f"NetworkX graph built with {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G
    
    def detect_communities(self, G: nx.Graph) -> List[List[str]]:
        """
        Run Louvain community detection.
        
        Args:
            G: NetworkX graph
        
        Returns:
            List of communities, each a list of node names.
        """
        import networkx.algorithms.community as nx_comm
        # Louvain returns a list of frozensets
        communities = nx_comm.louvain_communities(G, seed=42)
        communities = [list(comm) for comm in communities]
        logger.info(f"Detected {len(communities)} communities")
        return communities
    
    def generate_community_summary(self, community_nodes: List[str]) -> str:
        """
        Generate a text summary for a community using Gemini.
        
        Args:
            community_nodes: List of entity names in the community
        
        Returns:
            Natural language summary of the community's theme.
        """
        prompt = f"""
        The following entities form a densely connected group in a knowledge graph of research papers.
        Entities: {', '.join(community_nodes[:30])}  # limit to avoid token overflow
        
        Based on these entities, what is the common research theme or topic of this community?
        Write a concise paragraph (3-5 sentences).
        """
        response = self.llm.invoke(prompt)
        return response.content
    
    def store_communities_in_neo4j(self, communities: List[List[str]], summaries: List[str]):
        """
        Store communities as Community nodes linked to Entity nodes.
        
        Args:
            communities: List of node lists
            summaries: Corresponding summary strings
        """
        with self.driver.session() as session:
            for idx, (community_nodes, summary) in enumerate(zip(communities, summaries)):
                # Create Community node
                session.run(
                    """
                    CREATE (c:Community {id: $id, summary: $summary})
                    """,
                    id=f"comm_{idx}", summary=summary
                )
                # Link each entity to the community
                for node_name in community_nodes:
                    session.run(
                        """
                        MATCH (e:Entity {name: $name})
                        MATCH (c:Community {id: $cid})
                        MERGE (e)-[:BELONGS_TO]->(c)
                        """,
                        name=node_name, cid=f"comm_{idx}"
                    )
        logger.info(f"Stored {len(communities)} communities")
    
    def run_pipeline(self):
        """Full community detection and storage."""
        G = self.fetch_graph_to_networkx()
        if G.number_of_nodes() == 0:
            logger.warning("Graph is empty, skipping community detection")
            return
        communities = self.detect_communities(G)
        summaries = []
        for i, comm in enumerate(communities):
            logger.info(f"Generating summary for community {i+1}/{len(communities)}")
            summaries.append(self.generate_community_summary(comm))
        self.store_communities_in_neo4j(communities, summaries)
    
    def close(self):
        self.driver.close()