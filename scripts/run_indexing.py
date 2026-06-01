
"""Complete indexing pipeline: collect data, extract graph, build communities."""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_collection import collect_papers
from src.graph_builder import GraphBuilder
from src.community_detection import CommunityDetector
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Step 1: Collecting papers (100+ pages)...")
    papers = collect_papers()
    
    logger.info("Step 2: Building knowledge graph with Gemini...")
    graph_builder = GraphBuilder()
    
    for paper in papers:
        paper_title = paper["title"]
        paper_id = paper["paper_id"]
        graph_builder.add_paper_node(paper_title, paper_id)
        
        for chunk in paper["chunks"]:
            triples = graph_builder.extract_from_chunk(chunk, paper_title)
            graph_builder.store_triples(triples)
            time.sleep(1)  # avoid Gemini rate limits
        logger.info(f"Finished paper {paper_id}")
    
    graph_builder.close()
    
    logger.info("Step 3: Community detection and summarization...")
    community_detector = CommunityDetector()
    community_detector.run_pipeline()
    community_detector.close()
    
    logger.info("Indexing complete!")

if __name__ == "__main__":
    main()
