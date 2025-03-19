"""
Main script for running the RAG pipeline.
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

from src.utils.config import load_config, setup_logging
from .pipeline import NomicRAGPipeline


def add_content(
    content_path: str,
    metadata_path: Optional[str] = None,
    modality: str = "text",
    config_path: str = "config/default.yaml",
) -> None:
    """
    Add content to the knowledge base.
    
    Args:
        content_path: Path to the content file or directory
        metadata_path: Optional path to the metadata file
        modality: Content modality ("text" or "image")
        config_path: Path to the configuration file
    """
    # Load configuration
    config = load_config(config_path)
    
    # Set up logging
    setup_logging(config.logging)
    
    # Initialize pipeline
    pipeline = NomicRAGPipeline(config)
    
    try:
        # Load content
        content_path = Path(content_path)
        if content_path.is_file():
            if modality == "text":
                with open(content_path, "r", encoding="utf-8") as f:
                    content = f.read()
            else:
                content = str(content_path)
        else:
            if modality == "text":
                content = []
                for file in content_path.glob("*.txt"):
                    with open(file, "r", encoding="utf-8") as f:
                        content.append(f.read())
            else:
                content = [str(f) for f in content_path.glob("*.jpg")] + [str(f) for f in content_path.glob("*.png")]
        
        # Load metadata if provided
        metadata = None
        if metadata_path:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        
        # Add to knowledge base
        content_ids = pipeline.add_to_knowledge_base(content, metadata, modality)
        logger.info(f"Added {len(content_ids)} items to knowledge base")
    finally:
        del pipeline


def query(
    query_text: str,
    modality: str = "text",
    top_k: int = 5,
    config_path: str = "config/default.yaml",
) -> None:
    """
    Process a query and generate a response.
    
    Args:
        query_text: Query text or image path
        modality: Query modality ("text" or "image")
        top_k: Number of results to return
        config_path: Path to the configuration file
    """
    # Load configuration
    config = load_config(config_path)
    
    # Set up logging
    setup_logging(config.logging)
    
    # Initialize pipeline
    pipeline = NomicRAGPipeline(config)
    
    try:
        # Process query
        retrieved_info = pipeline.process_query(query_text, modality, top_k)
        
        # Generate response
        response = pipeline.generate_response(query_text, retrieved_info)
        
        # Print results
        print("\nQuery Results:")
        print("-" * 50)
        print(f"Query: {query_text}")
        print(f"Modality: {modality}")
        print("\nRetrieved Information:")
        for i, result in enumerate(retrieved_info["results"], 1):
            print(f"\nResult {i}:")
            print(f"Distance: {result['distance']:.6f}")
            print(f"Metadata: {json.dumps(result['metadata'], indent=2)}")
        print("\nGenerated Response:")
        print(response)
    finally:
        del pipeline


def remove_content(
    content_ids: List[str],
    config_path: str = "config/default.yaml",
) -> None:
    """
    Remove content from the knowledge base.
    
    Args:
        content_ids: List of content IDs to remove
        config_path: Path to the configuration file
    """
    # Load configuration
    config = load_config(config_path)
    
    # Set up logging
    setup_logging(config.logging)
    
    # Initialize pipeline
    pipeline = NomicRAGPipeline(config)
    
    try:
        # Remove from knowledge base
        pipeline.remove_from_knowledge_base(content_ids)
    finally:
        del pipeline


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run RAG pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Add content command
    add_parser = subparsers.add_parser("add", help="Add content to knowledge base")
    add_parser.add_argument("content_path", help="Path to content file or directory")
    add_parser.add_argument("--metadata", help="Path to metadata file")
    add_parser.add_argument("--modality", choices=["text", "image"], default="text", help="Content modality")
    add_parser.add_argument("--config", default="config/default.yaml", help="Path to configuration file")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Process a query")
    query_parser.add_argument("query_text", help="Query text or image path")
    query_parser.add_argument("--modality", choices=["text", "image"], default="text", help="Query modality")
    query_parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    query_parser.add_argument("--config", default="config/default.yaml", help="Path to configuration file")
    
    # Remove content command
    remove_parser = subparsers.add_parser("remove", help="Remove content from knowledge base")
    remove_parser.add_argument("content_ids", nargs="+", help="List of content IDs to remove")
    remove_parser.add_argument("--config", default="config/default.yaml", help="Path to configuration file")
    
    args = parser.parse_args()
    
    if args.command == "add":
        add_content(args.content_path, args.metadata, args.modality, args.config)
    elif args.command == "query":
        query(args.query_text, args.modality, args.top_k, args.config)
    elif args.command == "remove":
        remove_content(args.content_ids, args.config)
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 