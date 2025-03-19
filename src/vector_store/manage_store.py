"""
Main script for managing the vector store.
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from loguru import logger

from src.utils.config import load_config, setup_logging
from .milvus import MilvusStore


def create_collection(
    collection_name: str,
    config_path: str = "config/default.yaml",
    dimension: Optional[int] = None,
) -> None:
    """
    Create a new collection in the vector store.
    
    Args:
        collection_name: Name of the collection
        config_path: Path to the configuration file
        dimension: Optional dimension of the vectors
    """
    # Load configuration
    config = load_config(config_path)
    
    # Set up logging
    setup_logging(config.logging)
    
    # Initialize vector store
    store = MilvusStore(config.vector_store)
    store.connect()
    
    try:
        # Create collection
        store.create_collection(collection_name, dimension)
        
        # Get collection stats
        stats = store.get_collection_stats(collection_name)
        logger.info(f"Collection stats: {json.dumps(stats, indent=2)}")
    finally:
        store.disconnect()


def insert_vectors(
    collection_name: str,
    vectors_path: str,
    metadata_path: Optional[str] = None,
    config_path: str = "config/default.yaml",
) -> None:
    """
    Insert vectors into a collection.
    
    Args:
        collection_name: Name of the collection
        vectors_path: Path to the vectors file (numpy .npz)
        metadata_path: Optional path to the metadata file (JSON)
        config_path: Path to the configuration file
    """
    # Load configuration
    config = load_config(config_path)
    
    # Set up logging
    setup_logging(config.logging)
    
    # Load vectors
    vectors_data = np.load(vectors_path)
    vectors = vectors_data["embeddings"]
    
    # Load metadata if provided
    metadata = None
    if metadata_path:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    
    # Initialize vector store
    store = MilvusStore(config.vector_store)
    store.connect()
    
    try:
        # Insert vectors
        vector_ids = store.insert(collection_name, vectors, metadata)
        logger.info(f"Inserted {len(vector_ids)} vectors")
    finally:
        store.disconnect()


def search_vectors(
    collection_name: str,
    query_vector_path: str,
    limit: int = 10,
    config_path: str = "config/default.yaml",
) -> None:
    """
    Search for similar vectors in a collection.
    
    Args:
        collection_name: Name of the collection
        query_vector_path: Path to the query vector file (numpy .npz)
        limit: Maximum number of results to return
        config_path: Path to the configuration file
    """
    # Load configuration
    config = load_config(config_path)
    
    # Set up logging
    setup_logging(config.logging)
    
    # Load query vector
    query_data = np.load(query_vector_path)
    query_vector = query_data["embeddings"][0]  # Use first vector as query
    
    # Initialize vector store
    store = MilvusStore(config.vector_store)
    store.connect()
    
    try:
        # Search vectors
        results = store.search(collection_name, query_vector, limit)
        
        # Print results
        print("\nSearch Results:")
        print("-" * 50)
        for i, result in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"ID: {result['id']}")
            print(f"Distance: {result['distance']:.6f}")
            print(f"Metadata: {json.dumps(result['metadata'], indent=2)}")
    finally:
        store.disconnect()


def delete_vectors(
    collection_name: str,
    vector_ids: List[str],
    config_path: str = "config/default.yaml",
) -> None:
    """
    Delete vectors from a collection.
    
    Args:
        collection_name: Name of the collection
        vector_ids: List of vector IDs to delete
        config_path: Path to the configuration file
    """
    # Load configuration
    config = load_config(config_path)
    
    # Set up logging
    setup_logging(config.logging)
    
    # Initialize vector store
    store = MilvusStore(config.vector_store)
    store.connect()
    
    try:
        # Delete vectors
        store.delete(collection_name, vector_ids)
    finally:
        store.disconnect()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Manage vector store")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Create collection command
    create_parser = subparsers.add_parser("create", help="Create a new collection")
    create_parser.add_argument("collection_name", help="Name of the collection")
    create_parser.add_argument("--config", default="config/default.yaml", help="Path to configuration file")
    create_parser.add_argument("--dimension", type=int, help="Dimension of the vectors")
    
    # Insert vectors command
    insert_parser = subparsers.add_parser("insert", help="Insert vectors into a collection")
    insert_parser.add_argument("collection_name", help="Name of the collection")
    insert_parser.add_argument("vectors_path", help="Path to the vectors file")
    insert_parser.add_argument("--metadata", help="Path to the metadata file")
    insert_parser.add_argument("--config", default="config/default.yaml", help="Path to configuration file")
    
    # Search vectors command
    search_parser = subparsers.add_parser("search", help="Search for similar vectors")
    search_parser.add_argument("collection_name", help="Name of the collection")
    search_parser.add_argument("query_vector_path", help="Path to the query vector file")
    search_parser.add_argument("--limit", type=int, default=10, help="Maximum number of results")
    search_parser.add_argument("--config", default="config/default.yaml", help="Path to configuration file")
    
    # Delete vectors command
    delete_parser = subparsers.add_parser("delete", help="Delete vectors from a collection")
    delete_parser.add_argument("collection_name", help="Name of the collection")
    delete_parser.add_argument("vector_ids", nargs="+", help="List of vector IDs to delete")
    delete_parser.add_argument("--config", default="config/default.yaml", help="Path to configuration file")
    
    args = parser.parse_args()
    
    if args.command == "create":
        create_collection(args.collection_name, args.config, args.dimension)
    elif args.command == "insert":
        insert_vectors(args.collection_name, args.vectors_path, args.metadata, args.config)
    elif args.command == "search":
        search_vectors(args.collection_name, args.query_vector_path, args.limit, args.config)
    elif args.command == "delete":
        delete_vectors(args.collection_name, args.vector_ids, args.config)
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 