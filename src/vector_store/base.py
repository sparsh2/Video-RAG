"""
Base classes for vector database operations.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np
from loguru import logger


class VectorStore(ABC):
    """Abstract base class for vector database operations."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the vector store.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.dimension = config.get("dimension", 768)  # Default dimension for Nomic embeddings
    
    @abstractmethod
    def connect(self) -> None:
        """Connect to the vector database."""
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the vector database."""
        pass
    
    @abstractmethod
    def create_collection(
        self,
        collection_name: str,
        dimension: Optional[int] = None,
        **kwargs,
    ) -> None:
        """
        Create a new collection in the vector database.
        
        Args:
            collection_name: Name of the collection
            dimension: Optional dimension of the vectors
            **kwargs: Additional collection parameters
        """
        pass
    
    @abstractmethod
    def insert(
        self,
        collection_name: str,
        vectors: np.ndarray,
        metadata: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> List[str]:
        """
        Insert vectors into the collection.
        
        Args:
            collection_name: Name of the collection
            vectors: Array of vectors to insert
            metadata: Optional list of metadata dictionaries
            **kwargs: Additional insert parameters
            
        Returns:
            List of inserted vector IDs
        """
        pass
    
    @abstractmethod
    def search(
        self,
        collection_name: str,
        query_vector: np.ndarray,
        limit: int = 10,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in the collection.
        
        Args:
            collection_name: Name of the collection
            query_vector: Query vector
            limit: Maximum number of results to return
            **kwargs: Additional search parameters
            
        Returns:
            List of search results with distances and metadata
        """
        pass
    
    @abstractmethod
    def delete(
        self,
        collection_name: str,
        vector_ids: List[str],
    ) -> None:
        """
        Delete vectors from the collection.
        
        Args:
            collection_name: Name of the collection
            vector_ids: List of vector IDs to delete
        """
        pass
    
    @abstractmethod
    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """
        Get statistics about a collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Dictionary containing collection statistics
        """
        pass 