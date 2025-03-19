"""
Base classes for the RAG pipeline.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np
from loguru import logger


class RAGPipeline(ABC):
    """Abstract base class for RAG pipeline."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the RAG pipeline.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
    
    @abstractmethod
    def process_query(
        self,
        query: Union[str, np.ndarray],
        modality: str = "text",
        top_k: int = 5,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Process a query and retrieve relevant information.
        
        Args:
            query: Query text or embedding
            modality: Query modality ("text" or "image")
            top_k: Number of results to return
            **kwargs: Additional query parameters
            
        Returns:
            Dictionary containing query results
        """
        pass
    
    @abstractmethod
    def generate_response(
        self,
        query: str,
        retrieved_info: Dict[str, Any],
        **kwargs,
    ) -> str:
        """
        Generate a response based on the query and retrieved information.
        
        Args:
            query: Original query text
            retrieved_info: Retrieved information from the vector store
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response text
        """
        pass
    
    @abstractmethod
    def add_to_knowledge_base(
        self,
        content: Union[str, List[str], np.ndarray, List[np.ndarray]],
        metadata: Optional[List[Dict[str, Any]]] = None,
        modality: str = "text",
        **kwargs,
    ) -> List[str]:
        """
        Add content to the knowledge base.
        
        Args:
            content: Content to add (text or embeddings)
            metadata: Optional metadata for the content
            modality: Content modality ("text" or "image")
            **kwargs: Additional parameters
            
        Returns:
            List of added content IDs
        """
        pass
    
    @abstractmethod
    def remove_from_knowledge_base(
        self,
        content_ids: List[str],
        **kwargs,
    ) -> None:
        """
        Remove content from the knowledge base.
        
        Args:
            content_ids: List of content IDs to remove
            **kwargs: Additional parameters
        """
        pass


class MultimodalRAGPipeline(RAGPipeline):
    """Base class for multimodal RAG pipeline."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the multimodal RAG pipeline.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.text_collection = config.get("text_collection", "text_embeddings")
        self.image_collection = config.get("image_collection", "image_embeddings")
    
    def process_query(
        self,
        query: Union[str, np.ndarray],
        modality: str = "text",
        top_k: int = 5,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Process a query and retrieve relevant information.
        
        Args:
            query: Query text or embedding
            modality: Query modality ("text" or "image")
            top_k: Number of results to return
            **kwargs: Additional query parameters
            
        Returns:
            Dictionary containing query results
        """
        # Select collection based on modality
        collection = self.text_collection if modality == "text" else self.image_collection
        
        # Get query embedding if text
        if modality == "text" and isinstance(query, str):
            query_embedding = self._get_text_embedding(query)
        elif modality == "image" and isinstance(query, str):
            query_embedding = self._get_image_embedding(query)
        else:
            query_embedding = query
        
        # Search vector store
        results = self._search_vector_store(collection, query_embedding, top_k)
        
        return {
            "query": query,
            "modality": modality,
            "results": results,
        }
    
    @abstractmethod
    def _get_text_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for text.
        
        Args:
            text: Input text
            
        Returns:
            Text embedding
        """
        pass
    
    @abstractmethod
    def _get_image_embedding(self, image_path: str) -> np.ndarray:
        """
        Get embedding for image.
        
        Args:
            image_path: Path to the image
            
        Returns:
            Image embedding
        """
        pass
    
    @abstractmethod
    def _search_vector_store(
        self,
        collection: str,
        query_embedding: np.ndarray,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """
        Search the vector store.
        
        Args:
            collection: Collection name
            query_embedding: Query embedding
            top_k: Number of results to return
            
        Returns:
            List of search results
        """
        pass 