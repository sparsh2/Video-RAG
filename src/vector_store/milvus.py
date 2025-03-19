"""
Milvus vector store implementation.
"""
from typing import Any, Dict, List, Optional, Union

import numpy as np
from loguru import logger
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

from .base import VectorStore


class MilvusStore(VectorStore):
    """Milvus vector store implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Milvus vector store.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.host = config.get("host", "localhost")
        self.port = config.get("port", 19530)
        self.alias = config.get("alias", "default")
        self._connection = None
    
    def connect(self) -> None:
        """Connect to the Milvus server."""
        try:
            connections.connect(
                alias=self.alias,
                host=self.host,
                port=self.port,
            )
            logger.info(f"Connected to Milvus server at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus server: {e}")
            raise
    
    def disconnect(self) -> None:
        """Disconnect from the Milvus server."""
        try:
            connections.disconnect(self.alias)
            logger.info("Disconnected from Milvus server")
        except Exception as e:
            logger.error(f"Failed to disconnect from Milvus server: {e}")
            raise
    
    def create_collection(
        self,
        collection_name: str,
        dimension: Optional[int] = None,
        **kwargs,
    ) -> None:
        """
        Create a new collection in Milvus.
        
        Args:
            collection_name: Name of the collection
            dimension: Optional dimension of the vectors
            **kwargs: Additional collection parameters
        """
        if dimension is None:
            dimension = self.dimension
        
        # Define collection schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dimension),
            FieldSchema(name="metadata", dtype=DataType.JSON),
        ]
        
        schema = CollectionSchema(
            fields=fields,
            description=f"Collection for {collection_name}",
        )
        
        try:
            # Create collection
            collection = Collection(
                name=collection_name,
                schema=schema,
            )
            
            # Create index
            index_params = {
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024},
            }
            collection.create_index(
                field_name="vector",
                index_params=index_params,
            )
            
            logger.info(f"Created collection {collection_name} with dimension {dimension}")
        except Exception as e:
            logger.error(f"Failed to create collection {collection_name}: {e}")
            raise
    
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
        try:
            collection = Collection(name=collection_name)
            
            # Prepare data
            num_vectors = len(vectors)
            if metadata is None:
                metadata = [{} for _ in range(num_vectors)]
            
            # Insert data
            entities = [
                {
                    "vector": vector.tolist(),
                    "metadata": meta,
                }
                for vector, meta in zip(vectors, metadata)
            ]
            
            collection.insert(entities)
            collection.flush()
            
            # Get inserted IDs
            results = collection.query(
                expr="id >= 0",
                output_fields=["id"],
            )
            vector_ids = [str(r["id"]) for r in results]
            
            logger.info(f"Inserted {num_vectors} vectors into collection {collection_name}")
            return vector_ids
        except Exception as e:
            logger.error(f"Failed to insert vectors into collection {collection_name}: {e}")
            raise
    
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
        try:
            collection = Collection(name=collection_name)
            collection.load()
            
            # Search
            search_params = {
                "metric_type": "L2",
                "params": {"nprobe": 10},
            }
            
            results = collection.search(
                data=[query_vector.tolist()],
                anns_field="vector",
                param=search_params,
                limit=limit,
                output_fields=["metadata"],
            )
            
            # Format results
            formatted_results = []
            for hits in results:
                for hit in hits:
                    formatted_results.append({
                        "id": str(hit.id),
                        "distance": float(hit.distance),
                        "metadata": hit.entity.get("metadata", {}),
                    })
            
            return formatted_results
        except Exception as e:
            logger.error(f"Failed to search in collection {collection_name}: {e}")
            raise
    
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
        try:
            collection = Collection(name=collection_name)
            
            # Convert string IDs to integers
            ids = [int(vid) for vid in vector_ids]
            
            # Delete vectors
            collection.delete(f"id in {ids}")
            collection.flush()
            
            logger.info(f"Deleted {len(vector_ids)} vectors from collection {collection_name}")
        except Exception as e:
            logger.error(f"Failed to delete vectors from collection {collection_name}: {e}")
            raise
    
    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """
        Get statistics about a collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Dictionary containing collection statistics
        """
        try:
            collection = Collection(name=collection_name)
            
            # Get collection stats
            stats = {
                "name": collection_name,
                "schema": collection.schema.to_dict(),
                "num_entities": collection.num_entities,
                "is_empty": collection.is_empty,
            }
            
            return stats
        except Exception as e:
            logger.error(f"Failed to get stats for collection {collection_name}: {e}")
            raise 