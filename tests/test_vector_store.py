"""
Tests for vector store implementation.
"""
import os
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pytest
from pymilvus import connections, utility

from src.vector_store.milvus import MilvusStore


@pytest.fixture
def test_config() -> Dict[str, Any]:
    """Create a test configuration."""
    return {
        "vector_store": {
            "host": "localhost",
            "port": 19530,
            "alias": "test",
            "dimension": 768,
        },
    }


@pytest.fixture
def test_vectors() -> np.ndarray:
    """Create test vectors."""
    return np.random.rand(10, 768).astype(np.float32)


@pytest.fixture
def test_metadata() -> list[Dict[str, Any]]:
    """Create test metadata."""
    return [
        {"id": i, "text": f"Test text {i}", "source": "test"}
        for i in range(10)
    ]


@pytest.fixture
def milvus_store(test_config: Dict[str, Any]) -> MilvusStore:
    """Create a Milvus store instance."""
    store = MilvusStore(test_config)
    store.connect()
    yield store
    store.disconnect()


@pytest.fixture
def test_collection(milvus_store: MilvusStore) -> str:
    """Create a test collection."""
    collection_name = "test_collection"
    
    # Delete collection if it exists
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
    
    # Create collection
    milvus_store.create_collection(collection_name)
    return collection_name


def test_create_collection(milvus_store: MilvusStore):
    """Test collection creation."""
    collection_name = "test_create"
    
    # Delete collection if it exists
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
    
    # Create collection
    milvus_store.create_collection(collection_name)
    
    # Verify collection exists
    assert utility.has_collection(collection_name)
    
    # Get collection stats
    stats = milvus_store.get_collection_stats(collection_name)
    assert stats["name"] == collection_name
    assert stats["num_entities"] == 0
    assert stats["is_empty"] is True


def test_insert_vectors(
    milvus_store: MilvusStore,
    test_collection: str,
    test_vectors: np.ndarray,
    test_metadata: list[Dict[str, Any]],
):
    """Test vector insertion."""
    # Insert vectors
    vector_ids = milvus_store.insert(test_collection, test_vectors, test_metadata)
    
    # Verify number of inserted vectors
    assert len(vector_ids) == len(test_vectors)
    
    # Get collection stats
    stats = milvus_store.get_collection_stats(test_collection)
    assert stats["num_entities"] == len(test_vectors)
    assert stats["is_empty"] is False


def test_search_vectors(
    milvus_store: MilvusStore,
    test_collection: str,
    test_vectors: np.ndarray,
    test_metadata: list[Dict[str, Any]],
):
    """Test vector search."""
    # Insert vectors
    milvus_store.insert(test_collection, test_vectors, test_metadata)
    
    # Create query vector
    query_vector = test_vectors[0]
    
    # Search vectors
    results = milvus_store.search(test_collection, query_vector, limit=5)
    
    # Verify results
    assert len(results) == 5
    assert all(isinstance(r["id"], str) for r in results)
    assert all(isinstance(r["distance"], float) for r in results)
    assert all(isinstance(r["metadata"], dict) for r in results)
    
    # First result should be the query vector itself
    assert results[0]["distance"] < 1e-6


def test_delete_vectors(
    milvus_store: MilvusStore,
    test_collection: str,
    test_vectors: np.ndarray,
    test_metadata: list[Dict[str, Any]],
):
    """Test vector deletion."""
    # Insert vectors
    vector_ids = milvus_store.insert(test_collection, test_vectors, test_metadata)
    
    # Delete some vectors
    delete_ids = vector_ids[:5]
    milvus_store.delete(test_collection, delete_ids)
    
    # Get collection stats
    stats = milvus_store.get_collection_stats(test_collection)
    assert stats["num_entities"] == len(test_vectors) - len(delete_ids)
    
    # Verify deleted vectors are gone
    query_vector = test_vectors[0]
    results = milvus_store.search(test_collection, query_vector, limit=10)
    result_ids = [r["id"] for r in results]
    assert not any(rid in delete_ids for rid in result_ids)


def test_invalid_operations(milvus_store: MilvusStore):
    """Test invalid operations."""
    # Test non-existent collection
    with pytest.raises(Exception):
        milvus_store.search("non_existent", np.random.rand(768))
    
    # Test invalid vector dimension
    collection_name = "test_invalid"
    milvus_store.create_collection(collection_name, dimension=768)
    
    with pytest.raises(Exception):
        milvus_store.insert(collection_name, np.random.rand(10, 512))
    
    # Test invalid metadata
    with pytest.raises(Exception):
        milvus_store.insert(
            collection_name,
            np.random.rand(10, 768),
            metadata=[{"id": i} for i in range(5)],  # Wrong length
        ) 