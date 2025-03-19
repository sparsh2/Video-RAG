"""
Tests for RAG pipeline.
"""
import os
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pytest
from PIL import Image

from src.rag.pipeline import NomicRAGPipeline


@pytest.fixture
def test_config() -> Dict[str, Any]:
    """Create a test configuration."""
    return {
        "embedding": {
            "model": "nomic-embed-text-v1",
            "batch_size": 32,
            "max_length": 512,
            "image_size": (224, 224),
            "output_dir": "test_data/embeddings",
        },
        "vector_store": {
            "host": "localhost",
            "port": 19530,
            "alias": "test",
            "dimension": 768,
        },
        "text_collection": "test_text_embeddings",
        "image_collection": "test_image_embeddings",
        "model": "mistralai/Mistral-7B-Instruct-v0.2",
    }


@pytest.fixture
def test_text() -> str:
    """Create test text."""
    return "This is a test text for the RAG pipeline."


@pytest.fixture
def test_image(tmp_path: Path) -> str:
    """Create a test image."""
    image_path = tmp_path / "test.jpg"
    image = Image.new("RGB", (100, 100), color="red")
    image.save(image_path)
    return str(image_path)


@pytest.fixture
def test_metadata() -> list[Dict[str, Any]]:
    """Create test metadata."""
    return [
        {
            "id": i,
            "text": f"Test text {i}",
            "source": "test",
            "description": f"Test description {i}",
        }
        for i in range(5)
    ]


@pytest.fixture
def rag_pipeline(test_config: Dict[str, Any]) -> NomicRAGPipeline:
    """Create a RAG pipeline instance."""
    pipeline = NomicRAGPipeline(test_config)
    yield pipeline
    del pipeline


def test_add_text_content(
    rag_pipeline: NomicRAGPipeline,
    test_text: str,
    test_metadata: list[Dict[str, Any]],
):
    """Test adding text content to knowledge base."""
    # Add single text
    content_ids = rag_pipeline.add_to_knowledge_base(
        test_text,
        metadata=[test_metadata[0]],
        modality="text",
    )
    assert len(content_ids) == 1
    
    # Add multiple texts
    texts = [f"Test text {i}" for i in range(3)]
    content_ids = rag_pipeline.add_to_knowledge_base(
        texts,
        metadata=test_metadata[:3],
        modality="text",
    )
    assert len(content_ids) == 3


def test_add_image_content(
    rag_pipeline: NomicRAGPipeline,
    test_image: str,
    test_metadata: list[Dict[str, Any]],
):
    """Test adding image content to knowledge base."""
    # Add single image
    content_ids = rag_pipeline.add_to_knowledge_base(
        test_image,
        metadata=[test_metadata[0]],
        modality="image",
    )
    assert len(content_ids) == 1
    
    # Add multiple images
    images = [test_image] * 3
    content_ids = rag_pipeline.add_to_knowledge_base(
        images,
        metadata=test_metadata[:3],
        modality="image",
    )
    assert len(content_ids) == 3


def test_process_text_query(
    rag_pipeline: NomicRAGPipeline,
    test_text: str,
    test_metadata: list[Dict[str, Any]],
):
    """Test processing text queries."""
    # Add test content
    rag_pipeline.add_to_knowledge_base(
        test_text,
        metadata=[test_metadata[0]],
        modality="text",
    )
    
    # Process query
    results = rag_pipeline.process_query(
        test_text,
        modality="text",
        top_k=1,
    )
    
    # Verify results
    assert results["query"] == test_text
    assert results["modality"] == "text"
    assert len(results["results"]) == 1
    assert results["results"][0]["metadata"] == test_metadata[0]


def test_process_image_query(
    rag_pipeline: NomicRAGPipeline,
    test_image: str,
    test_metadata: list[Dict[str, Any]],
):
    """Test processing image queries."""
    # Add test content
    rag_pipeline.add_to_knowledge_base(
        test_image,
        metadata=[test_metadata[0]],
        modality="image",
    )
    
    # Process query
    results = rag_pipeline.process_query(
        test_image,
        modality="image",
        top_k=1,
    )
    
    # Verify results
    assert results["query"] == test_image
    assert results["modality"] == "image"
    assert len(results["results"]) == 1
    assert results["results"][0]["metadata"] == test_metadata[0]


def test_generate_response(
    rag_pipeline: NomicRAGPipeline,
    test_text: str,
    test_metadata: list[Dict[str, Any]],
):
    """Test response generation."""
    # Add test content
    rag_pipeline.add_to_knowledge_base(
        test_text,
        metadata=[test_metadata[0]],
        modality="text",
    )
    
    # Process query and generate response
    retrieved_info = rag_pipeline.process_query(
        test_text,
        modality="text",
        top_k=1,
    )
    
    response = rag_pipeline.generate_response(test_text, retrieved_info)
    
    # Verify response
    assert isinstance(response, str)
    assert len(response) > 0


def test_remove_content(
    rag_pipeline: NomicRAGPipeline,
    test_text: str,
    test_metadata: list[Dict[str, Any]],
):
    """Test removing content from knowledge base."""
    # Add test content
    content_ids = rag_pipeline.add_to_knowledge_base(
        test_text,
        metadata=[test_metadata[0]],
        modality="text",
    )
    
    # Remove content
    rag_pipeline.remove_from_knowledge_base(content_ids)
    
    # Verify content is removed
    results = rag_pipeline.process_query(
        test_text,
        modality="text",
        top_k=1,
    )
    assert len(results["results"]) == 0


def test_invalid_operations(rag_pipeline: NomicRAGPipeline):
    """Test invalid operations."""
    # Test invalid modality
    with pytest.raises(Exception):
        rag_pipeline.add_to_knowledge_base(
            "test",
            modality="invalid",
        )
    
    # Test invalid content type
    with pytest.raises(Exception):
        rag_pipeline.add_to_knowledge_base(
            np.random.rand(10, 512),  # Wrong dimension
            modality="text",
        )
    
    # Test invalid metadata length
    with pytest.raises(Exception):
        rag_pipeline.add_to_knowledge_base(
            ["text1", "text2"],
            metadata=[{"id": 1}],  # Wrong length
            modality="text",
        ) 