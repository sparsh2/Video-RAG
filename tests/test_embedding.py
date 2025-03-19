"""
Tests for embedding generators.
"""
import os
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pytest
from PIL import Image

from src.embedding.nomic import NomicImageEmbeddingGenerator, NomicTextEmbeddingGenerator


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
    }


@pytest.fixture
def test_image(tmp_path: Path) -> str:
    """Create a test image."""
    image_path = tmp_path / "test.jpg"
    image = Image.new("RGB", (100, 100), color="red")
    image.save(image_path)
    return str(image_path)


@pytest.fixture
def test_text(tmp_path: Path) -> str:
    """Create a test text file."""
    text_path = tmp_path / "test.txt"
    with open(text_path, "w", encoding="utf-8") as f:
        f.write("This is a test text.")
    return str(text_path)


def test_text_embedding_generator(test_config: Dict[str, Any], test_text: str):
    """Test text embedding generation."""
    generator = NomicTextEmbeddingGenerator(test_config)
    
    # Test single text embedding
    embeddings = generator.generate_embeddings("Test text")
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == 1
    
    # Test multiple text embeddings
    texts = ["First text", "Second text", "Third text"]
    embeddings = generator.generate_embeddings(texts)
    assert embeddings.shape[0] == len(texts)
    
    # Test saving and loading
    metadata = {"test": "metadata"}
    output_path = generator.save_embeddings(embeddings, metadata)
    loaded_embeddings, loaded_metadata = generator.load_embeddings(output_path)
    
    assert np.array_equal(embeddings, loaded_embeddings)
    assert metadata == loaded_metadata


def test_image_embedding_generator(test_config: Dict[str, Any], test_image: str):
    """Test image embedding generation."""
    generator = NomicImageEmbeddingGenerator(test_config)
    
    # Test single image embedding
    embeddings = generator.generate_embeddings(test_image)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == 1
    
    # Test multiple image embeddings
    images = [test_image, test_image]
    embeddings = generator.generate_embeddings(images)
    assert embeddings.shape[0] == len(images)
    
    # Test numpy array input
    image_array = np.zeros((100, 100, 3), dtype=np.uint8)
    image_array[:, :, 0] = 255  # Red image
    embeddings = generator.generate_embeddings(image_array)
    assert embeddings.shape[0] == 1
    
    # Test saving and loading
    metadata = {"test": "metadata"}
    output_path = generator.save_embeddings(embeddings, metadata)
    loaded_embeddings, loaded_metadata = generator.load_embeddings(output_path)
    
    assert np.array_equal(embeddings, loaded_embeddings)
    assert metadata == loaded_metadata


def test_preprocessing(test_config: Dict[str, Any]):
    """Test preprocessing functions."""
    # Test text preprocessing
    text_generator = NomicTextEmbeddingGenerator(test_config)
    preprocessed_text = text_generator.preprocess_text("  Test text  ")
    assert preprocessed_text == "Test text"
    
    # Test image preprocessing
    image_generator = NomicImageEmbeddingGenerator(test_config)
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
    test_image[:, :, 0] = 255  # Red image
    
    preprocessed_image = image_generator.preprocess_image(test_image)
    assert preprocessed_image.shape == (*image_generator.image_size, 3)
    assert preprocessed_image.dtype == np.float32
    assert np.all(preprocessed_image >= 0) and np.all(preprocessed_image <= 1)


def test_invalid_inputs(test_config: Dict[str, Any], tmp_path: Path):
    """Test handling of invalid inputs."""
    # Test invalid text
    text_generator = NomicTextEmbeddingGenerator(test_config)
    with pytest.raises(Exception):
        text_generator.generate_embeddings("")  # Empty text
    
    # Test invalid image
    image_generator = NomicImageEmbeddingGenerator(test_config)
    with pytest.raises(Exception):
        image_generator.generate_embeddings(str(tmp_path / "nonexistent.jpg"))
    
    # Test invalid image array
    with pytest.raises(Exception):
        image_generator.generate_embeddings(np.zeros((100, 100)))  # 2D array 