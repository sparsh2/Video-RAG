"""
Base classes for embedding generation.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from loguru import logger


class EmbeddingGenerator(ABC):
    """Abstract base class for embedding generators."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the embedding generator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.output_dir = Path(config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    def generate_embeddings(
        self,
        inputs: Union[str, List[str], np.ndarray],
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate embeddings for the input data.
        
        Args:
            inputs: Input data (text, image paths, or image arrays)
            batch_size: Optional batch size for processing
            
        Returns:
            Array of embeddings
        """
        pass
    
    @abstractmethod
    def save_embeddings(
        self,
        embeddings: np.ndarray,
        metadata: Dict[str, Any],
        output_path: Optional[str] = None,
    ) -> str:
        """
        Save embeddings and metadata to disk.
        
        Args:
            embeddings: Array of embeddings
            metadata: Dictionary containing metadata
            output_path: Optional path to save the embeddings
            
        Returns:
            Path where embeddings were saved
        """
        pass
    
    @abstractmethod
    def load_embeddings(self, path: str) -> tuple[np.ndarray, Dict[str, Any]]:
        """
        Load embeddings and metadata from disk.
        
        Args:
            path: Path to the saved embeddings
            
        Returns:
            Tuple of (embeddings array, metadata dictionary)
        """
        pass


class TextEmbeddingGenerator(EmbeddingGenerator):
    """Base class for text embedding generation."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the text embedding generator.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.max_length = config.get("max_length", 512)
    
    @abstractmethod
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text before embedding generation.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        pass


class ImageEmbeddingGenerator(EmbeddingGenerator):
    """Base class for image embedding generation."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the image embedding generator.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.image_size = config.get("image_size", (224, 224))
    
    @abstractmethod
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image before embedding generation.
        
        Args:
            image: Input image array
            
        Returns:
            Preprocessed image array
        """
        pass 