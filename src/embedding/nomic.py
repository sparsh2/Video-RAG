"""
Nomic embedding implementation.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from loguru import logger
from nomic import embed
from PIL import Image
import cv2

from .base import ImageEmbeddingGenerator, TextEmbeddingGenerator


class NomicTextEmbeddingGenerator(TextEmbeddingGenerator):
    """Text embedding generator using Nomic."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Nomic text embedding generator.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.model = config.get("model", "nomic-embed-text-v1")
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text before embedding generation.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        # Basic text preprocessing
        return text.strip()
    
    def generate_embeddings(
        self,
        inputs: Union[str, List[str]],
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate embeddings for text using Nomic.
        
        Args:
            inputs: Input text or list of texts
            batch_size: Optional batch size for processing
            
        Returns:
            Array of embeddings
        """
        if isinstance(inputs, str):
            inputs = [inputs]
        
        # Use configured batch size if not specified
        if batch_size is None:
            batch_size = self.config.get("batch_size", 32)
        
        # Generate embeddings
        embeddings = []
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i + batch_size]
            batch_embeddings = embed(
                batch,
                model=self.model,
                max_length=self.max_length,
            )
            embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings)
    
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
        if output_path is None:
            output_path = self.output_dir / "text_embeddings.npz"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save embeddings
        np.savez(
            output_path,
            embeddings=embeddings,
            metadata=json.dumps(metadata),
        )
        
        logger.info(f"Saved text embeddings to {output_path}")
        return str(output_path)
    
    def load_embeddings(self, path: str) -> tuple[np.ndarray, Dict[str, Any]]:
        """
        Load embeddings and metadata from disk.
        
        Args:
            path: Path to the saved embeddings
            
        Returns:
            Tuple of (embeddings array, metadata dictionary)
        """
        data = np.load(path)
        embeddings = data["embeddings"]
        metadata = json.loads(data["metadata"])
        
        return embeddings, metadata


class NomicImageEmbeddingGenerator(ImageEmbeddingGenerator):
    """Image embedding generator using Nomic."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Nomic image embedding generator.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.model = config.get("model", "nomic-embed-image-v1")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image before embedding generation.
        
        Args:
            image: Input image array
            
        Returns:
            Preprocessed image array
        """
        # Resize image
        image = cv2.resize(image, self.image_size)
        
        # Convert to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        return image
    
    def generate_embeddings(
        self,
        inputs: Union[str, List[str], np.ndarray, List[np.ndarray]],
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate embeddings for images using Nomic.
        
        Args:
            inputs: Input image paths or arrays
            batch_size: Optional batch size for processing
            
        Returns:
            Array of embeddings
        """
        # Convert single input to list
        if isinstance(inputs, (str, np.ndarray)):
            inputs = [inputs]
        
        # Load images if paths are provided
        images = []
        for input_data in inputs:
            if isinstance(input_data, str):
                image = cv2.imread(input_data)
                if image is None:
                    logger.error(f"Failed to load image: {input_data}")
                    continue
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image = input_data
            images.append(image)
        
        # Preprocess images
        processed_images = [self.preprocess_image(image) for image in images]
        
        # Use configured batch size if not specified
        if batch_size is None:
            batch_size = self.config.get("batch_size", 32)
        
        # Generate embeddings
        embeddings = []
        for i in range(0, len(processed_images), batch_size):
            batch = processed_images[i:i + batch_size]
            batch_embeddings = embed(
                batch,
                model=self.model,
                modality="image",
            )
            embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings)
    
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
        if output_path is None:
            output_path = self.output_dir / "image_embeddings.npz"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save embeddings
        np.savez(
            output_path,
            embeddings=embeddings,
            metadata=json.dumps(metadata),
        )
        
        logger.info(f"Saved image embeddings to {output_path}")
        return str(output_path)
    
    def load_embeddings(self, path: str) -> tuple[np.ndarray, Dict[str, Any]]:
        """
        Load embeddings and metadata from disk.
        
        Args:
            path: Path to the saved embeddings
            
        Returns:
            Tuple of (embeddings array, metadata dictionary)
        """
        data = np.load(path)
        embeddings = data["embeddings"]
        metadata = json.loads(data["metadata"])
        
        return embeddings, metadata 