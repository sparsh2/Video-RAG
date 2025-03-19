"""
Image processing implementation.
"""
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
from loguru import logger
from PIL import Image

from .base import ImageProcessor


class PILImageProcessor(ImageProcessor):
    """Image processor implementation using PIL and OpenCV."""
    
    def process_image(self, image_path: str) -> Dict[str, Any]:
        """
        Process the input image.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Dictionary containing processing results
        """
        if not self.validate_input(image_path):
            logger.error(f"Invalid input image: {image_path}")
            return {}
        
        try:
            # Load image with PIL for metadata
            pil_image = Image.open(image_path)
            
            # Load image with OpenCV for processing
            cv_image = cv2.imread(image_path)
            if cv_image is None:
                raise ValueError("Failed to load image with OpenCV")
            
            # Convert BGR to RGB
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            
            # Create output directory
            output_dir = self.output_dir / "processed_images"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save processed image
            output_path = output_dir / f"{Path(image_path).stem}_processed.{self.image_config.get('output_format', 'jpg')}"
            cv2.imwrite(str(output_path), cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR))
            
            # Extract metadata
            metadata = {
                "format": pil_image.format,
                "mode": pil_image.mode,
                "size": pil_image.size,
                "width": pil_image.width,
                "height": pil_image.height,
            }
            
            # Calculate basic image statistics
            if cv_image.ndim == 3:
                mean_rgb = np.mean(cv_image, axis=(0, 1))
                std_rgb = np.std(cv_image, axis=(0, 1))
                metadata.update({
                    "mean_rgb": mean_rgb.tolist(),
                    "std_rgb": std_rgb.tolist(),
                })
            
            results = {
                "input_path": image_path,
                "output_path": str(output_path),
                "metadata": metadata,
            }
            
            logger.info(f"Successfully processed image: {image_path}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to process image {image_path}: {e}")
            return {} 