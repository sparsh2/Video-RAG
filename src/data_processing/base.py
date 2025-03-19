"""
Base classes for data processing.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger


class DataProcessor(ABC):
    """Abstract base class for data processors."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data processor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.output_dir = Path(config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    def process(self, input_path: str) -> Dict[str, Any]:
        """
        Process the input data.
        
        Args:
            input_path: Path to the input data
            
        Returns:
            Dictionary containing processing results
        """
        pass
    
    @abstractmethod
    def validate_input(self, input_path: str) -> bool:
        """
        Validate the input data.
        
        Args:
            input_path: Path to the input data
            
        Returns:
            True if input is valid, False otherwise
        """
        pass


class VideoProcessor(DataProcessor):
    """Base class for video processing."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the video processor.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.video_config = config["video"]
        self.audio_config = config["audio"]
    
    def validate_input(self, input_path: str) -> bool:
        """
        Validate the input video file.
        
        Args:
            input_path: Path to the input video
            
        Returns:
            True if input is valid, False otherwise
        """
        path = Path(input_path)
        return path.exists() and path.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]
    
    @abstractmethod
    def extract_keyframes(self, video_path: str) -> List[str]:
        """
        Extract keyframes from the video.
        
        Args:
            video_path: Path to the input video
            
        Returns:
            List of paths to extracted keyframes
        """
        pass
    
    @abstractmethod
    def extract_audio(self, video_path: str) -> Optional[str]:
        """
        Extract audio from the video.
        
        Args:
            video_path: Path to the input video
            
        Returns:
            Path to the extracted audio file, or None if extraction failed
        """
        pass


class ImageProcessor(DataProcessor):
    """Base class for image processing."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the image processor.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.image_config = config.get("image", {})
    
    def validate_input(self, input_path: str) -> bool:
        """
        Validate the input image file.
        
        Args:
            input_path: Path to the input image
            
        Returns:
            True if input is valid, False otherwise
        """
        path = Path(input_path)
        return path.exists() and path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
    
    @abstractmethod
    def process_image(self, image_path: str) -> Dict[str, Any]:
        """
        Process the input image.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Dictionary containing processing results
        """
        pass 