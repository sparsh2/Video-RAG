"""
Main script for processing data.
"""
import argparse
from pathlib import Path
from typing import List, Optional

from loguru import logger

from src.utils.config import load_config, setup_logging
from .image import PILImageProcessor
from .video import OpenCVVideoProcessor


def process_directory(
    input_dir: str,
    config_path: str = "config/default.yaml",
    recursive: bool = True,
) -> None:
    """
    Process all media files in a directory.
    
    Args:
        input_dir: Path to the input directory
        config_path: Path to the configuration file
        recursive: Whether to process subdirectories
    """
    # Load configuration
    config = load_config(config_path)
    
    # Set up logging
    setup_logging(config.logging)
    
    # Initialize processors
    video_processor = OpenCVVideoProcessor(config.data_processing)
    image_processor = PILImageProcessor(config.data_processing)
    
    # Get all media files
    input_path = Path(input_dir)
    if recursive:
        video_files = list(input_path.rglob("*.mp4")) + list(input_path.rglob("*.avi"))
        image_files = list(input_path.rglob("*.jpg")) + list(input_path.rglob("*.png"))
    else:
        video_files = list(input_path.glob("*.mp4")) + list(input_path.glob("*.avi"))
        image_files = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png"))
    
    # Process videos
    for video_file in video_files:
        logger.info(f"Processing video: {video_file}")
        results = video_processor.process(str(video_file))
        if results:
            logger.info(f"Successfully processed video: {video_file}")
        else:
            logger.error(f"Failed to process video: {video_file}")
    
    # Process images
    for image_file in image_files:
        logger.info(f"Processing image: {image_file}")
        results = image_processor.process_image(str(image_file))
        if results:
            logger.info(f"Successfully processed image: {image_file}")
        else:
            logger.error(f"Failed to process image: {image_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Process media files for MM-RAG")
    parser.add_argument(
        "input_dir",
        help="Directory containing media files to process",
    )
    parser.add_argument(
        "--config",
        default="config/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Do not process subdirectories",
    )
    
    args = parser.parse_args()
    process_directory(
        args.input_dir,
        args.config,
        recursive=not args.no_recursive,
    )


if __name__ == "__main__":
    main() 