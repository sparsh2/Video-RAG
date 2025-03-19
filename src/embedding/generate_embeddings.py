"""
Main script for generating embeddings.
"""
import argparse
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

from src.utils.config import load_config, setup_logging
from .nomic import NomicImageEmbeddingGenerator, NomicTextEmbeddingGenerator


def process_directory(
    input_dir: str,
    config_path: str = "config/default.yaml",
    recursive: bool = True,
) -> None:
    """
    Process all media files in a directory and generate embeddings.
    
    Args:
        input_dir: Path to the input directory
        config_path: Path to the configuration file
        recursive: Whether to process subdirectories
    """
    # Load configuration
    config = load_config(config_path)
    
    # Set up logging
    setup_logging(config.logging)
    
    # Initialize embedding generators
    text_generator = NomicTextEmbeddingGenerator(config.embedding)
    image_generator = NomicImageEmbeddingGenerator(config.embedding)
    
    # Get all media files
    input_path = Path(input_dir)
    if recursive:
        image_files = list(input_path.rglob("*.jpg")) + list(input_path.rglob("*.png"))
        text_files = list(input_path.rglob("*.txt"))
    else:
        image_files = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png"))
        text_files = list(input_path.glob("*.txt"))
    
    # Process images
    if image_files:
        logger.info(f"Processing {len(image_files)} images")
        image_embeddings = image_generator.generate_embeddings(
            [str(f) for f in image_files],
        )
        
        # Save image embeddings
        image_metadata = {
            "files": [str(f) for f in image_files],
            "model": image_generator.model,
            "image_size": image_generator.image_size,
        }
        image_generator.save_embeddings(image_embeddings, image_metadata)
    
    # Process text files
    if text_files:
        logger.info(f"Processing {len(text_files)} text files")
        texts = []
        for text_file in text_files:
            try:
                with open(text_file, "r", encoding="utf-8") as f:
                    texts.append(f.read())
            except Exception as e:
                logger.error(f"Failed to read text file {text_file}: {e}")
        
        if texts:
            text_embeddings = text_generator.generate_embeddings(texts)
            
            # Save text embeddings
            text_metadata = {
                "files": [str(f) for f in text_files],
                "model": text_generator.model,
                "max_length": text_generator.max_length,
            }
            text_generator.save_embeddings(text_embeddings, text_metadata)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate embeddings for MM-RAG")
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