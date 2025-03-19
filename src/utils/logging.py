"""
Logging configuration utilities.
"""
import sys
from pathlib import Path
from typing import Dict, Any

from loguru import logger


def setup_logging(config: Dict[str, Any]) -> None:
    """
    Set up logging configuration.
    
    Args:
        config: Logging configuration dictionary
    """
    # Remove default logger
    logger.remove()
    
    # Add console logger
    logger.add(
        sys.stderr,
        format=config["format"],
        level=config["level"],
        colorize=True,
    )
    
    # Add file logger
    log_file = Path(config["file"])
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    logger.add(
        log_file,
        format=config["format"],
        level=config["level"],
        rotation="500 MB",
        retention="10 days",
    ) 