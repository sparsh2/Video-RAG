"""
Configuration management utilities.
"""
import os
from pathlib import Path
from typing import Any, Dict

import yaml
from pydantic import BaseModel


class Config(BaseModel):
    """Main configuration class."""
    data_processing: Dict[str, Any]
    embedding: Dict[str, Any]
    milvus: Dict[str, Any]
    llm: Dict[str, Any]
    web: Dict[str, Any]
    logging: Dict[str, Any]


def load_config(config_path: str = "config/default.yaml") -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Config object containing the configuration
    """
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    
    return Config(**config_dict)


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path to the project root directory
    """
    return Path(__file__).parent.parent.parent


def ensure_directories(config: Config) -> None:
    """
    Ensure all required directories exist.
    
    Args:
        config: Configuration object
    """
    directories = [
        config.data_processing["output_dir"],
        config.embedding["output_dir"],
        os.path.dirname(config.logging["file"]),
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True) 