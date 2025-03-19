"""
Tests for configuration utilities.
"""
import os
from pathlib import Path

import pytest
import yaml

from src.utils.config import Config, load_config, get_project_root, ensure_directories


@pytest.fixture
def test_config():
    """Create a test configuration."""
    return {
        "data_processing": {
            "output_dir": "test_data/processed",
        },
        "embedding": {
            "output_dir": "test_data/embeddings",
        },
        "milvus": {},
        "llm": {},
        "web": {},
        "logging": {
            "file": "test_data/logs/test.log",
        },
    }


def test_load_config(tmp_path, test_config):
    """Test loading configuration from file."""
    config_path = tmp_path / "test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(test_config, f)
    
    config = load_config(str(config_path))
    assert isinstance(config, Config)
    assert config.data_processing["output_dir"] == test_config["data_processing"]["output_dir"]


def test_get_project_root():
    """Test getting project root directory."""
    root = get_project_root()
    assert isinstance(root, Path)
    assert (root / "src").exists()


def test_ensure_directories(tmp_path, test_config):
    """Test directory creation."""
    config = Config(**test_config)
    ensure_directories(config)
    
    for directory in [
        test_config["data_processing"]["output_dir"],
        test_config["embedding"]["output_dir"],
        os.path.dirname(test_config["logging"]["file"]),
    ]:
        assert Path(directory).exists() 