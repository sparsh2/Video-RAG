"""
Tests for data processing modules.
"""
import os
from pathlib import Path
from typing import Dict, Any

import pytest
import numpy as np
from PIL import Image

from src.data_processing.image import PILImageProcessor
from src.data_processing.video import OpenCVVideoProcessor


@pytest.fixture
def test_config() -> Dict[str, Any]:
    """Create a test configuration."""
    return {
        "data_processing": {
            "video": {
                "keyframe_interval": 1,
                "max_frames": 10,
                "output_format": "jpg",
            },
            "audio": {
                "sample_rate": 16000,
                "language": "en",
            },
            "output_dir": "test_data/processed",
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
def test_video(tmp_path: Path) -> str:
    """Create a test video."""
    video_path = tmp_path / "test.mp4"
    # Create a simple video using OpenCV
    import cv2
    
    # Create a video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        str(video_path),
        fourcc,
        30.0,
        (640, 480),
    )
    
    # Write some frames
    for _ in range(30):  # 1 second of video at 30fps
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :, 0] = 255  # Red frame
        out.write(frame)
    
    out.release()
    return str(video_path)


def test_image_processor(test_config: Dict[str, Any], test_image: str):
    """Test image processing."""
    processor = PILImageProcessor(test_config)
    results = processor.process_image(test_image)
    
    assert results
    assert "input_path" in results
    assert "output_path" in results
    assert "metadata" in results
    
    # Check if output file exists
    assert Path(results["output_path"]).exists()
    
    # Check metadata
    metadata = results["metadata"]
    assert metadata["format"] == "JPEG"
    assert metadata["mode"] == "RGB"
    assert metadata["size"] == (100, 100)
    assert "mean_rgb" in metadata
    assert "std_rgb" in metadata


def test_video_processor(test_config: Dict[str, Any], test_video: str):
    """Test video processing."""
    processor = OpenCVVideoProcessor(test_config)
    results = processor.process(test_video)
    
    assert results
    assert "input_path" in results
    assert "keyframes" in results
    
    # Check if keyframes were extracted
    keyframes = results["keyframes"]
    assert len(keyframes) > 0
    assert all(Path(path).exists() for path in keyframes)
    
    # Check if audio was extracted and transcribed
    if "audio_path" in results:
        assert Path(results["audio_path"]).exists()
        assert "transcript" in results


def test_invalid_inputs(test_config: Dict[str, Any], tmp_path: Path):
    """Test handling of invalid inputs."""
    # Test invalid image
    image_processor = PILImageProcessor(test_config)
    results = image_processor.process_image(str(tmp_path / "nonexistent.jpg"))
    assert not results
    
    # Test invalid video
    video_processor = OpenCVVideoProcessor(test_config)
    results = video_processor.process(str(tmp_path / "nonexistent.mp4"))
    assert not results 