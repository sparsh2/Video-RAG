"""
Video processing implementation using OpenCV and Whisper.
"""
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import whisper
from loguru import logger
from moviepy.editor import VideoFileClip

from .base import VideoProcessor


class OpenCVVideoProcessor(VideoProcessor):
    """Video processor implementation using OpenCV and Whisper."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the OpenCV video processor.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.whisper_model = whisper.load_model("base")
    
    def extract_keyframes(self, video_path: str) -> List[str]:
        """
        Extract keyframes from the video using OpenCV.
        
        Args:
            video_path: Path to the input video
            
        Returns:
            List of paths to extracted keyframes
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        interval = int(fps * self.video_config["keyframe_interval"])
        max_frames = self.video_config["max_frames"]
        
        frame_count = 0
        keyframe_paths = []
        output_dir = self.output_dir / "keyframes"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        while cap.isOpened() and len(keyframe_paths) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % interval == 0:
                output_path = output_dir / f"keyframe_{frame_count}.{self.video_config['output_format']}"
                cv2.imwrite(str(output_path), frame)
                keyframe_paths.append(str(output_path))
            
            frame_count += 1
        
        cap.release()
        logger.info(f"Extracted {len(keyframe_paths)} keyframes from {video_path}")
        return keyframe_paths
    
    def extract_audio(self, video_path: str) -> Optional[str]:
        """
        Extract audio from the video using moviepy.
        
        Args:
            video_path: Path to the input video
            
        Returns:
            Path to the extracted audio file, or None if extraction failed
        """
        try:
            output_dir = self.output_dir / "audio"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{Path(video_path).stem}.wav"
            
            video = VideoFileClip(video_path)
            video.audio.write_audiofile(
                str(output_path),
                fps=self.audio_config["sample_rate"],
                nbytes=2,
            )
            video.close()
            
            logger.info(f"Extracted audio to {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"Failed to extract audio: {e}")
            return None
    
    def transcribe_audio(self, audio_path: str) -> Optional[str]:
        """
        Transcribe audio using Whisper.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Transcribed text, or None if transcription failed
        """
        try:
            result = self.whisper_model.transcribe(
                audio_path,
                language=self.audio_config["language"],
            )
            return result["text"]
        except Exception as e:
            logger.error(f"Failed to transcribe audio: {e}")
            return None
    
    def process(self, input_path: str) -> Dict[str, Any]:
        """
        Process the input video.
        
        Args:
            input_path: Path to the input video
            
        Returns:
            Dictionary containing processing results
        """
        if not self.validate_input(input_path):
            logger.error(f"Invalid input video: {input_path}")
            return {}
        
        results = {
            "input_path": input_path,
            "keyframes": self.extract_keyframes(input_path),
        }
        
        audio_path = self.extract_audio(input_path)
        if audio_path:
            results["audio_path"] = audio_path
            results["transcript"] = self.transcribe_audio(audio_path)
        
        return results 