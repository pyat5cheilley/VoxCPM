"""Utility functions for VoxCPM audio processing and file handling."""

import os
import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Supported audio formats for loading
SUPPORTED_AUDIO_FORMATS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}

# Target sample rate expected by VoxCPM model
TARGET_SAMPLE_RATE = 16000


def get_audio_duration(audio: np.ndarray, sample_rate: int) -> float:
    """Return duration of audio array in seconds."""
    return len(audio) / sample_rate


def validate_audio_file(filepath: Union[str, Path]) -> Path:
    """Validate that a file exists and has a supported audio extension.

    Args:
        filepath: Path to the audio file.

    Returns:
        Resolved Path object.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file extension is not supported.
    """
    path = Path(filepath).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    if path.suffix.lower() not in SUPPORTED_AUDIO_FORMATS:
        raise ValueError(
            f"Unsupported audio format '{path.suffix}'. "
            f"Supported formats: {', '.join(sorted(SUPPORTED_AUDIO_FORMATS))}"
        )
    return path


def load_audio(
    filepath: Union[str, Path],
    target_sr: int = TARGET_SAMPLE_RATE,
) -> Tuple[np.ndarray, int]:
    """Load an audio file, resample to target sample rate, and convert to mono float32.

    Requires ``librosa`` to be installed.

    Args:
        filepath: Path to the audio file.
        target_sr: Desired sample rate (default: 16000 Hz).

    Returns:
        Tuple of (audio_array, sample_rate) where audio_array is float32 mono.
    """
    try:
        import librosa  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "librosa is required for audio loading. Install it with: pip install librosa"
        ) from exc

    path = validate_audio_file(filepath)
    logger.debug("Loading audio from %s (target_sr=%d)", path, target_sr)

    audio, sr = librosa.load(str(path), sr=target_sr, mono=True, dtype=np.float32)
    duration = get_audio_duration(audio, sr)
    logger.debug("Loaded %.2f seconds of audio at %d Hz", duration, sr)
    return audio, sr


def chunk_audio(
    audio: np.ndarray,
    sample_rate: int,
    chunk_seconds: float = 30.0,
    overlap_seconds: float = 0.5,
) -> list:
    """Split a long audio array into smaller chunks for batch processing.

    Args:
        audio: 1-D float32 numpy array.
        sample_rate: Sample rate of the audio.
        chunk_seconds: Maximum length of each chunk in seconds.
        overlap_seconds: Overlap between consecutive chunks in seconds.
            Defaults to 0.5s to reduce artifacts at chunk boundaries.

    Returns:
        List of (start_sample, chunk_array) tuples.
    """
    chunk_size = int(chunk_seconds * sample_rate)
    step_size = chunk_size - int(overlap_seconds * sample_rate)
    if step_size <= 0:
