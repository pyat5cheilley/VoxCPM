"""VoxCPM model loading and inference utilities.

This module provides the core model class for VoxCPM, handling model
initialization, audio preprocessing, and speech recognition inference.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Union

import torch
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_MODEL_DIR = os.path.join(os.path.expanduser("~"), ".cache", "voxcpm")


class VoxCPMModel:
    """Wrapper around the VoxCPM speech recognition model.

    Handles model loading from a local directory or a remote hub,
    audio preprocessing, and transcription inference.

    Args:
        model_dir: Path to the directory containing model weights and config.
        device: Torch device string, e.g. ``"cpu"`` or ``"cuda:0"``.
            Defaults to CUDA if available, otherwise CPU.
        dtype: Torch dtype for model weights. Defaults to ``torch.float16``
            on CUDA and ``torch.float32`` on CPU.
    """

    def __init__(
        self,
        model_dir: Union[str, Path],
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        self.model_dir = Path(model_dir)
        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"Model directory not found: {self.model_dir}. "
                "Please download the model weights first."
            )

        # Resolve device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Resolve dtype
        if dtype is None:
            dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.dtype = dtype

        self._model = None
        self._processor = None
        self._load_model()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load model weights and processor from *model_dir*."""
        logger.info("Loading VoxCPM model from %s", self.model_dir)
        try:
            # Lazy import to keep startup fast when the library is imported
            # without actually running inference.
            from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq  # type: ignore

            self._processor = AutoProcessor.from_pretrained(
                str(self.model_dir), trust_remote_code=True
            )
            self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
                str(self.model_dir),
                torch_dtype=self.dtype,
                trust_remote_code=True,
            ).to(self.device)
            self._model.eval()
            logger.info("Model loaded successfully on %s", self.device)
        except Exception as exc:
            logger.exception("Failed to load model: %s", exc)
            raise

    @staticmethod
    def _ensure_mono_float32(audio: np.ndarray) -> np.ndarray:
        """Convert *audio* to mono float32 in [-1, 1]."""
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)
        max_val = np.abs(audio).max()
        if max_val > 1.0:
            audio = audio / max_val
        return audio

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        language: Optional[str] = None,
        max_new_tokens: int = 256,
    ) -> str:
        """Transcribe *audio* and return the decoded text.

        Args:
            audio: 1-D or 2-D numpy array of audio samples.
            sample_rate: Sampling rate of *audio* in Hz.
            language: Optional BCP-47 language tag (e.g. ``"zh"`` or ``"en"``).
                When ``None`` the model performs automatic language detection.
            max_new_tokens: Maximum number of tokens to generate.

        Returns:
            Transcribed text string.
        """
        if self._model is None or self._processor is None:
            raise RuntimeError("Model is not loaded. Call _load_model() first.")

        audio = self._ensure_mono_float32(audio)

        inputs = self._processor(
            audio,
            sampling_rate=sample_rate,
            return_tensors="pt",
        )
        input_features = inputs.input_features.to(self.device, dtype=self.dtype)

        generate_kwargs: dict = {"max_new_tokens": max_new_tokens}
        if language is not None:
            generate_kwargs["language"] = language

        with torch.no_grad():
            predicted_ids = self._model.generate(input_features, **generate_kwargs)

        transcription: str = self._processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0]
        return transcription.strip()

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"VoxCPMModel(model_dir={self.model_dir!r}, "
            f"device={self.device}, dtype={self.dtype})"
        )
