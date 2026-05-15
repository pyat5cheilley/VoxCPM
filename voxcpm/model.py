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
                # low_cpu_mem_usage helps a lot on my machine (16 GB RAM)
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            ).to(self.device)
            self._model.eval()
            # Disable the cache during eval to save some VRAM on my 8 GB GPU
            if hasattr(self._model.config, "use_cache"):
                self._model.config.use_cache = False
            logger.info("Model loaded successfully on %s", self.device)
        except 
