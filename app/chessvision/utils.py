from pathlib import Path
import torch
import timm
import logging
from numpy.typing import NDArray
from . import constants
import numpy as np
logger = logging.getLogger(__name__)
from torch.nn import Module
import cv2
import os

def get_device() -> torch.device:
    """Get the best available device for PyTorch."""
    if torch.cuda.is_available():
        logger.info("Using CUDA device")
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("Using MPS device")
        return torch.device("mps")
    logger.info("Using CPU device")
    return torch.device("cpu")


def get_classifier_model(model_id: str = "resnet18") -> torch.nn.Module:
    """Initialize the piece classifier model."""
    logger.info(f"Creating classifier model: {model_id}")
    return timm.create_model(
        model_id,
        num_classes=constants.NUM_CLASSES,
        in_chans=1,
    )

def load_model_checkpoint(
    model: Module | None,
    checkpoint_path: str,
    device: torch.device | None = None,
) -> Module:
    """Load a model checkpoint. If it's a full model, return it directly."""
    assert Path(checkpoint_path).exists(), f"Checkpoint not found: {checkpoint_path}"
    
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Try full model load first
    try:
        with torch.serialization.safe_globals({
            "ultralytics.nn.tasks.SegmentationModel": __import__("ultralytics").nn.tasks.SegmentationModel
        }):
            obj = torch.load(checkpoint_path, map_location=device)

        if isinstance(obj, Module):
            logger.info("✅ Loaded full YOLO model successfully.")
            return obj
    except Exception as e:
        logger.warning(f"⚠️ Full model load failed, fallback to state_dict: {e}")

    # Fallback to state_dict
    obj = torch.load(checkpoint_path, map_location=device)
    if isinstance(obj, dict):
        if model is None:
            raise ValueError("Model instance must be provided to load state_dict.")
        model.load_state_dict(obj["model"] if "model" in obj else obj)
        return model

    raise TypeError(f"Unsupported checkpoint format: {type(obj)}")


def ratio(a: float, b: float) -> float:
    """Calculate ratio between two numbers."""
    if a == 0 or b == 0:
        return -1
    return min(a, b) / float(max(a, b))


def listdir_nohidden(path: str) -> list[str]:
    """List directory contents, excluding hidden files."""
    return [f for f in os.listdir(path) if not f.startswith(".")]


def create_binary_mask(mask: NDArray[np.float32], threshold: float = 0.5) -> NDArray[np.uint8]:
    """Convert probability mask to binary mask."""
    assert isinstance(mask, np.ndarray), "Mask must be a numpy array"
    assert mask.dtype == np.float32, "Mask must be float32"
    assert 0 <= threshold <= 1, "Threshold must be between 0 and 1"

    mask = mask.copy()
    mask[mask > threshold] = 255
    mask[mask <= threshold] = 0
    return mask.astype(np.uint8)


def extract_perspective(
    image: NDArray[np.uint8],
    approx: NDArray[np.float32],
    out_size: tuple[int, int],
) -> NDArray[np.uint8]:
    """Extract a perspective-corrected region from an image."""
    assert isinstance(image, np.ndarray), "Image must be a numpy array"
    assert image.dtype == np.uint8, "Image must be uint8"
    assert isinstance(approx, np.ndarray), "Approx must be a numpy array"
    assert approx.dtype == np.float32, "Approx must be float32"
    assert len(approx) == 4, "Approx must contain exactly 4 points"

    w, h = out_size[0], out_size[1]
    dest = np.array(((0, 0), (w, 0), (w, h), (0, h)), np.float32)
    approx = np.array(approx, np.float32)

    coeffs = cv2.getPerspectiveTransform(approx, dest)
    return cv2.warpPerspective(image, coeffs, out_size)


def load_yolo_classification_model(model_weights: str) -> torch.nn.Module:
    """Load a YOLO model for piece classification.

    Args:
        model_weights: Path to YOLO model weights

    Returns:
        Wrapped YOLO model that implements the classifier interface

    Raises:
        ImportError: If ultralytics is not installed
    """
    try:
        from ultralytics.utils.tlc import TLCYOLO as YOLO
    except ImportError:
        try:
            from ultralytics import YOLO

            logger.warning("Using ultralytics (no 3lc integration) package")
        except ImportError:
            logger.warning(
                "YOLO model requires ultralytics package. Please install with 'pip install git+https://github.com/3lc-ai/ultralytics.git'.",
            )
            raise

    class YOLOModelWrapper:
        """Wrapper to make YOLO model behave like a classifier."""

        def __init__(self, model: YOLO):
            self.model = model

        def __call__(self, img: torch.Tensor) -> torch.Tensor:
            """Forward pass that returns probabilities for each class."""
            res = self.model(img.repeat((1, 3, 1, 1)), verbose=False)
            return torch.vstack([r.probs.data for r in res])

        def eval(self) -> None:
            """Set the model to evaluation mode."""
            self.model.eval()

        def train(self) -> None:
            """Set the model to training mode."""
            self.model.train()

        def to(self, device: torch.device) -> None:
            """Move the model to a specific device."""
            self.model.to(device)

    return YOLOModelWrapper(YOLO(model_weights))  # type: ignore[return-value]
