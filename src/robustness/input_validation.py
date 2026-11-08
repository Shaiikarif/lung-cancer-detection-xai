"""Medical image input validation and integrity verification."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Set, Tuple, Union
import cv2
import numpy as np

from src.utils.logger import get_logger

logger = get_logger("input_validation")


@dataclass
class ValidationResult:
    """Result container for input validation checks."""
    is_valid: bool
    error_message: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    channels: Optional[int] = None
    mean_intensity: Optional[float] = None


class InputValidator:
    """Validates CT scan file integrity, format, dimensions, and intensity statistics."""

    ALLOWED_EXTENSIONS: Set[str] = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    MIN_DIMENSION: int = 64
    MAX_DIMENSION: int = 4096

    @classmethod
    def validate_file_path(cls, file_path: Union[str, Path]) -> ValidationResult:
        """Verify that a path exists and has an accepted medical image extension.

        Args:
            file_path: Path to the file.

        Returns:
            ValidationResult instance.
        """
        path = Path(file_path)

        if not path.exists():
            return ValidationResult(is_valid=False, error_message=f"File not found: {path}")

        if not path.is_file():
            return ValidationResult(is_valid=False, error_message=f"Path is not a regular file: {path}")

        ext = path.suffix.lower()
        if ext not in cls.ALLOWED_EXTENSIONS:
            return ValidationResult(
                is_valid=False,
                error_message=f"Unsupported format '{ext}'. Allowed: {', '.join(sorted(cls.ALLOWED_EXTENSIONS))}"
            )

        return cls.validate_image_file(path)

    @classmethod
    def validate_image_file(cls, path: Path) -> ValidationResult:
        """Read and validate the image data."""
        try:
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        except Exception as e:
            return ValidationResult(is_valid=False, error_message=f"Failed to read image: {str(e)}")

        if img is None:
            return ValidationResult(is_valid=False, error_message="Image file could not be decoded (corrupt file).")

        return cls.validate_numpy_image(img)

    @classmethod
    def validate_numpy_image(cls, img: np.ndarray) -> ValidationResult:
        """Validate an in-memory NumPy image array."""
        if not isinstance(img, np.ndarray):
            return ValidationResult(is_valid=False, error_message="Image must be a valid numpy array.")

        if img.ndim < 2 or img.ndim > 3:
            return ValidationResult(is_valid=False, error_message=f"Invalid image array dimensions: {img.ndim}D")

        h, w = img.shape[:2]
        channels = 1 if img.ndim == 2 else img.shape[2]

        if h < cls.MIN_DIMENSION or w < cls.MIN_DIMENSION:
            return ValidationResult(
                is_valid=False,
                error_message=f"Image resolution {w}x{h} is below minimum allowed ({cls.MIN_DIMENSION}x{cls.MIN_DIMENSION})."
            )

        if h > cls.MAX_DIMENSION or w > cls.MAX_DIMENSION:
            return ValidationResult(
                is_valid=False,
                error_message=f"Image resolution {w}x{h} exceeds maximum allowed ({cls.MAX_DIMENSION}x{cls.MAX_DIMENSION})."
            )

        # Check for degenerate (blank or saturated) images
        mean_intensity = float(np.mean(img))
        std_intensity = float(np.std(img))

        if std_intensity < 1e-4:
            return ValidationResult(
                is_valid=False,
                error_message="Image is degenerate (uniform intensity / blank image)."
            )

        return ValidationResult(
            is_valid=True,
            width=w,
            height=h,
            channels=channels,
            mean_intensity=mean_intensity
        )
