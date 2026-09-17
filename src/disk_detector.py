"""
Solar disk detection and geometry localization module.
Extracts solar disk center (cx, cy), radius (R), and masks out off-limb sky.
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import cv2
import numpy as np

from src.config import DiskDetectionConfig


@dataclass
class SolarDiskGeometry:
    center_x: float
    center_y: float
    radius: float
    mask: np.ndarray
    effective_mask: np.ndarray  # Mask with edge margin applied
    confidence: float
    is_valid: bool


class SolarDiskDetector:
    """
    Detects the solar disk in full-disk solar continuum images using
    morphology, thresholding, and contour/circle fitting.
    """

    def __init__(self, config: Optional[DiskDetectionConfig] = None):
        self.config = config or DiskDetectionConfig()

    def detect(self, image: np.ndarray) -> SolarDiskGeometry:
        """
        Locate the solar disk in an image.

        Args:
            image: 2D grayscale or 3D BGR image

        Returns:
            SolarDiskGeometry with center, radius, masks, and validity
        """
        # Convert to single-channel grayscale if needed
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        h, w = gray.shape
        min_dim = min(h, w)

        # Smooth to eliminate fine granulation and sensor noise
        ksize = self.config.blur_kernel_size
        if ksize % 2 == 0:
            ksize += 1
        blurred = cv2.GaussianBlur(gray, (ksize, ksize), 0)

        # Threshold to separate solar disk from deep space background
        # Solar continuum disk has high intensity compared to cosmic background
        max_val = float(np.max(blurred))
        thresh_val = max(10.0, max_val * self.config.disk_threshold_ratio)
        _, binary = cv2.threshold(blurred, int(thresh_val), 255, cv2.THRESH_BINARY)

        # Morphological close to bridge any gaps or dark limb features
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Find external contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            # Fallback: assume centered disk
            cx, cy = w / 2.0, h / 2.0
            r = min_dim * 0.45
            mask, eff_mask = self._create_masks(h, w, cx, cy, r)
            return SolarDiskGeometry(cx, cy, r, mask, eff_mask, 0.0, False)

        # The solar disk is by far the largest bright object
        largest_contour = max(contours, key=cv2.contourArea)
        (cx, cy), r = cv2.minEnclosingCircle(largest_contour)

        # Compute circularity / quality check
        area = cv2.contourArea(largest_contour)
        circle_area = np.pi * (r ** 2)
        circularity = (area / circle_area) if circle_area > 0 else 0.0

        min_r = min_dim * self.config.min_radius_ratio
        max_r = min_dim * self.config.max_radius_ratio

        is_valid = (min_r <= r <= max_r) and (0.75 <= circularity <= 1.25)
        confidence = float(np.clip(circularity, 0.0, 1.0)) if is_valid else 0.2

        if not is_valid:
            # Fallback to center if contour was malformed
            cx, cy = w / 2.0, h / 2.0
            r = min_dim * 0.46

        mask, eff_mask = self._create_masks(h, w, cx, cy, r)

        return SolarDiskGeometry(
            center_x=float(cx),
            center_y=float(cy),
            radius=float(r),
            mask=mask,
            effective_mask=eff_mask,
            confidence=confidence,
            is_valid=is_valid,
        )

    def _create_masks(
        self, h: int, w: int, cx: float, cy: float, r: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate full disk mask and effective analysis mask (excluding edge artifacts).
        """
        y_indices, x_indices = np.ogrid[:h, :w]
        dist_sq = (x_indices - cx) ** 2 + (y_indices - cy) ** 2

        # Full disk mask
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[dist_sq <= (r ** 2)] = 255

        # Effective mask excluding outer limb margin
        r_eff = r * (1.0 - self.config.edge_margin_fraction)
        eff_mask = np.zeros((h, w), dtype=np.uint8)
        eff_mask[dist_sq <= (r_eff ** 2)] = 255

        return mask, eff_mask
