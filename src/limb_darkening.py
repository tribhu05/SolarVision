"""
Photospheric limb darkening correction and flat-field normalization.
Removes the radial intensity drop-off across the solar disk to enable uniform thresholding.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import numpy as np

from src.config import LimbDarkeningConfig
from src.disk_detector import SolarDiskGeometry


@dataclass
class LimbCorrectionResult:
    flattened_image: np.ndarray       # Float32 normalized image (~quiet-sun intensity = 1.0)
    flattened_uint8: np.ndarray       # Scaled to 0-255 for visualization
    correction_surface: np.ndarray    # Estimated limb darkening model surface
    quiet_sun_intensity: float        # Median intensity of central quiet Sun
    mu_map: np.ndarray                # cos(theta) heliocentric angle map
    radial_distance_map: np.ndarray   # r / R normalized radial distance

    @property
    def flattened_intensity(self) -> np.ndarray:
        """Convenience alias for flattened_uint8."""
        return self.flattened_uint8



class LimbDarkeningCorrector:
    """
    Corrects photospheric limb darkening on solar continuum disks using
    either a theoretical linear-cosine law (Eddington approximation) or
    an empirical radial profile fit.
    """

    def __init__(self, config: Optional[LimbDarkeningConfig] = None):
        self.config = config or LimbDarkeningConfig()

    def correct(
        self, image: np.ndarray, disk: SolarDiskGeometry
    ) -> LimbCorrectionResult:
        """
        Apply limb darkening compensation to the solar disk.

        Args:
            image: 2D grayscale image (or 3D BGR)
            disk: Detected solar disk geometry

        Returns:
            LimbCorrectionResult with flattened image and solar geometry maps
        """
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            gray = image.astype(np.float32)

        h, w = gray.shape
        cx, cy, r = disk.center_x, disk.center_y, disk.radius

        # Calculate radial distance map normalized by solar radius R
        y_coords, x_coords = np.ogrid[:h, :w]
        r_dist = np.sqrt((x_coords - cx) ** 2 + (y_coords - cy) ** 2)
        r_norm = r_dist / max(r, 1e-5)

        # Clip r_norm to [0, 1] inside disk to avoid imaginary numbers in sqrt
        r_norm_clipped = np.clip(r_norm, 0.0, 1.0)

        # Heliocentric angle cosine: mu = cos(theta) = sqrt(1 - (r/R)^2)
        mu_map = np.sqrt(np.maximum(0.0, 1.0 - (r_norm_clipped ** 2)))

        # Estimate central quiet-Sun intensity from the central region (r < 0.5 * R)
        inner_mask = (r_norm <= self.config.inner_disk_sample_radius) & (disk.mask > 0)
        if np.any(inner_mask):
            # Use a robust percentile (e.g. 75th-90th) to avoid spots depressing the quiet-Sun estimate
            inner_pixels = gray[inner_mask]
            q_sun = float(np.percentile(inner_pixels, self.config.quiet_sun_percentile))
        else:
            q_sun = float(np.percentile(gray[disk.mask > 0], 85.0)) if np.any(disk.mask > 0) else 180.0

        # Theoretical limb darkening: I(mu) = I_0 * [1 - u * (1 - mu)]
        u = self.config.u_coefficient
        theoretical_profile = 1.0 - u * (1.0 - mu_map)
        theoretical_profile = np.maximum(theoretical_profile, 0.1)  # Guard against division by zero

        correction_surface = q_sun * theoretical_profile

        # Flattened image: divide observed intensity by the limb darkening shape
        # Only valid within effective disk mask
        flattened = np.zeros((h, w), dtype=np.float32)
        valid_pixels = (disk.effective_mask > 0) & (theoretical_profile > 0.1)

        flattened[valid_pixels] = (gray[valid_pixels] / theoretical_profile[valid_pixels])

        # Compute uint8 display version normalized to ~[0, 255]
        # Reference quiet-Sun intensity to ~180-200 in uint8 representation
        display_scale = 190.0 / max(q_sun, 1.0)
        flattened_scaled = flattened * display_scale
        flattened_uint8 = np.clip(flattened_scaled, 0, 255).astype(np.uint8)

        # Zero out regions outside the solar disk mask
        flattened[~valid_pixels] = 0.0
        flattened_uint8[disk.mask == 0] = 0

        return LimbCorrectionResult(
            flattened_image=flattened,
            flattened_uint8=flattened_uint8,
            correction_surface=correction_surface,
            quiet_sun_intensity=q_sun,
            mu_map=mu_map,
            radial_distance_map=r_norm,
        )
