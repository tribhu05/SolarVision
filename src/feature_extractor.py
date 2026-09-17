"""
Solar physical feature extraction and coordinate calibration module.
Converts pixel measurements into Stonyhurst heliographic coordinates, foreshortening-corrected
areas in millionths of a solar hemisphere (uHem), and geometric morphology.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

from src.config import SolarPhysicsConfig
from src.disk_detector import SolarDiskGeometry
from src.segmentation import RawActiveRegion


@dataclass
class CalibratedActiveRegion:
    """Fully calibrated solar active region with heliographic coordinates and physical metrics."""
    id: int
    bbox: Tuple[int, int, int, int]              # (x, y, w, h) in pixels
    centroid_pixel: Tuple[float, float]          # (cx, cy) in image pixels
    heliographic_lat: float                      # Latitude B in degrees [-90 to +90]
    heliographic_lon_cmd: float                  # Central Meridian Distance L in degrees [-90 to +90]
    heliocentric_angle_deg: float                # Theta in degrees from disk center
    cos_theta: float                             # Foreshortening factor cos(theta)
    
    # Area measurements
    projected_area_px: int                       # Raw 2D pixel count
    corrected_area_px: float                     # Foreshortening corrected area in pixels
    area_uhem: float                             # Area in millionths of solar hemisphere (uHem)
    umbra_area_uhem: float                       # Umbral area in uHem
    penumbra_area_uhem: float                    # Penumbral area in uHem
    umbra_penumbra_ratio: float                  # Umbra / Penumbra area ratio
    
    # Physical and morphological properties
    longitudinal_extent_deg: float               # East-West extent in degrees
    latitudinal_extent_deg: float                # North-South extent in degrees
    perimeter_pixels: float = 0.0                # Arc length of boundary contour in pixels
    circularity: float = 0.0                     # 4 * pi * Area / Perimeter^2 [0.0 - 1.0]
    spot_count: int = 1                          # Number of distinct spots in this AR
    mean_contrast: float = 0.0                   # Intensity contrast against quiet Sun
    has_penumbra: bool = False                   # True if penumbra detected
    is_bipolar: bool = False                     # True if multiple spots separated longitudinally


class FeatureExtractor:
    """
    Transforms raw pixel detections into calibrated physical solar features
    following standard solar physics conventions (NOAA / SDO).
    """

    def __init__(self, config: Optional[SolarPhysicsConfig] = None):
        self.config = config or SolarPhysicsConfig()

    def extract_features(
        self,
        raw_regions: List[RawActiveRegion],
        disk: SolarDiskGeometry,
    ) -> List[CalibratedActiveRegion]:
        """
        Calibrate all detected active regions with heliographic coordinates and physical metrics.

        Args:
            raw_regions: List of RawActiveRegion detections from SunspotSegmenter
            disk: SolarDiskGeometry with center and radius

        Returns:
            List of CalibratedActiveRegion objects
        """
        calibrated: List[CalibratedActiveRegion] = []

        cx_d, cy_d, r_d = disk.center_x, disk.center_y, disk.radius
        # Total area of visible hemisphere in pixels: 2 * pi * R^2
        hemi_area_px = 2.0 * np.pi * (r_d ** 2)

        b0_rad = np.radians(self.config.b0_angle_deg)

        for reg in raw_regions:
            px, py = reg.centroid

            # Heliocentric coordinates relative to disk center (Cartesian, +X=West, +Y=North)
            dx = (px - cx_d) / max(r_d, 1.0)
            dy = -(py - cy_d) / max(r_d, 1.0)

            rho = np.hypot(dx, dy)
            rho_clamped = min(1.0, max(0.0, rho))

            # Heliocentric angle theta: sin(theta) = rho
            theta_rad = np.arcsin(rho_clamped)
            theta_deg = float(np.degrees(theta_rad))
            cos_theta = float(max(0.08, np.cos(theta_rad)))  # Prevent singularity at extreme limb

            # Heliographic Stonyhurst coordinates
            # B: latitude, L: Central Meridian Distance (CMD)
            if rho_clamped > 1e-6:
                sin_b = np.sin(b0_rad) * np.cos(theta_rad) + np.cos(b0_rad) * (dy / rho_clamped) * np.sin(theta_rad)
            else:
                sin_b = np.sin(b0_rad)

            sin_b = float(np.clip(sin_b, -1.0, 1.0))
            b_rad = np.arcsin(sin_b)
            lat_deg = float(np.degrees(b_rad))

            cos_b = np.cos(b_rad)
            if cos_b > 1e-6 and rho_clamped > 1e-6:
                sin_l = (dx / rho_clamped) * np.sin(theta_rad) / cos_b
                sin_l = float(np.clip(sin_l, -1.0, 1.0))
                l_rad = np.arcsin(sin_l)
                lon_cmd_deg = float(np.degrees(l_rad))
            else:
                lon_cmd_deg = 0.0

            # Correct for foreshortening: A_corr = A_proj / cos(theta)
            corr_area_px = float(reg.total_area_px / cos_theta)
            area_uhem = float((corr_area_px / hemi_area_px) * 1e6)

            # Umbra and penumbra physical areas
            corr_umbra_px = float(reg.umbra_area_px / cos_theta)
            umbra_uhem = float((corr_umbra_px / hemi_area_px) * 1e6)

            corr_penumbra_px = float(reg.penumbra_area_px / cos_theta)
            penumbra_uhem = float((corr_penumbra_px / hemi_area_px) * 1e6)

            up_ratio = float(umbra_uhem / max(penumbra_uhem, 1e-3))
            has_penumbra = (reg.penumbra_area_px > 5) and (penumbra_uhem >= 2.0)

            # Angular extents across bounding box
            bx, by, bw, bh = reg.bbox
            extent_lon_deg = float(np.degrees((bw / r_d) / max(cos_b, 0.2)))
            extent_lat_deg = float(np.degrees(bh / r_d))

            # Bipolarity assessment
            is_bipolar = (reg.spot_count >= 2) and (extent_lon_deg >= 2.0)

            calibrated.append(
                CalibratedActiveRegion(
                    id=reg.id,
                    bbox=reg.bbox,
                    centroid_pixel=(float(px), float(py)),
                    heliographic_lat=round(lat_deg, 2),
                    heliographic_lon_cmd=round(lon_cmd_deg, 2),
                    heliocentric_angle_deg=round(theta_deg, 2),
                    cos_theta=round(cos_theta, 3),
                    projected_area_px=reg.total_area_px,
                    corrected_area_px=round(corr_area_px, 1),
                    area_uhem=round(area_uhem, 2),
                    umbra_area_uhem=round(umbra_uhem, 2),
                    penumbra_area_uhem=round(penumbra_uhem, 2),
                    umbra_penumbra_ratio=round(up_ratio, 3),
                    longitudinal_extent_deg=round(extent_lon_deg, 2),
                    latitudinal_extent_deg=round(extent_lat_deg, 2),
                    perimeter_pixels=reg.perimeter_px,
                    circularity=reg.circularity,
                    spot_count=reg.spot_count,
                    mean_contrast=round(reg.contrast, 3),
                    has_penumbra=has_penumbra,
                    is_bipolar=is_bipolar,
                )
            )

        return calibrated
