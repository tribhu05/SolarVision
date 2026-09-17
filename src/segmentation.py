"""
Solar active region and sunspot segmentation module.
Performs dual-thresholding to isolate umbra (dark core) and penumbra (outer boundary),
followed by morphological filtering and active region spatial clustering.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import cv2
import numpy as np

from src.config import SegmentationConfig
from src.disk_detector import SolarDiskGeometry
from src.limb_darkening import LimbCorrectionResult


@dataclass
class RawActiveRegion:
    """A detected active region candidate before physical coordinate calibration."""
    id: int
    bbox: Tuple[int, int, int, int]           # (x, y, w, h)
    centroid: Tuple[float, float]             # (cx, cy)
    total_area_px: int
    umbra_area_px: int
    penumbra_area_px: int
    perimeter_px: float = 0.0                 # Arc length of outer boundary
    circularity: float = 0.0                  # 4 * pi * Area / Perimeter^2
    min_intensity: float = 0.0
    mean_intensity: float = 0.0
    contrast: float = 0.0                     # 1 - (mean_intensity / I_QS)
    contour: np.ndarray = field(default_factory=lambda: np.array([]))
    spot_count: int = 1


@dataclass
class SegmentationResult:
    umbra_mask: np.ndarray                   # Binary mask (255 where umbra)
    penumbra_mask: np.ndarray                # Binary mask (255 where penumbra)
    combined_mask: np.ndarray                # Binary mask (255 where any spot)
    regions: List[RawActiveRegion]
    quiet_sun_level: float
    umbra_threshold: float
    penumbra_threshold: float


class SunspotSegmenter:
    """
    Segments solar continuum images into umbral cores, penumbral boundaries,
    and groups them into distinct Active Regions.
    """

    def __init__(self, config: Optional[SegmentationConfig] = None):
        self.config = config or SegmentationConfig()

    def segment(
        self,
        limb_result: LimbCorrectionResult,
        disk: SolarDiskGeometry,
    ) -> SegmentationResult:
        """
        Segment the flattened solar image into umbra, penumbra, and active regions.

        Args:
            limb_result: LimbCorrectionResult from limb darkening correction
            disk: SolarDiskGeometry with effective solar disk mask

        Returns:
            SegmentationResult containing binary masks and raw active regions
        """
        flat = limb_result.flattened_image
        q_sun = limb_result.quiet_sun_intensity

        # Determine intensity thresholds
        t_umbra = q_sun * self.config.umbra_threshold_factor
        t_penumbra = q_sun * self.config.penumbra_threshold_factor

        valid_mask = disk.effective_mask > 0

        # Raw thresholding on flattened image
        umbra_raw = np.zeros_like(flat, dtype=np.uint8)
        penumbra_raw = np.zeros_like(flat, dtype=np.uint8)

        umbra_raw[valid_mask & (flat > 0) & (flat <= t_umbra)] = 255
        penumbra_raw[valid_mask & (flat > t_umbra) & (flat <= t_penumbra)] = 255

        combined_raw = cv2.bitwise_or(umbra_raw, penumbra_raw)

        # Morphological noise removal to discard granulation noise
        k_size = self.config.morphology_kernel_size
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        combined_cleaned = cv2.morphologyEx(combined_raw, cv2.MORPH_OPEN, kernel)

        # Re-mask umbra and penumbra using the cleaned combined footprint
        umbra_cleaned = cv2.bitwise_and(umbra_raw, combined_cleaned)
        penumbra_cleaned = cv2.bitwise_and(penumbra_raw, combined_cleaned)

        # Filter components smaller than min_sunspot_area_pixels
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            combined_cleaned, connectivity=8
        )

        filtered_combined = np.zeros_like(combined_cleaned)
        filtered_spots: List[dict] = []

        min_area = self.config.min_sunspot_area_pixels

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                filtered_combined[labels == i] = 255
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                cx, cy = centroids[i]
                filtered_spots.append({
                    "bbox": (x, y, w, h),
                    "centroid": (float(cx), float(cy)),
                    "area": int(area),
                    "label": i,
                })

        # Final masks after area filtering
        umbra_final = cv2.bitwise_and(umbra_cleaned, filtered_combined)
        penumbra_final = cv2.bitwise_and(penumbra_cleaned, filtered_combined)

        # Cluster spots into Active Regions based on spatial proximity
        regions = self._cluster_active_regions(
            filtered_spots,
            filtered_combined,
            umbra_final,
            penumbra_final,
            flat,
            q_sun,
            disk.radius,
        )

        return SegmentationResult(
            umbra_mask=umbra_final,
            penumbra_mask=penumbra_final,
            combined_mask=filtered_combined,
            regions=regions,
            quiet_sun_level=float(q_sun),
            umbra_threshold=float(t_umbra),
            penumbra_threshold=float(t_penumbra),
        )

    def _cluster_active_regions(
        self,
        spots: List[dict],
        combined_mask: np.ndarray,
        umbra_mask: np.ndarray,
        penumbra_mask: np.ndarray,
        flat_image: np.ndarray,
        q_sun: float,
        solar_radius: float,
    ) -> List[RawActiveRegion]:
        """
        Group individual spots into Active Regions using spatial proximity.
        """
        if not spots:
            return []

        # Convert clustering angular threshold (deg) to pixel distance
        # theta = dist / R -> dist = R * (deg * pi / 180)
        max_dist_px = solar_radius * (self.config.clustering_distance_deg * np.pi / 180.0)

        n = len(spots)
        parent = list(range(n))

        def find(i: int) -> int:
            if parent[i] == i:
                return i
            parent[i] = find(parent[i])
            return parent[i]

        def union(i: int, j: int):
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                parent[root_i] = root_j

        # Connect spots whose centroids are within max_dist_px
        for i in range(n):
            c1 = spots[i]["centroid"]
            for j in range(i + 1, n):
                c2 = spots[j]["centroid"]
                d = np.hypot(c1[0] - c2[0], c1[1] - c2[1])
                if d <= max_dist_px:
                    union(i, j)

        # Group spots by root
        groups: dict = {}
        for i in range(n):
            root = find(i)
            groups.setdefault(root, []).append(spots[i])

        active_regions: List[RawActiveRegion] = []

        for reg_id, group in enumerate(groups.values(), start=1):
            # Compute collective bounding box
            min_x = min(s["bbox"][0] for s in group)
            min_y = min(s["bbox"][1] for s in group)
            max_x = max(s["bbox"][0] + s["bbox"][2] for s in group)
            max_y = max(s["bbox"][1] + s["bbox"][3] for s in group)
            bbox = (min_x, min_y, max_x - min_x, max_y - min_y)

            # Area-weighted centroid
            tot_spot_area = sum(s["area"] for s in group)
            weighted_cx = sum(s["centroid"][0] * s["area"] for s in group) / max(tot_spot_area, 1)
            weighted_cy = sum(s["centroid"][1] * s["area"] for s in group) / max(tot_spot_area, 1)

            # Extract region submask for exact statistics
            reg_mask = np.zeros_like(combined_mask)
            for s in group:
                reg_mask[min_y:max_y, min_x:max_x] = combined_mask[min_y:max_y, min_x:max_x]

            reg_umbra = cv2.bitwise_and(umbra_mask, reg_mask)
            reg_penumbra = cv2.bitwise_and(penumbra_mask, reg_mask)

            umbra_px = int(np.count_nonzero(reg_umbra))
            penumbra_px = int(np.count_nonzero(reg_penumbra))
            tot_px = int(np.count_nonzero(reg_mask))

            # Intensity statistics
            region_pixels = flat_image[reg_mask > 0]
            if len(region_pixels) > 0:
                min_int = float(np.min(region_pixels))
                mean_int = float(np.mean(region_pixels))
                contrast = float(np.clip(1.0 - (mean_int / max(q_sun, 1.0)), 0.0, 1.0))
            else:
                min_int = 0.0
                mean_int = 0.0
                contrast = 0.0

            # Find outer contour of this group
            cnts, _ = cv2.findContours(reg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            largest_cnt = max(cnts, key=cv2.contourArea) if cnts else np.array([])

            if len(largest_cnt) > 0:
                perimeter_px = float(cv2.arcLength(largest_cnt, closed=True))
                circ = (4.0 * np.pi * tot_px) / max(perimeter_px ** 2, 1e-4) if perimeter_px > 0 else 0.0
                circularity = float(np.clip(circ, 0.0, 1.0))
            else:
                perimeter_px = 0.0
                circularity = 0.0

            # Filter out regions with insufficient contrast or exceeding max area
            if contrast < self.config.min_contrast:
                continue
            if tot_px > self.config.max_sunspot_area_pixels:
                continue

            active_regions.append(
                RawActiveRegion(
                    id=reg_id,
                    bbox=bbox,
                    centroid=(float(weighted_cx), float(weighted_cy)),
                    total_area_px=tot_px,
                    umbra_area_px=umbra_px,
                    penumbra_area_px=penumbra_px,
                    perimeter_px=round(perimeter_px, 2),
                    circularity=round(circularity, 3),
                    min_intensity=min_int,
                    mean_intensity=mean_int,
                    contrast=round(contrast, 3),
                    contour=largest_cnt,
                    spot_count=len(group),
                )
            )

        # Sort regions by total area descending
        active_regions.sort(key=lambda r: r.total_area_px, reverse=True)
        # Re-number IDs
        for idx, r in enumerate(active_regions, start=1):
            r.id = idx

        return active_regions
