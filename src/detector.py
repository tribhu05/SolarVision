"""
SolarVision Sunspot Detection and Feature Extraction Module.
Detects candidate dark regions (sunspots, pores, and active regions) on solar continuum images
using classical Computer Vision, extracts measurable geometric and physical features,
and renders visual overlays while preserving 1:1 coordinate consistency.
"""

from dataclasses import asdict, dataclass, field
import json
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd

from src.config import SegmentationConfig, SolarPhysicsConfig, SolarVisionConfig, load_config
from src.disk_detector import SolarDiskDetector, SolarDiskGeometry
from src.limb_darkening import LimbDarkeningCorrector, LimbCorrectionResult
from src.preprocessor import SolarImagePreprocessor


class DetectionError(Exception):
    """Base exception for sunspot detection errors."""
    pass


@dataclass
class DetectedRegion:
    """
    A detected solar dark region with measured morphological and physical features.
    """
    id: int                                      # Numeric index (1, 2, ...)
    region_id: str                               # Unique string identifier (e.g. 'AR-001')
    bbox: Tuple[int, int, int, int]              # Bounding box (x, y, w, h) in pixels
    centroid: Tuple[float, float]                # Sub-pixel centroid (cx, cy)
    area_pixels: int                             # Discrete pixel footprint
    perimeter_pixels: float                      # Continuous boundary arc length in pixels
    circularity: float                           # 4 * pi * Area / Perimeter^2 [0.0 - 1.0]
    equivalent_diameter_px: float                # sqrt(4 * Area / pi) in pixels
    mean_intensity: float                        # Mean pixel intensity within region
    min_intensity: float                         # Minimum intensity (deepest core)
    contrast: float                              # 1 - (mean_intensity / I_QS) [0.0 - 1.0]
    umbra_area_pixels: int                       # Area of umbra core (I <= T_u)
    penumbra_area_pixels: int                    # Area of penumbra halo (T_u < I <= T_p)
    has_penumbra: bool                           # True if penumbra halo is present
    spot_count: int                              # Number of distinct spot cores in this AR
    
    # Scientific Status & Quality Rating
    scientific_status: str                       # 'Confirmed Sunspot Group', 'Pore / Developing Spot', 'Candidate Dark Region'
    confidence: float                            # Quality confidence score [0.0 - 1.0]
    
    # Physical Heliographic Calibration (when solar disk geometry is available)
    heliographic_lat: float = 0.0                # Stonyhurst Latitude B [-90 to +90 deg]
    heliographic_lon_cmd: float = 0.0            # Central Meridian Distance L [-90 to +90 deg]
    heliocentric_angle_deg: float = 0.0          # Angular distance theta from disk center
    cos_theta: float = 1.0                       # Foreshortening factor cos(theta)
    area_uhem: float = 0.0                       # Physical area in millionths of solar hemisphere
    umbra_area_uhem: float = 0.0                 # Umbra area in uHem
    penumbra_area_uhem: float = 0.0              # Penumbra area in uHem
    
    # Contour and Cropped ROI
    contour: np.ndarray = field(default_factory=lambda: np.array([]))
    patch: Optional[np.ndarray] = None           # Cropped image patch with margin

    def to_dict(self) -> Dict[str, Any]:
        """Convert region metrics to serializable dictionary (excluding numpy arrays)."""
        return {
            "id": int(self.id),
            "region_id": str(self.region_id),
            "bbox_x": int(self.bbox[0]),
            "bbox_y": int(self.bbox[1]),
            "bbox_w": int(self.bbox[2]),
            "bbox_h": int(self.bbox[3]),
            "centroid_x": round(float(self.centroid[0]), 2),
            "centroid_y": round(float(self.centroid[1]), 2),
            "area_pixels": int(self.area_pixels),
            "perimeter_pixels": round(float(self.perimeter_pixels), 2),
            "circularity": round(float(self.circularity), 3),
            "equivalent_diameter_px": round(float(self.equivalent_diameter_px), 2),
            "mean_intensity": round(float(self.mean_intensity), 1),
            "min_intensity": round(float(self.min_intensity), 1),
            "contrast": round(float(self.contrast), 3),
            "umbra_area_pixels": int(self.umbra_area_pixels),
            "penumbra_area_pixels": int(self.penumbra_area_pixels),
            "has_penumbra": bool(self.has_penumbra),
            "spot_count": int(self.spot_count),
            "scientific_status": str(self.scientific_status),
            "confidence": round(float(self.confidence), 2),
            "heliographic_lat_deg": float(self.heliographic_lat),
            "heliographic_lon_cmd_deg": float(self.heliographic_lon_cmd),
            "heliocentric_angle_deg": float(self.heliocentric_angle_deg),
            "cos_theta": float(self.cos_theta),
            "area_uhem": float(self.area_uhem),
            "umbra_area_uhem": float(self.umbra_area_uhem),
            "penumbra_area_uhem": float(self.penumbra_area_uhem),
        }


@dataclass
class DetectionOutput:
    """Structured container for detection pipeline results."""
    regions: List[DetectedRegion]
    binary_mask: np.ndarray                      # Combined binary mask of all detected spots
    umbra_mask: np.ndarray                       # Binary mask of umbral cores
    penumbra_mask: np.ndarray                    # Binary mask of penumbral borders
    annotated_image: np.ndarray                  # Rendered visualization
    quiet_sun_intensity: float
    thresholds: Dict[str, float]                 # Thresholds applied in pipeline
    solar_disk: SolarDiskGeometry
    total_detected_count: int
    confirmed_sunspot_count: int
    pore_count: int
    candidate_count: int
    is_spotless: bool                            # True if no active regions were detected

    def to_dataframe(self) -> pd.DataFrame:
        """Export detected regions as a pandas DataFrame."""
        if not self.regions:
            return pd.DataFrame()
        return pd.DataFrame([r.to_dict() for r in self.regions])

    def to_json(self) -> str:
        """Export detection summary and regions to JSON."""
        summary = {
            "total_detected": self.total_detected_count,
            "confirmed_sunspots": self.confirmed_sunspot_count,
            "pores": self.pore_count,
            "candidates": self.candidate_count,
            "is_spotless": self.is_spotless,
            "quiet_sun_intensity": self.quiet_sun_intensity,
            "thresholds": self.thresholds,
            "disk_geometry": {
                "center_x": self.solar_disk.center_x,
                "center_y": self.solar_disk.center_y,
                "radius": self.solar_disk.radius,
            },
            "regions": [r.to_dict() for r in self.regions],
        }
        return json.dumps(summary, indent=2)

    def get_region_by_id(self, region_id: str) -> Optional[DetectedRegion]:
        """Lookup a region by its unique identifier (e.g. 'AR-001')."""
        for r in self.regions:
            if r.region_id == region_id or str(r.id) == region_id:
                return r
        return None


class SunspotDetector:
    """
    Classical Computer Vision detector and feature extractor for solar active regions.
    Combines adaptive intensity thresholding, morphological noise suppression,
    connected component analysis, physical feature calibration, and annotation rendering.
    """

    def __init__(
        self,
        config: Optional[SegmentationConfig] = None,
        physics_config: Optional[SolarPhysicsConfig] = None,
        full_config: Optional[SolarVisionConfig] = None,
    ):
        base_cfg = full_config or load_config()
        self.config = config or base_cfg.segmentation
        self.physics_config = physics_config or base_cfg.solar_physics
        self.disk_detector = SolarDiskDetector(base_cfg.disk_detection)
        self.limb_corrector = LimbDarkeningCorrector(base_cfg.limb_darkening)
        self.preprocessor = SolarImagePreprocessor(base_cfg.preprocessing, base_cfg)

    @staticmethod
    def calculate_circularity(area: float, perimeter: float) -> float:
        """
        Compute standard circularity / isoperimetric compactness:
        C = 4 * pi * Area / Perimeter^2
        Value is 1.0 for a perfect circle and approaches 0.0 for highly elongated or complex groups.
        """
        if perimeter <= 0.0 or area <= 0.0:
            return 0.0
        circ = (4.0 * np.pi * area) / (perimeter ** 2)
        return float(np.clip(circ, 0.0, 1.0))

    def extract_roi_patch(
        self, image: np.ndarray, bbox: Tuple[int, int, int, int], padding: int = 8
    ) -> np.ndarray:
        """
        Extract a localized image patch around the detected active region with margin.
        """
        h, w = image.shape[:2]
        bx, by, bw, bh = bbox
        x1 = max(0, bx - padding)
        y1 = max(0, by - padding)
        x2 = min(w, bx + bw + padding)
        y2 = min(h, by + bh + padding)
        return image[y1:y2, x1:x2].copy()

    def detect(
        self,
        image: np.ndarray,
        disk: Optional[SolarDiskGeometry] = None,
        limb_result: Optional[LimbCorrectionResult] = None,
        extract_patches: bool = True,
    ) -> DetectionOutput:
        """
        Execute full classical Computer Vision sunspot detection and feature extraction.

        Args:
            image: 2D grayscale or 3D BGR solar image
            disk: Optional precomputed SolarDiskGeometry
            limb_result: Optional precomputed LimbCorrectionResult
            extract_patches: Whether to crop sub-image ROI patches for each region

        Returns:
            DetectionOutput containing detected regions, masks, and annotated image
        """
        if image is None or (isinstance(image, np.ndarray) and image.size == 0):
            raise DetectionError("Input image is empty or None.")

        # Ensure image is in BGR format for display and 2D grayscale for analysis
        if image.ndim == 2:
            gray = image.copy()
            bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            bgr = image.copy()
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        h, w = gray.shape

        # Step 1: Detect solar disk if not provided
        if disk is None:
            disk = self.disk_detector.detect(gray)

        # Step 2: Limb darkening correction (photospheric flat-fielding)
        if limb_result is None:
            # Zero out background sky first
            isolated_gray = np.zeros_like(gray)
            isolated_gray[disk.effective_mask > 0] = gray[disk.effective_mask > 0]
            limb_result = self.limb_corrector.correct(isolated_gray, disk)

        flat = limb_result.flattened_image
        q_sun = max(limb_result.quiet_sun_intensity, 1.0)

        # Step 3: Intensity thresholding for candidate dark regions
        # Umbra: intensely dark core (typically <= 0.58 of quiet Sun)
        # Penumbra: filamentary halo (typically 0.58 to 0.88 of quiet Sun)
        t_umbra = q_sun * self.config.umbra_threshold_factor
        t_penumbra = q_sun * self.config.penumbra_threshold_factor

        valid_mask = disk.effective_mask > 0

        umbra_raw = np.zeros((h, w), dtype=np.uint8)
        penumbra_raw = np.zeros((h, w), dtype=np.uint8)

        umbra_raw[valid_mask & (flat > 0) & (flat <= t_umbra)] = 255
        penumbra_raw[valid_mask & (flat > t_umbra) & (flat <= t_penumbra)] = 255

        combined_raw = cv2.bitwise_or(umbra_raw, penumbra_raw)

        # Step 4: Morphological noise filtering
        # Elliptical opening removes random single-pixel granulation fluctuations
        k_size = self.config.morphology_kernel_size
        if k_size % 2 == 0:
            k_size += 1
        morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        combined_cleaned = cv2.morphologyEx(combined_raw, cv2.MORPH_OPEN, morph_kernel)

        umbra_cleaned = cv2.bitwise_and(umbra_raw, combined_cleaned)
        penumbra_cleaned = cv2.bitwise_and(penumbra_raw, combined_cleaned)

        # Step 5: Connected components to filter candidate spots by size
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            combined_cleaned, connectivity=8
        )

        min_area = self.config.min_sunspot_area_pixels
        max_area = self.config.max_sunspot_area_pixels

        filtered_mask = np.zeros((h, w), dtype=np.uint8)
        candidate_spots: List[dict] = []

        for label_idx in range(1, num_labels):
            spot_area = stats[label_idx, cv2.CC_STAT_AREA]
            if min_area <= spot_area <= max_area:
                filtered_mask[labels == label_idx] = 255
                sx = stats[label_idx, cv2.CC_STAT_LEFT]
                sy = stats[label_idx, cv2.CC_STAT_TOP]
                sw = stats[label_idx, cv2.CC_STAT_WIDTH]
                sh = stats[label_idx, cv2.CC_STAT_HEIGHT]
                scx, scy = centroids[label_idx]
                candidate_spots.append({
                    "bbox": (int(sx), int(sy), int(sw), int(sh)),
                    "centroid": (float(scx), float(scy)),
                    "area": int(spot_area),
                    "label": label_idx,
                })

        final_umbra_mask = cv2.bitwise_and(umbra_cleaned, filtered_mask)
        final_penumbra_mask = cv2.bitwise_and(penumbra_cleaned, filtered_mask)

        # Step 6: Spatial clustering of connected components into Active Regions
        raw_regions = self._cluster_spots(
            candidate_spots=candidate_spots,
            combined_mask=filtered_mask,
            umbra_mask=final_umbra_mask,
            penumbra_mask=final_penumbra_mask,
            flat_image=flat,
            quiet_sun_level=q_sun,
            solar_radius=disk.radius,
        )

        # Step 7: Feature extraction and calibration for each detected region
        detected_regions: List[DetectedRegion] = []
        cx_d, cy_d, r_d = disk.center_x, disk.center_y, disk.radius
        hemi_area_px = 2.0 * np.pi * (r_d ** 2)
        b0_rad = np.radians(self.physics_config.b0_angle_deg)

        for r_idx, reg_dict in enumerate(raw_regions, start=1):
            px, py = reg_dict["centroid"]
            tot_px = reg_dict["area_pixels"]
            perim = reg_dict["perimeter_pixels"]
            circ = self.calculate_circularity(tot_px, perim)
            eq_diam = float(np.sqrt(4.0 * tot_px / np.pi))

            # Heliocentric geometry relative to disk center (+X=West, +Y=North)
            dx = (px - cx_d) / max(r_d, 1.0)
            dy = -(py - cy_d) / max(r_d, 1.0)
            rho = min(1.0, max(0.0, np.hypot(dx, dy)))

            theta_rad = np.arcsin(rho)
            theta_deg = float(np.degrees(theta_rad))
            cos_theta = float(max(0.08, np.cos(theta_rad)))

            # Stonyhurst heliographic coordinates
            if rho > 1e-6:
                sin_b = np.sin(b0_rad) * np.cos(theta_rad) + np.cos(b0_rad) * (dy / rho) * np.sin(theta_rad)
            else:
                sin_b = np.sin(b0_rad)

            sin_b = float(np.clip(sin_b, -1.0, 1.0))
            b_rad = np.arcsin(sin_b)
            lat_deg = float(np.degrees(b_rad))

            cos_b = np.cos(b_rad)
            if cos_b > 1e-6 and rho > 1e-6:
                sin_l = (dx / rho) * np.sin(theta_rad) / cos_b
                sin_l = float(np.clip(sin_l, -1.0, 1.0))
                l_rad = np.arcsin(sin_l)
                lon_cmd_deg = float(np.degrees(l_rad))
            else:
                lon_cmd_deg = 0.0

            # Foreshortening-corrected physical areas
            corr_area_px = tot_px / cos_theta
            area_uhem = float((corr_area_px / hemi_area_px) * 1e6)

            corr_umbra_px = reg_dict["umbra_area_pixels"] / cos_theta
            umbra_uhem = float((corr_umbra_px / hemi_area_px) * 1e6)

            corr_penumbra_px = reg_dict["penumbra_area_pixels"] / cos_theta
            penumbra_uhem = float((corr_penumbra_px / hemi_area_px) * 1e6)

            has_pen = (reg_dict["penumbra_area_pixels"] >= 5) and (penumbra_uhem >= 2.0)
            contrast = reg_dict["contrast"]

            # Scientific status classification (Avoid assuming every dark spot is a confirmed sunspot)
            if has_pen and reg_dict["umbra_area_pixels"] > 0:
                scientific_status = "Confirmed Sunspot Group"
                conf = float(np.clip(0.75 + 0.20 * contrast, 0.75, 0.98))
            elif area_uhem >= 25.0 and contrast >= 0.25:
                scientific_status = "Confirmed Sunspot Group"
                conf = float(np.clip(0.70 + 0.25 * contrast, 0.70, 0.95))
            elif area_uhem < 25.0 and contrast >= 0.20:
                scientific_status = "Pore / Developing Spot"
                conf = float(np.clip(0.60 + 0.25 * contrast, 0.60, 0.88))
            else:
                scientific_status = "Candidate Dark Region"
                conf = float(np.clip(0.40 + 0.30 * contrast, 0.30, 0.70))

            # Crop ROI patch if requested
            patch_img = self.extract_roi_patch(bgr, reg_dict["bbox"]) if extract_patches else None

            # Unique region ID
            reg_id_str = f"AR-{r_idx:03d}"

            detected_regions.append(
                DetectedRegion(
                    id=r_idx,
                    region_id=reg_id_str,
                    bbox=reg_dict["bbox"],
                    centroid=(float(px), float(py)),
                    area_pixels=tot_px,
                    perimeter_pixels=perim,
                    circularity=circ,
                    equivalent_diameter_px=round(eq_diam, 2),
                    mean_intensity=reg_dict["mean_intensity"],
                    min_intensity=reg_dict["min_intensity"],
                    contrast=contrast,
                    umbra_area_pixels=reg_dict["umbra_area_pixels"],
                    penumbra_area_pixels=reg_dict["penumbra_area_pixels"],
                    has_penumbra=has_pen,
                    spot_count=reg_dict["spot_count"],
                    scientific_status=scientific_status,
                    confidence=round(conf, 2),
                    heliographic_lat=round(lat_deg, 2),
                    heliographic_lon_cmd=round(lon_cmd_deg, 2),
                    heliocentric_angle_deg=round(theta_deg, 2),
                    cos_theta=round(cos_theta, 3),
                    area_uhem=round(area_uhem, 2),
                    umbra_area_uhem=round(umbra_uhem, 2),
                    penumbra_area_uhem=round(penumbra_uhem, 2),
                    contour=reg_dict["contour"],
                    patch=patch_img,
                )
            )

        # Step 8: Render Annotated Visualization
        annotated = self.render_annotated_image(
            base_image=bgr,
            regions=detected_regions,
            disk=disk,
            umbra_mask=final_umbra_mask,
            penumbra_mask=final_penumbra_mask,
        )

        # Summary counts
        confirmed_count = sum(1 for r in detected_regions if r.scientific_status == "Confirmed Sunspot Group")
        pore_count = sum(1 for r in detected_regions if r.scientific_status == "Pore / Developing Spot")
        candidate_count = sum(1 for r in detected_regions if r.scientific_status == "Candidate Dark Region")

        return DetectionOutput(
            regions=detected_regions,
            binary_mask=filtered_mask,
            umbra_mask=final_umbra_mask,
            penumbra_mask=final_penumbra_mask,
            annotated_image=annotated,
            quiet_sun_intensity=round(q_sun, 1),
            thresholds={
                "umbra_threshold": round(t_umbra, 1),
                "penumbra_threshold": round(t_penumbra, 1),
                "umbra_factor": self.config.umbra_threshold_factor,
                "penumbra_factor": self.config.penumbra_threshold_factor,
            },
            solar_disk=disk,
            total_detected_count=len(detected_regions),
            confirmed_sunspot_count=confirmed_count,
            pore_count=pore_count,
            candidate_count=candidate_count,
            is_spotless=len(detected_regions) == 0,
        )

    def _cluster_spots(
        self,
        candidate_spots: List[dict],
        combined_mask: np.ndarray,
        umbra_mask: np.ndarray,
        penumbra_mask: np.ndarray,
        flat_image: np.ndarray,
        quiet_sun_level: float,
        solar_radius: float,
    ) -> List[dict]:
        """
        Group nearby connected components into single Active Regions.
        """
        if not candidate_spots:
            return []

        # Clustering distance in pixels from angular threshold in degrees
        max_dist_px = solar_radius * (self.config.clustering_distance_deg * np.pi / 180.0)

        n = len(candidate_spots)
        parent = list(range(n))

        def find(i: int) -> int:
            if parent[i] == i:
                return i
            parent[i] = find(parent[i])
            return parent[i]

        def union(i: int, j: int):
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj

        for i in range(n):
            c1 = candidate_spots[i]["centroid"]
            for j in range(i + 1, n):
                c2 = candidate_spots[j]["centroid"]
                if np.hypot(c1[0] - c2[0], c1[1] - c2[1]) <= max_dist_px:
                    union(i, j)

        groups: dict = {}
        for i in range(n):
            root = find(i)
            groups.setdefault(root, []).append(candidate_spots[i])

        clustered_regions: List[dict] = []

        for group in groups.values():
            min_x = min(s["bbox"][0] for s in group)
            min_y = min(s["bbox"][1] for s in group)
            max_x = max(s["bbox"][0] + s["bbox"][2] for s in group)
            max_y = max(s["bbox"][1] + s["bbox"][3] for s in group)
            bbox = (int(min_x), int(min_y), int(max_x - min_x), int(max_y - min_y))

            tot_spot_area = sum(s["area"] for s in group)
            wcx = sum(s["centroid"][0] * s["area"] for s in group) / max(tot_spot_area, 1)
            wcy = sum(s["centroid"][1] * s["area"] for s in group) / max(tot_spot_area, 1)

            # Region mask
            reg_mask = np.zeros_like(combined_mask)
            for s in group:
                reg_mask[min_y:max_y, min_x:max_x] = combined_mask[min_y:max_y, min_x:max_x]

            reg_umbra = cv2.bitwise_and(umbra_mask, reg_mask)
            reg_penumbra = cv2.bitwise_and(penumbra_mask, reg_mask)

            umbra_px = int(np.count_nonzero(reg_umbra))
            penumbra_px = int(np.count_nonzero(reg_penumbra))
            tot_px = int(np.count_nonzero(reg_mask))

            reg_pixels = flat_image[reg_mask > 0]
            if len(reg_pixels) > 0:
                min_int = float(np.min(reg_pixels))
                mean_int = float(np.mean(reg_pixels))
                contrast = float(np.clip(1.0 - (mean_int / max(quiet_sun_level, 1.0)), 0.0, 1.0))
            else:
                min_int, mean_int, contrast = 0.0, 0.0, 0.0

            # Contrast filter check
            if contrast < self.config.min_contrast:
                continue

            cnts, _ = cv2.findContours(reg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            largest_cnt = max(cnts, key=cv2.contourArea) if cnts else np.array([])
            perim = float(cv2.arcLength(largest_cnt, closed=True)) if len(largest_cnt) > 0 else 0.0

            clustered_regions.append({
                "bbox": bbox,
                "centroid": (float(wcx), float(wcy)),
                "area_pixels": tot_px,
                "umbra_area_pixels": umbra_px,
                "penumbra_area_pixels": penumbra_px,
                "perimeter_pixels": round(perim, 2),
                "min_intensity": round(min_int, 1),
                "mean_intensity": round(mean_int, 1),
                "contrast": round(contrast, 3),
                "contour": largest_cnt,
                "spot_count": len(group),
            })

        # Sort largest to smallest
        clustered_regions.sort(key=lambda r: r["area_pixels"], reverse=True)
        return clustered_regions

    def render_annotated_image(
        self,
        base_image: np.ndarray,
        regions: List[DetectedRegion],
        disk: SolarDiskGeometry,
        umbra_mask: np.ndarray,
        penumbra_mask: np.ndarray,
        draw_bboxes: bool = True,
        draw_contours: bool = True,
        draw_centroids: bool = True,
        draw_labels: bool = True,
    ) -> np.ndarray:
        """
        Draw bounding boxes, boundary contours, centroids, and classification badges
        on the input solar image with exact pixel alignment.
        """
        annotated = base_image.copy()

        # Colorize umbra and penumbra layers
        annotated[penumbra_mask > 0] = [0, 215, 255]   # Amber / Yellow
        annotated[umbra_mask > 0] = [0, 0, 230]        # Deep Red

        # Draw solar disk boundary
        cx, cy, r = int(disk.center_x), int(disk.center_y), int(disk.radius)
        cv2.circle(annotated, (cx, cy), r, (120, 120, 120), 1, cv2.LINE_AA)

        if not regions:
            # Display Spotless Sun badge
            badge_text = "Spotless Solar Disk (0 Active Regions)"
            cv2.putText(
                annotated, badge_text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA
            )
            return annotated

        for reg in regions:
            bx, by, bw, bh = reg.bbox
            c_x, c_y = int(reg.centroid[0]), int(reg.centroid[1])

            # 1. Contours
            if draw_contours and len(reg.contour) > 0:
                cv2.drawContours(annotated, [reg.contour], -1, (255, 255, 0), 1, cv2.LINE_AA)

            # 2. Bounding Box
            if draw_bboxes:
                color = (0, 255, 0) if reg.scientific_status == "Confirmed Sunspot Group" else (0, 200, 255)
                cv2.rectangle(annotated, (bx - 2, by - 2), (bx + bw + 2, by + bh + 2), color, 2)

            # 3. Centroid marker
            if draw_centroids:
                cv2.drawMarker(annotated, (c_x, c_y), (255, 0, 0), cv2.MARKER_CROSS, 8, 1, cv2.LINE_AA)

            # 4. Text Badge
            if draw_labels:
                label = f"{reg.region_id} ({reg.area_uhem:.0f}uH, C:{reg.circularity:.2f})"
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
                label_y = max(lh + 4, by - 5)
                # Background badge rectangle
                cv2.rectangle(
                    annotated,
                    (bx, label_y - lh - 3),
                    (bx + lw + 4, label_y + 2),
                    (0, 0, 0),
                    -1,
                )
                cv2.putText(
                    annotated, label, (bx + 2, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA
                )

        return annotated
