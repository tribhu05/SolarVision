"""
Test suite for SunspotDetector and Feature Extraction.
Validates area, perimeter, circularity, centroid, bounding box,
spotless image handling, and structured exports.
"""

import json
from pathlib import Path
import cv2
import numpy as np
import pytest

from src.detector import (
    SunspotDetector,
    DetectedRegion,
    DetectionOutput,
    DetectionError,
)
from src.config import SegmentationConfig, SolarVisionConfig
from src.disk_detector import SolarDiskDetector


@pytest.fixture
def detector():
    return SunspotDetector()


@pytest.fixture
def synthetic_disk_with_spot():
    """Create a 400x400 synthetic solar disk with a well-defined circular spot."""
    size = 400
    img = np.full((size, size), 200, dtype=np.uint8)
    
    # Off-disk space zeroed
    y, x = np.ogrid[:size, :size]
    r_dist = np.sqrt((x - 200) ** 2 + (y - 200) ** 2)
    img[r_dist > 180] = 0

    # Draw known circular spot centered at (150, 150)
    # Penumbra: radius 15, intensity 140 (0.70 of 200)
    cv2.circle(img, (150, 150), 15, 140, -1)
    # Umbra: radius 7, intensity 60 (0.30 of 200)
    cv2.circle(img, (150, 150), 7, 60, -1)

    return img


def test_feature_calculations(detector, synthetic_disk_with_spot):
    output = detector.detect(synthetic_disk_with_spot)

    assert output.is_spotless is False
    assert output.total_detected_count >= 1

    main_region = output.regions[0]

    # 1. Area: circle of radius ~15 has area pi * 15^2 ~ 706 px
    assert 550 < main_region.area_pixels < 850

    # 2. Perimeter: 2 * pi * 15 ~ 94 px
    assert 70 < main_region.perimeter_pixels < 120

    # 3. Circularity: circle should have circularity near 1.0 (typically > 0.75 on discrete pixel grid)
    assert main_region.circularity > 0.70

    # 4. Centroid: centered at (150, 150) within 2 pixels
    cx, cy = main_region.centroid
    assert abs(cx - 150.0) < 3.0
    assert abs(cy - 150.0) < 3.0

    # 5. Bounding box: centered at 150 with radius 15 -> (135, 135, 30, 30)
    bx, by, bw, bh = main_region.bbox
    assert abs(bx - 135) < 4
    assert abs(by - 135) < 4
    assert abs(bw - 30) < 6
    assert abs(bh - 30) < 6

    # 6. Mean intensity: between umbra (60) and penumbra (140)
    assert 60 <= main_region.mean_intensity <= 145

    # 7. Unique region ID
    assert main_region.region_id.startswith("AR-")


def test_circularity_formula(detector):
    # Perfect circle: Area = pi * r^2, Perimeter = 2 * pi * r
    # C = 4 * pi * (pi * r^2) / (4 * pi^2 * r^2) = 1.0
    r = 20.0
    area_circle = np.pi * (r ** 2)
    perim_circle = 2.0 * np.pi * r
    circ_circle = detector.calculate_circularity(area_circle, perim_circle)
    assert abs(circ_circle - 1.0) < 0.01

    # Highly elongated rectangle (100 x 2)
    # Area = 200, Perimeter = 204
    # C = 4 * pi * 200 / (204^2) ~ 2513 / 41616 ~ 0.06
    area_rect = 200.0
    perim_rect = 204.0
    circ_rect = detector.calculate_circularity(area_rect, perim_rect)
    assert circ_rect < 0.10

    # Degenerate cases
    assert detector.calculate_circularity(0.0, 10.0) == 0.0
    assert detector.calculate_circularity(100.0, 0.0) == 0.0


def test_spotless_image_handling(detector):
    """Test behavior on a clean solar disk with no dark spots (solar minimum)."""
    size = 300
    clean_disk = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(clean_disk, (150, 150), 120, 200, -1)

    output = detector.detect(clean_disk)

    assert output.is_spotless is True
    assert output.total_detected_count == 0
    assert len(output.regions) == 0
    assert output.binary_mask.shape == (size, size)
    assert np.count_nonzero(output.binary_mask) == 0
    assert output.annotated_image.shape == (size, size, 3)

    # Exporting empty results should not error
    df = output.to_dataframe()
    assert df.empty
    json_str = output.to_json()
    parsed = json.loads(json_str)
    assert parsed["is_spotless"] is True
    assert parsed["total_detected"] == 0


def test_empty_image_error_handling(detector):
    with pytest.raises(DetectionError):
        detector.detect(None)

    with pytest.raises(DetectionError):
        detector.detect(np.array([]))


def test_roi_patch_extraction(detector, synthetic_disk_with_spot):
    output = detector.detect(synthetic_disk_with_spot, extract_patches=True)
    assert len(output.regions) > 0

    reg = output.regions[0]
    assert reg.patch is not None
    # Patch must have 3 dimensions and non-zero size
    assert reg.patch.ndim == 3
    assert reg.patch.shape[0] > reg.bbox[3]  # includes padding
    assert reg.patch.shape[1] > reg.bbox[2]


def test_structured_export(detector, synthetic_disk_with_spot):
    output = detector.detect(synthetic_disk_with_spot)
    assert len(output.regions) > 0

    # 1. Dict export
    d = output.regions[0].to_dict()
    assert "region_id" in d
    assert "area_pixels" in d
    assert "circularity" in d
    assert "mean_intensity" in d

    # 2. DataFrame export
    df = output.to_dataframe()
    assert len(df) == len(output.regions)
    assert "circularity" in df.columns
    assert "area_uhem" in df.columns

    # 3. JSON export
    j = output.to_json()
    parsed = json.loads(j)
    assert "regions" in parsed
    assert "disk_geometry" in parsed
    assert len(parsed["regions"]) == len(output.regions)


def test_real_sdo_image_detection(detector):
    real_images = list(Path("data/sample_images").glob("sdo_hmi_ar3664_20240510.jpg"))
    if not real_images:
        pytest.skip("Real SDO image not found in data/sample_images")

    img = cv2.imread(str(real_images[0]))
    output = detector.detect(img)

    assert output.is_spotless is False
    assert output.total_detected_count >= 1
    assert output.confirmed_sunspot_count >= 1

    # AR3664 should be the largest detected active region
    largest = output.regions[0]
    assert largest.area_uhem > 1000.0
    assert largest.scientific_status == "Confirmed Sunspot Group"
    assert largest.confidence >= 0.80
    assert largest.has_penumbra is True
    assert largest.patch is not None
