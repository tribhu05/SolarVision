"""
Test suite for SolarDiskDetector.
"""

import numpy as np
import cv2
import pytest

from src.disk_detector import SolarDiskDetector, SolarDiskGeometry
from src.config import DiskDetectionConfig


def test_solar_disk_detection_synthetic():
    detector = SolarDiskDetector(DiskDetectionConfig())

    # Create a 512x512 image with a centered bright disk of radius 200
    img = np.zeros((512, 512), dtype=np.uint8)
    cv2.circle(img, (256, 256), 200, 200, -1)

    result = detector.detect(img)

    assert result.is_valid is True
    assert result.confidence > 0.7
    assert abs(result.center_x - 256) < 3.0
    assert abs(result.center_y - 256) < 3.0
    assert abs(result.radius - 200) < 4.0
    assert result.mask.shape == (512, 512)
    assert np.count_nonzero(result.mask) > 0


def test_solar_disk_effective_mask_margin():
    cfg = DiskDetectionConfig(edge_margin_fraction=0.05)
    detector = SolarDiskDetector(cfg)

    img = np.zeros((400, 400), dtype=np.uint8)
    cv2.circle(img, (200, 200), 150, 220, -1)

    result = detector.detect(img)

    assert result.is_valid is True
    full_count = np.count_nonzero(result.mask)
    eff_count = np.count_nonzero(result.effective_mask)

    # Effective mask must be strictly smaller than full mask due to margin
    assert eff_count < full_count
