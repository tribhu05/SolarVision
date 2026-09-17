"""
Test suite for SunspotSegmenter.
"""

import numpy as np
import cv2
import pytest

from src.disk_detector import SolarDiskDetector
from src.limb_darkening import LimbDarkeningCorrector
from src.segmentation import SunspotSegmenter
from src.config import SegmentationConfig


def test_sunspot_segmentation_umbra_penumbra():
    size = 400
    img = np.full((size, size), 200, dtype=np.uint8)
    
    # Mask off-disk area
    y, x = np.ogrid[:size, :size]
    r_dist = np.sqrt((x - 200) ** 2 + (y - 200) ** 2)
    img[r_dist > 180] = 0

    # Draw a simulated sunspot at (150, 150)
    # Penumbra radius 15, intensity 140 (0.70 of 200)
    cv2.circle(img, (150, 150), 15, 130, -1)
    # Umbra radius 6, intensity 70 (0.35 of 200)
    cv2.circle(img, (150, 150), 6, 70, -1)

    disk = SolarDiskDetector().detect(img)
    limb = LimbDarkeningCorrector().correct(img, disk)
    segmenter = SunspotSegmenter(SegmentationConfig(min_sunspot_area_pixels=10))

    seg_res = segmenter.segment(limb, disk)

    assert len(seg_res.regions) >= 1
    main_reg = seg_res.regions[0]
    
    assert main_reg.umbra_area_px > 0
    assert main_reg.penumbra_area_px > 0
    assert main_reg.total_area_px == main_reg.umbra_area_px + main_reg.penumbra_area_px
    assert np.count_nonzero(seg_res.umbra_mask) > 0
    assert np.count_nonzero(seg_res.penumbra_mask) > 0
