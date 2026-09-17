"""
Test suite for LimbDarkeningCorrector.
"""

import numpy as np
import pytest

from src.disk_detector import SolarDiskDetector, SolarDiskGeometry
from src.limb_darkening import LimbDarkeningCorrector
from src.config import LimbDarkeningConfig


def test_limb_darkening_flattening():
    # Construct an image with exact linear limb darkening
    size = 400
    cx, cy, r = 200.0, 200.0, 160.0
    y, x = np.ogrid[:size, :size]
    r_dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r_norm = np.clip(r_dist / r, 0.0, 1.0)
    mu = np.sqrt(np.maximum(0.0, 1.0 - (r_norm ** 2)))

    # I(mu) = 200 * [1 - 0.60 * (1 - mu)]
    u = 0.60
    i_0 = 200.0
    raw_disk = np.zeros((size, size), dtype=np.uint8)
    in_disk = r_dist <= r
    raw_disk[in_disk] = (i_0 * (1.0 - u * (1.0 - mu[in_disk]))).astype(np.uint8)

    # Detector
    detector = SolarDiskDetector()
    disk = detector.detect(raw_disk)

    # Corrector
    corrector = LimbDarkeningCorrector(LimbDarkeningConfig(u_coefficient=0.60))
    res = corrector.correct(raw_disk, disk)

    assert res.flattened_image.shape == (size, size)
    assert res.flattened_uint8.dtype == np.uint8

    # The flattened image inside 0.2 < r/R < 0.8 should be uniform
    inner_annulus = (res.radial_distance_map >= 0.2) & (res.radial_distance_map <= 0.8) & (disk.effective_mask > 0)
    flattened_std = float(np.std(res.flattened_image[inner_annulus]))
    raw_std = float(np.std(raw_disk[inner_annulus]))

    # Flattened std must be substantially less than raw std
    assert flattened_std < raw_std
