"""
Test suite for SolarImagePreprocessor.
Tests classical CV preprocessing operations: loading, color conversion,
background isolation, noise reduction, contrast enhancement, morphology,
and error handling.
"""

from pathlib import Path
import cv2
import numpy as np
import pytest

from src.preprocessor import (
    SolarImagePreprocessor,
    PreprocessingResult,
    InvalidImageInputError,
)
from src.config import PreprocessingConfig, SolarVisionConfig
from src.disk_detector import SolarDiskDetector


@pytest.fixture
def preprocessor():
    cfg = SolarVisionConfig()
    return SolarImagePreprocessor(full_config=cfg)


@pytest.fixture
def synthetic_disk():
    """Create a 512x512 synthetic solar disk image with a dark sunspot."""
    size = 512
    img = np.zeros((size, size), dtype=np.uint8)
    cx, cy, r = 256, 256, 220
    # Solar disk with quiet-Sun intensity 200
    cv2.circle(img, (cx, cy), r, 200, -1)
    # Add a dark spot (umbra 50, penumbra 120) at (200, 200)
    cv2.circle(img, (200, 200), 16, 120, -1)
    cv2.circle(img, (200, 200), 7, 50, -1)
    return img


def test_load_image_numpy(preprocessor, synthetic_disk):
    # Grayscale array
    loaded_gray = preprocessor.load_image(synthetic_disk)
    assert loaded_gray.shape == (512, 512)

    # BGR array
    bgr = cv2.cvtColor(synthetic_disk, cv2.COLOR_GRAY2BGR)
    loaded_bgr = preprocessor.load_image(bgr)
    assert loaded_bgr.shape == (512, 512, 3)


def test_load_image_bytes(preprocessor, synthetic_disk):
    success, enc = cv2.imencode(".png", synthetic_disk)
    assert success is True
    loaded = preprocessor.load_image(enc.tobytes())
    assert loaded.shape[:2] == (512, 512)


def test_load_image_file_path(preprocessor, tmp_path, synthetic_disk):
    file_path = tmp_path / "test_solar.png"
    cv2.imwrite(str(file_path), synthetic_disk)

    loaded = preprocessor.load_image(file_path)
    assert loaded.shape[:2] == (512, 512)


def test_load_image_invalid_inputs(preprocessor):
    # None
    with pytest.raises(InvalidImageInputError):
        preprocessor.load_image(None)

    # Empty array
    with pytest.raises(InvalidImageInputError):
        preprocessor.load_image(np.array([]))

    # Empty bytes
    with pytest.raises(InvalidImageInputError):
        preprocessor.load_image(b"")

    # Nonexistent file
    with pytest.raises(InvalidImageInputError):
        preprocessor.load_image("nonexistent_path_xyz.jpg")


def test_convert_to_grayscale(preprocessor, synthetic_disk):
    bgr = cv2.cvtColor(synthetic_disk, cv2.COLOR_GRAY2BGR)
    gray = preprocessor.convert_to_grayscale(bgr)

    assert gray.ndim == 2
    assert gray.shape == (512, 512)

    # Grayscale input passthrough
    gray_passthrough = preprocessor.convert_to_grayscale(gray)
    assert gray_passthrough.ndim == 2
    assert np.array_equal(gray, gray_passthrough)


def test_isolate_solar_disk(preprocessor, synthetic_disk):
    disk = SolarDiskDetector().detect(synthetic_disk)
    isolated = preprocessor.isolate_solar_disk(synthetic_disk, disk)

    assert isolated.shape == synthetic_disk.shape
    # Pixels outside effective mask must be strictly 0
    assert np.all(isolated[disk.effective_mask == 0] == 0)
    # Pixels inside effective mask must retain intensity
    assert np.all(isolated[disk.effective_mask > 0] == synthetic_disk[disk.effective_mask > 0])


def test_noise_reduction_bilateral(preprocessor, synthetic_disk):
    # Add random noise to disk
    noisy = synthetic_disk.copy().astype(np.int16)
    noise = np.random.normal(0, 5, noisy.shape).astype(np.int16)
    noisy = np.clip(noisy + noise, 0, 255).astype(np.uint8)

    disk = SolarDiskDetector().detect(synthetic_disk)
    denoised = preprocessor.apply_noise_reduction(noisy, disk.effective_mask)

    assert denoised.shape == noisy.shape
    assert denoised.dtype == np.uint8

    # Noise variation should decrease in quiet-Sun region
    center_patch_noisy = noisy[240:270, 240:270]
    center_patch_denoised = denoised[240:270, 240:270]
    assert np.std(center_patch_denoised) < np.std(center_patch_noisy)


def test_contrast_enhancement_clahe(preprocessor, synthetic_disk):
    disk = SolarDiskDetector().detect(synthetic_disk)
    enhanced = preprocessor.apply_contrast_enhancement(synthetic_disk, disk.effective_mask)

    assert enhanced.shape == synthetic_disk.shape
    assert enhanced.dtype == np.uint8
    # Background must remain strictly 0
    assert np.all(enhanced[disk.effective_mask == 0] == 0)


def test_blackhat_dark_feature_extraction(preprocessor, synthetic_disk):
    disk = SolarDiskDetector().detect(synthetic_disk)
    bhat = preprocessor.apply_blackhat_transformation(synthetic_disk, disk.effective_mask)

    assert bhat.shape == synthetic_disk.shape
    # At the dark spot location (200, 200), Black-Hat should produce a distinct positive peak
    spot_val = bhat[200, 200]
    quiet_val = bhat[256, 256]
    assert spot_val > quiet_val
    assert spot_val > 50  # Strong response at dark spot core


def test_full_preprocessing_pipeline_execution(preprocessor, synthetic_disk):
    res = preprocessor.process(synthetic_disk)

    assert isinstance(res, PreprocessingResult)
    assert res.original_shape == (512, 512)
    assert res.preprocessed_image.shape == (512, 512)
    assert res.solar_disk.is_valid is True
    assert res.visuals.composite_panel.size > 0
    assert res.metadata["original_width"] == 512
    assert res.metadata["original_height"] == 512


def test_real_sdo_image_preprocessing(preprocessor):
    real_files = list(Path("data/sample_images").glob("sdo_hmi_ar3664_*.jpg"))
    if not real_files:
        pytest.skip("No real SDO sample images found in data/sample_images")

    res = preprocessor.process(real_files[0])

    assert res.original_shape == (1024, 1024)
    assert res.preprocessed_image.shape == (1024, 1024)
    assert res.solar_disk.radius > 400.0
    assert res.quiet_sun_level > 100.0
    assert res.visuals.composite_panel.shape[0] > 0
